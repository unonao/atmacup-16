"""
候補を結合して最終的な候補を作る&特徴量を作成することで、LightGBMで学習・推論するためのデータセットを生成する

trainに対しては抜けているラベルのデータがあれば追加しておく（ただし、検証時にはこのデータは使わないように、フラグを立てておく）
"""

import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils.data import convert_to_32bit
from utils.load import load_label_data, load_log_data, load_yad_data

numerical_col = [
    "total_room_cnt",
    "wireless_lan_flg",
    "onsen_flg",
    "kd_stn_5min",
    "kd_bch_5min",
    "kd_slp_5min",
]
"""
categorical_col = [
    "yad_type",
    "wid_cd",
    "ken_cd",
    "lrg_cd",
    "sml_cd",
]
"""


def load_and_union_candidates(cfg, mode: str):
    # logデータのsession中のyad_noを候補に加える
    log_df = load_log_data(Path(cfg.dir.data_dir), mode)
    df = log_df.group_by("session_id").agg(pl.col("yad_no").alias("candidates"))
    dfs = [df]
    for candidate_info in cfg.exp.candidate_info_list:
        df = pl.read_parquet(Path(candidate_info["dir"]) / f"{mode}_candidate.parquet")
        df = df.with_columns(
            pl.col("candidates")
            .list.head(candidate_info["max_num_candidates"])
            .alias("candidates")
        ).filter(pl.col("candidates").list.len() > 0)

        dfs.append(df)
    df = pl.concat(dfs)
    df = (
        df.group_by("session_id")
        .agg(pl.col("candidates").flatten())
        .with_columns(pl.col("candidates").list.unique())
    ).select(["session_id", "candidates"])

    # リストを展開
    candidate_df = df.explode("candidates")

    # セッション最後のyad_noを除外
    last_df = (
        load_log_data(Path(cfg.dir.data_dir), mode)
        .group_by("session_id")
        .agg(pl.col("yad_no").last().alias("candidates"))
        .with_columns(pl.lit(True).alias("last"))
        .sort(by="session_id")
    )
    candidate_df = (
        candidate_df.join(last_df, on=["session_id", "candidates"], how="left")
        .filter(pl.col("last").is_null())
        .drop("last")
    )
    return candidate_df


def concat_label_fold(cfg, mode: str, candidate_df):
    """
    train に対して original, label, fold を付与する
    validationのスコア計算時にはoriginalを外して計算を行う
    """
    if mode == "train":
        candidate_df = (
            pl.concat(
                [
                    candidate_df.with_columns(
                        pl.lit(True).alias("original"), pl.lit(False).alias("label")
                    ),
                    load_label_data(Path(cfg.dir.data_dir))
                    .with_columns(
                        pl.col("yad_no").alias("candidates"),
                        pl.lit(False).alias("original"),
                        pl.lit(True).alias("label"),
                    )
                    .drop("yad_no"),
                ]
            )
            .group_by(["session_id", "candidates"])
            .agg(pl.sum("original"), pl.sum("label"))
        )
        fold_df = pl.read_parquet(cfg.exp.fold_path)
        candidate_df = candidate_df.join(fold_df, on="session_id")
    return candidate_df


def concat_session_feature(cfg, mode: str, candidate_df: pl.DataFrame):
    """
    # TODO: categorical_colの情報もあとで追加する
    session_id, seq_no, yad_no に yado.csv を結合して集約し、セッションに関する特徴量を作成する
    """
    log_df = load_log_data(Path(cfg.dir.data_dir), mode)
    yad_df = load_yad_data(Path(cfg.dir.data_dir))
    log_yad_df = log_df.join(yad_df.fill_null(0), on="yad_no")
    log_yad_df = log_yad_df.group_by(by="session_id").agg(
        [pl.sum(col).name.suffix("_session_sum") for col in numerical_col]
        + [pl.min(col).name.suffix("_session_min") for col in numerical_col]
        + [pl.max(col).name.suffix("_session_max") for col in numerical_col]
        + [pl.std(col).name.suffix("_session_std") for col in numerical_col]
    )

    candidate_df = candidate_df.join(log_yad_df, on="session_id")

    return candidate_df


def concat_candidate_feature(cfg, mode: str, candidate_df: pl.DataFrame):
    """
    # TODO: categorical_colの情報もあとで追加する
    candidateの特徴量を抽出する
    """
    yad_df = load_yad_data(Path(cfg.dir.data_dir))
    candidate_yad_df = candidate_df.join(
        yad_df.select(["yad_no"] + numerical_col),
        left_on="candidates",
        right_on="yad_no",
    )
    return candidate_yad_df


def concat_session_candidate_feature(cfg, mode: str, candidate_df: pl.DataFrame):
    """
    session中の特徴とcandidateの関係性を特徴量として抽出する
    例: session中におけるcandidateの出現回数(割合)、candidateと同一地域のものを見た回数(割合)
    """
    pass


def make_datasets(cfg, mode: str):
    print(f"make_datasets: {mode}")
    candidate_df = load_and_union_candidates(cfg, mode)
    print(f"candidate_df: {candidate_df.shape}")
    candidate_df = concat_label_fold(cfg, mode, candidate_df)
    print(f"candidate_df: {candidate_df.shape}")
    candidate_df = concat_session_feature(cfg, mode, candidate_df)
    print(f"candidate_df: {candidate_df.shape}")
    candidate_df = concat_candidate_feature(cfg, mode, candidate_df)
    print(f"candidate_df: {candidate_df.shape}")
    #  candidate_df = concat_session_candidate_feature(cfg, mode, candidate_df)

    candidate_df = convert_to_32bit(candidate_df)
    return candidate_df


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"
    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.datasets_dir) / exp_name
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # train
    train_df = make_datasets(cfg, "train")
    train_df.write_parquet(output_path / "train.parquet", use_pyarrow=True)

    print("train_df")
    print(train_df.head(10))
    print(train_df.shape)
    print(train_df.columns)

    # test
    test_df = make_datasets(cfg, "test")
    test_df.write_parquet(output_path / "test.parquet", use_pyarrow=True)


if __name__ == "__main__":
    main()
