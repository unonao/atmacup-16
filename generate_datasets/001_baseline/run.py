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

import utils
from utils.load import load_label_data, load_log_data, load_session_data
from utils.metrics import calculate_metrics


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


def concat_session_feature(cfg, mode: str, candidate_df: pl.DataFrame):
    """
    session_id, seq_no, yad_no に yado.csv を結合して集約し、セッションに関する特徴量を作成する
    """
    pass


def concat_candidate_feature(cfg, mode: str, candidate_df: pl.DataFrame):
    """
    candidateの特徴量を抽出する
    """
    pass


def concat_session_candidate_feature(cfg, mode: str, candidate_df: pl.DataFrame):
    """
    session中の特徴とcandidateの関係性を特徴量として抽出する
    例: session中におけるcandidateの出現回数(割合)、candidateと同一地域のものを見た回数(割合)
    """
    pass


def concat_label_fold(cfg, mode: str, candidate_df: pl.DataFrame):
    """
    fold情報とlabel情報を結合する
    """
    pass


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"
    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.cand_unsupervised_dir) / exp_name
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)


if __name__ == "__main__":
    main()
