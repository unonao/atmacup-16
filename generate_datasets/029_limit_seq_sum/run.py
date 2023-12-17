import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import OrdinalEncoder

from utils.data import convert_to_32bit
from utils.load import load_label_data, load_log_data, load_yad_data
from utils.logger import get_logger

numerical_cols = [  # あとで書き換えるので注意
    "total_room_cnt",
    "wireless_lan_flg",
    "onsen_flg",
    "kd_stn_5min",
    "kd_bch_5min",
    "kd_slp_5min",
]

categorical_cols = [
    "yad_type",
    "wid_cd",
    "ken_cd",
    "lrg_cd",
    "sml_cd",
]

logger = None
ordinal_encoder = None


def load_limit_log_data(cfg, mode: str):
    log_df = load_log_data(Path(cfg.dir.data_dir), mode)
    if cfg.exp.limit_seq is not None:
        log_df = (
            log_df.group_by("session_id")
            .agg(
                pl.col("seq_no").slice(-cfg.exp.limit_seq, cfg.exp.limit_seq),
                pl.col("yad_no").slice(-cfg.exp.limit_seq, cfg.exp.limit_seq),
            )
            .explode(["yad_no", "seq_no"])
            .sort(by="session_id")
        )
    return log_df


def load_yad_data_with_features(cfg):
    global numerical_cols
    yad_df = load_yad_data(Path(cfg.dir.data_dir))
    original_cols = yad_df.columns
    for path in cfg.exp.yad_feature_paths:
        feature_df = pl.read_parquet(path)
        yad_df = yad_df.join(feature_df, on="yad_no")
    new_cols = [col for col in yad_df.columns if col not in original_cols]
    numerical_cols = list(set(numerical_cols) | set(new_cols))
    return yad_df


def load_and_union_candidates(cfg, mode: str):
    # logデータのsession中のyad_noを候補に加える
    log_df = load_limit_log_data(cfg, mode)
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

    if cfg.debug:
        df = df.with_columns(pl.col("candidates").list.head(2).alias("candidates"))

    # リストを展開
    candidate_df = df.explode("candidates")

    # セッション最後のyad_noを除外
    last_df = (
        load_limit_log_data(cfg, mode)
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

    # seq_len

    return candidate_df


def concat_session_feature(cfg, mode: str, candidate_df: pl.DataFrame):
    """
    session_id, seq_no, yad_no に yado.csv を結合して集約し、セッションに関する特徴量を作成する
    """
    log_df = load_limit_log_data(cfg, mode)
    yad_df = load_yad_data_with_features(cfg)
    log_yad_df = log_df.join(yad_df.fill_null(0), on="yad_no")
    log_yad_df = log_yad_df.group_by(by="session_id").agg(
        [pl.sum(col).name.suffix("_session_sum") for col in numerical_cols]
        + [pl.min(col).name.suffix("_session_min") for col in numerical_cols]
        + [pl.max(col).name.suffix("_session_max") for col in numerical_cols]
        + [pl.std(col).name.suffix("_session_std") for col in numerical_cols]
        + [pl.mean(col).name.suffix("_session_mean") for col in numerical_cols]
        + [(pl.max("seq_no") + 1).alias("session_count")]
    )

    candidate_df = candidate_df.join(log_yad_df, on="session_id")

    return candidate_df


def concat_candidate_feature(cfg, mode: str, candidate_df: pl.DataFrame):
    """
    candidateの特徴量を抽出する
    """
    original_cols = candidate_df.columns

    yad_df = load_yad_data_with_features(cfg)
    candidate_yad_df = candidate_df.join(
        yad_df.select(["yad_no"] + numerical_cols + categorical_cols),
        left_on="candidates",
        right_on="yad_no",
    )

    new_cols = [col for col in candidate_yad_df.columns if col not in original_cols]
    print(f"new_cols: {new_cols}")
    return candidate_yad_df


def concat_session_candidate_feature(cfg, mode: str, candidate_df: pl.DataFrame):
    """
    session中の特徴とcandidateの関係性を特徴量として抽出する
    例: session中におけるcandidateの出現回数(割合)、candidateと同一地域のものを見た回数(割合)
    """
    # セッションと候補の差や割合の特徴量を追加
    candidate_df = candidate_df.with_columns(
        [
            (pl.col(f"{col}_session_mean") / pl.col(col)).alias(
                f"{col}_session_mean_div"
            )
            for col in numerical_cols
        ]
    )

    # 同じ candidate の出現回数
    log_df = load_limit_log_data(cfg, mode)
    tmp = (
        log_df.group_by(by=["session_id", "yad_no"])
        .agg(pl.count("session_id").alias("appear_count"))
        .with_columns(
            (
                pl.col("appear_count") / pl.col("appear_count").sum().over("session_id")
            ).alias("appear_rate"),
            pl.col("yad_no").alias("candidates"),
        )
    )
    candidate_df = candidate_df.join(
        tmp.select(["session_id", "candidates", "appear_count", "appear_rate"]),
        on=["session_id", "candidates"],
        how="left",
    )

    # 最初or最後から2番目に出現したかどうか
    for i in [0, -2]:
        log_df = load_limit_log_data(cfg, mode)
        log_df = (
            log_df.sort(by=["session_id", "seq_no"])
            .group_by("session_id")
            .agg(pl.col("yad_no").alias("candidates"))
            .with_columns(
                pl.col("candidates").list.get(i),
                pl.lit(1.0).alias(f"appear_last{i}_yad_no"),
            )
        )
        candidate_df = candidate_df.join(
            log_df, on=["session_id", "candidates"], how="left"
        )

    # 同じ categorical の出現回数
    ## (series_id, categorical) でグループ化して、session_id ごとに出現回数を集計する
    log_df = load_limit_log_data(cfg, mode)
    yad_df = load_yad_data(Path(cfg.dir.data_dir))
    log_yad_df = log_df.join(yad_df.fill_null(0), on="yad_no")
    for col in categorical_cols:
        tmp = (
            log_yad_df.group_by(by=["session_id", col])
            .agg(pl.count("session_id").alias(f"same_{col}_count"))
            .with_columns(
                pl.col(f"same_{col}_count").sum().over("session_id").alias("seq_sum")
            )
            .with_columns(
                (pl.col(f"same_{col}_count") / pl.col("seq_sum")).alias(
                    f"same_{col}_rate"
                )
            )
        )
        candidate_df = candidate_df.join(
            tmp.select(["session_id", col, f"same_{col}_count", f"same_{col}_rate"]),
            on=["session_id", col],
            how="left",
        )

    # location ごとの transition prob を追加
    for location_col in ["sml_cd", "lrg_cd", "ken_cd", "wid_cd"]:
        prob_col = f"transition_prob_{location_col}"
        loc2loc_prob_df = pl.read_parquet(
            Path(cfg.exp.feat_transition_prob_location_dir)
            / f"{location_col}2{location_col}_feature.parquet"
        )
        col = f"transition_prob_{location_col}"
        log_df = load_limit_log_data(cfg, mode)
        yad_df = load_yad_data(Path(cfg.dir.data_dir))
        log_df = log_df.join(yad_df.select(["yad_no", location_col]), on="yad_no")
        log_df = (
            log_df.sort(by="session_id").with_columns(
                pl.col(location_col).alias(f"from_{location_col}")
            )
        ).select(["session_id", f"from_{location_col}"])
        log_df = (
            log_df.join(loc2loc_prob_df, on=f"from_{location_col}")
            .group_by(["session_id", f"to_{location_col}"])
            .agg(pl.sum(prob_col).alias(prob_col + "_from_all"))
        )
        candidate_df = candidate_df.join(
            log_df,
            left_on=["session_id", location_col],
            right_on=["session_id", f"to_{location_col}"],
            how="left",
        ).drop("from_yad_no")

    # transition probを追加
    yad2yad_prob = pl.read_parquet(cfg.exp.transition_prob_path)
    log_df = load_limit_log_data(cfg, mode)
    last_log_df = (
        log_df.group_by("session_id")
        .agg(pl.all().sort_by("seq_no").last())
        .sort(by="session_id")
        .with_columns(pl.col("yad_no").alias("from_yad_no"))
    ).select(["session_id", "from_yad_no"])
    last_log_prob_df = last_log_df.join(yad2yad_prob, on="from_yad_no")
    candidate_df = candidate_df.join(
        last_log_prob_df,
        left_on=["session_id", "candidates"],
        right_on=["session_id", "to_yad_no"],
        how="left",
    ).drop("from_yad_no")

    # last 以外からのtransition probも追加
    yad2yad_prob = pl.read_parquet(cfg.exp.transition_prob_path)
    prob_col = "transition_prob_transition_prob_fix/base"
    log_df = load_limit_log_data(cfg, mode)
    log_df = (
        log_df.sort(by="session_id").with_columns(pl.col("yad_no").alias("from_yad_no"))
    ).select(["session_id", "from_yad_no"])
    log_df = (
        log_df.join(yad2yad_prob, on="from_yad_no")
        .group_by(["session_id", "to_yad_no"])
        .agg(pl.sum(prob_col).alias(prob_col + "_from_all"))
    )
    candidate_df = candidate_df.join(
        log_df,
        left_on=["session_id", "candidates"],
        right_on=["session_id", "to_yad_no"],
        how="left",
    ).drop("from_yad_no")

    # last 以外からのtransition probも追加(transition_prob_all_path)
    yad2yad_prob = pl.read_parquet(cfg.exp.transition_prob_all_path)
    prob_col = "transition_prob_transition_prob_all_fix/base"
    log_df = load_limit_log_data(cfg, mode)
    log_df = (
        log_df.sort(by="session_id").with_columns(pl.col("yad_no").alias("from_yad_no"))
    ).select(["session_id", "from_yad_no"])
    log_df = (
        log_df.join(yad2yad_prob, on="from_yad_no")
        .group_by(["session_id", "to_yad_no"])
        .agg(pl.sum(prob_col).alias(prob_col + "_from_all"))
    )
    candidate_df = candidate_df.join(
        log_df,
        left_on=["session_id", "candidates"],
        right_on=["session_id", "to_yad_no"],
        how="left",
    ).drop("from_yad_no")

    # last 以外からのtransition probも追加(transition_prob_bidirect_all_path)
    yad2yad_prob = pl.read_parquet(cfg.exp.transition_prob_bidirect_all_path)
    prob_col = "transition_prob_transition_prob_bidirect_all_fix/base"
    log_df = load_limit_log_data(cfg, mode)
    log_df = (
        log_df.sort(by="session_id").with_columns(pl.col("yad_no").alias("from_yad_no"))
    ).select(["session_id", "from_yad_no"])
    log_df = (
        log_df.join(yad2yad_prob, on="from_yad_no")
        .group_by(["session_id", "to_yad_no"])
        .agg(pl.sum(prob_col).alias(prob_col + "_from_all"))
    )
    candidate_df = candidate_df.join(
        log_df,
        left_on=["session_id", "candidates"],
        right_on=["session_id", "to_yad_no"],
        how="left",
    ).drop("from_yad_no")

    # 確率の和を作成
    prob_cols = [
        col for col in candidate_df.columns if col.startswith("transition_prob")
    ]
    candidate_df = candidate_df.with_columns(
        pl.sum_horizontal(prob_cols).alias("transition_prob_sum")
    )

    """
    """
    # last 以外からのtransition probも追加(prob_matrix_path)
    yad2yad_prob = pl.read_parquet(cfg.exp.prob_matrix_path)
    prob_col = "transition_prob"
    log_df = load_limit_log_data(cfg, mode)
    log_df = (
        log_df.sort(by="session_id").with_columns(pl.col("yad_no").alias("from_yad_no"))
    ).select(["session_id", "from_yad_no"])
    log_df = (
        log_df.join(yad2yad_prob, on="from_yad_no")
        .group_by(["session_id", "to_yad_no"])
        .agg(pl.sum(prob_col).alias(prob_col + "_prob_matrix"))
    )
    candidate_df = candidate_df.join(
        log_df,
        left_on=["session_id", "candidates"],
        right_on=["session_id", "to_yad_no"],
        how="left",
    ).drop("from_yad_no")

    limit_seq = 10
    if cfg.exp.limit_seq is not None:
        limit_seq = cfg.exp.limit_seq
    # last 以外からのtransition probも追加(prob_matrix_path)
    yad2yad_prob = pl.read_parquet(cfg.exp.prob_matrix_path)
    prob_col = "transition_prob"
    log_df = load_limit_log_data(cfg, mode)
    log_df = (
        log_df.sort(by=["session_id", "seq_no"]).with_columns(
            [
                pl.col("yad_no").shift(si).over("session_id").alias(f"yad_no_{si}")
                for si in range(limit_seq)
            ]
        )
    ).drop(["yad_no"])
    log_df = log_df.group_by("session_id").agg(pl.all().last()).sort(by="session_id")
    for si in range(limit_seq):
        tmp = log_df.join(
            yad2yad_prob, left_on=f"yad_no_{si}", right_on="from_yad_no"
        ).with_columns(pl.col(prob_col).alias(prob_col + f"_prob_matrix_{si}"))

        candidate_df = candidate_df.join(
            tmp.select(["session_id", "to_yad_no", prob_col + f"_prob_matrix_{si}"]),
            left_on=["session_id", "candidates"],
            right_on=["session_id", "to_yad_no"],
            how="left",
        ).drop("from_yad_no")

    return candidate_df


def make_datasets(cfg, mode: str):
    logger.info(f"make_datasets: {mode}")

    candidate_df = load_and_union_candidates(cfg, mode)
    logger.info(f"load_and_union_candidates: {candidate_df.shape}")

    candidate_df = concat_label_fold(cfg, mode, candidate_df)
    logger.info(f"concat_label_fold: {candidate_df.shape}")

    candidate_df = concat_session_feature(cfg, mode, candidate_df)
    logger.info(f"concat_session_feature: {candidate_df.shape}")

    candidate_df = concat_candidate_feature(cfg, mode, candidate_df)
    logger.info(f"concat_candidate_feature: {candidate_df.shape}")

    candidate_df = concat_session_candidate_feature(cfg, mode, candidate_df)
    logger.info(f"concat_session_candidate_feature: {candidate_df.shape}")

    # カテゴリカル変数を数値に変換
    transformed = ordinal_encoder.transform(candidate_df[categorical_cols].to_numpy())
    candidate_df = candidate_df.with_columns(
        [
            pl.Series(name=col, values=transformed[:, i])
            for i, col in enumerate(categorical_cols)
        ]
    )
    # 型変換
    candidate_df = convert_to_32bit(candidate_df)

    # 内容確認
    if mode == "train":
        logger.info("train_df")
        logger.info(candidate_df.head(10))
        logger.info(candidate_df.shape)
        logger.info(candidate_df.columns)
        avg_candidates = len(candidate_df) / candidate_df["session_id"].unique().len()
        recall_rate = (
            candidate_df.filter(pl.col("original") == True)["label"].sum()
            / candidate_df["session_id"].unique().len()
        )
        logger.info(f"avg_candidates: {avg_candidates}, recall_rate: {recall_rate}")
        for i in range(1, 11):
            cad_df = candidate_df.filter(pl.col("session_count") == i)
            logger.info(f"session_count: {i}, len: {len(cad_df)}")
            logger.info(
                f"session_count: {i}, avg_candidates: {len(cad_df) / cad_df['session_id'].unique().len()}"
            )
            logger.info(
                f"session_count: {i}, recall_rate: {cad_df.filter(pl.col('original') == True)['label'].sum() / cad_df['session_id'].unique().len()}"
            )

    return candidate_df


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"
    output_path = Path(cfg.dir.datasets_dir) / exp_name
    os.makedirs(output_path, exist_ok=True)

    global logger
    logger = get_logger(__name__, file_path=output_path / "run.log")
    logger.info(f"exp_name: {exp_name}")
    logger.info(f"output_path: {output_path}")

    # encoder
    global ordinal_encoder
    yad_df = load_yad_data(Path(cfg.dir.data_dir))
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    ordinal_encoder.fit(yad_df[categorical_cols].to_numpy())

    # train
    train_df = make_datasets(cfg, "train")
    train_df.write_parquet(output_path / "train.parquet", use_pyarrow=True)

    # test
    test_df = make_datasets(cfg, "test")
    test_df.write_parquet(output_path / "test.parquet", use_pyarrow=True)


if __name__ == "__main__":
    main()
