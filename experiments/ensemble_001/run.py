import logging
import os
import pickle
import sys
import time
from pathlib import Path

import hydra
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import utils
import wandb
from utils.load import load_label_data, load_log_data, load_session_data
from utils.logger import get_logger
from utils.metrics import calculate_metrics
from wandb.lightgbm import log_summary, wandb_callback

logger = None


def make_eval_df(cfg, other_oof_df: pl.DataFrame, first_oof_df: pl.DataFrame):
    other_oof_df = other_oof_df.filter(pl.col("session_count") != 1).drop(
        "session_count"
    )
    first_oof_df = first_oof_df.filter(pl.col("session_count") == 1).drop(
        "session_count"
    )
    pred_df = pl.concat([other_oof_df, first_oof_df]).sort(
        by=["session_id", "pred"], descending=True
    )
    pred_candidates_df = pred_df.group_by("session_id").agg(pl.col("candidates"))
    train_label_df = load_label_data(Path(cfg.dir.data_dir))
    candidaates_df = pred_candidates_df.join(
        train_label_df, on="session_id", how="left"
    )
    return candidaates_df


def make_submission(cfg, other_test_df: pl.DataFrame, first_test_df: pl.DataFrame):
    other_test_df = other_test_df.filter(pl.col("session_count") != 1).drop(
        "session_count"
    )
    first_test_df = first_test_df.filter(pl.col("session_count") == 1).drop(
        "session_count"
    )
    pred_df = pl.concat([other_test_df, first_test_df]).sort(
        by=["session_id", "pred"], descending=True
    )
    session_df = load_session_data(Path(cfg.dir.data_dir), "test")
    pred_candidates_df = pred_df.group_by("session_id").agg(pl.col("candidates"))
    submission_df = (
        session_df.join(
            pred_candidates_df.with_columns(
                [
                    pl.col("candidates").list.get(i).alias(f"predict_{i}")
                    for i in range(10)
                ]
            ).drop("candidates"),
            on="session_id",
            how="left",
        )
        .fill_null(-1)
        .drop("session_id")
    )
    return submission_df


def concat_label_pred(cfg, first_df, mode):
    # 最後のyad_noだけを残す & labelを付与
    train_log_df = load_log_data(Path(cfg.dir.data_dir), "train")
    train_label_df = load_label_data(Path(cfg.dir.data_dir))
    train_last_log_label_df = (
        train_log_df.join(train_label_df, on="session_id", suffix="_label")
        .with_columns(
            (pl.col("seq_no").max().over("session_id") + 1).alias("session_count")
        )
        .filter(pl.col("seq_no") == pl.col("session_count") - 1)
    )
    # 実績ラベルからyad_noごとに良さそうな対象を探す
    label_pred_df = (
        train_last_log_label_df.group_by(["yad_no", "yad_no_label"])
        .agg(pl.col("yad_no").count().alias("pred"))
        .with_columns(pl.col("pred") * 100.0, pl.lit(1).alias("session_count"))
        .sort(by=["yad_no", "pred", "session_count"], descending=True)
    )

    # 予測値作成
    log_df = load_log_data(Path(cfg.dir.data_dir), mode)
    last_log_df = log_df.with_columns(
        (pl.col("seq_no").max().over("session_id") + 1).alias("session_count")
    ).filter(pl.col("seq_no") == pl.col("session_count") - 1)
    session_df = load_session_data(Path(cfg.dir.data_dir), mode)
    session_last_df = (
        session_df.join(
            last_log_df.select(["session_id", "yad_no", "session_count"]),
            on="session_id",
        )
        .filter(pl.col("session_count") == 1)
        .drop("session_count")
    )
    first_df_from_label = (
        session_last_df.join(label_pred_df, on="yad_no")
        .with_columns(
            pl.col("yad_no_label").alias("candidates").cast(pl.Int32),
            pl.col("session_count").cast(pl.Int32),
        )
        .drop(["yad_no", "yad_no_label"])
        .select(["session_id", "candidates", "pred", "session_count"])
    )
    # first と結合
    result = (
        pl.concat([first_df, first_df_from_label])
        .group_by(["session_id", "candidates"])
        .agg(pl.col("pred").sum(), pl.col("session_count").max())
        .sort(by=["session_id", "pred"], descending=True)
    )
    return result


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    output_path = Path(cfg.dir.exp_dir) / exp_name
    os.makedirs(output_path, exist_ok=True)

    global logger
    logger = get_logger(__name__, file_path=output_path / "run.log")

    logger.info(f"exp_name: {exp_name}")
    logger.info(f"ouput_path: {output_path}")
    logger.info(OmegaConf.to_yaml(cfg))

    other_oof_df = pl.read_parquet(Path(cfg.exp.other_dirs[0]) / "oof_pred.parquet")
    other_test_df = pl.read_parquet(Path(cfg.exp.other_dirs[0]) / "test_pred.parquet")
    first_oof_df = pl.read_parquet(Path(cfg.exp.first_dirs[0]) / "oof_pred.parquet")
    first_test_df = pl.read_parquet(Path(cfg.exp.first_dirs[0]) / "test_pred.parquet")

    with utils.trace("eval"):
        oof_candidate_df = make_eval_df(cfg, other_oof_df, first_oof_df)
        logger.info(oof_candidate_df.head())
        metrics = calculate_metrics(
            oof_candidate_df, candidates_col="candidates", label_col="yad_no", k=[10]
        )
        logger.info(metrics)

    with utils.trace("submission"):
        test_submission_df = make_submission(cfg, other_test_df, first_test_df)
        logger.info(test_submission_df.head())
        test_submission_df.write_csv(output_path / "submission.csv")

    with utils.trace("post process for eval"):
        oof_candidate_df = make_eval_df(
            cfg, other_oof_df, concat_label_pred(cfg, first_oof_df, "train")
        )
        logger.info(oof_candidate_df.head())
        metrics = calculate_metrics(
            oof_candidate_df, candidates_col="candidates", label_col="yad_no", k=[10]
        )
        logger.info(metrics)

    with utils.trace("post process for submission"):
        test_submission_df = make_submission(
            cfg, other_test_df, concat_label_pred(cfg, first_test_df, "test")
        )
        logger.info(test_submission_df.head())
        test_submission_df.write_csv(output_path / "submission_pp.csv")


if __name__ == "__main__":
    main()
