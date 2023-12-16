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
from utils.load import load_label_data, load_log_data, load_session_data
from utils.logger import get_logger
from utils.metrics import calculate_metrics

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


def concat_label_pred(cfg, first_df, transition_df, mode):
    # 最後のyad_noを作る＆そのセッションでの長さを計算&長さ１のものだけ残す
    log_df = load_log_data(Path(cfg.dir.data_dir), mode)
    last_log_df = (
        log_df.with_columns(
            (pl.col("seq_no").max().over("session_id") + 1).alias("session_count")
        )
        .filter(pl.col("session_count") == 1)
        .rename({"yad_no": "from_yad_no"})
    )
    # session と結合
    session_df = load_session_data(Path(cfg.dir.data_dir), mode)
    session_last_df = session_df.join(
        last_log_df.select(["session_id", "from_yad_no", "session_count"]),
        on="session_id",
    )

    # transitionと結合
    first_df_from_label = (
        session_last_df.join(
            transition_df.rename({cfg.exp.score_col: "pred"}), on="from_yad_no"
        )
        .with_columns(
            pl.col("to_yad_no").alias("candidates").cast(pl.Int32),
            pl.col("session_count").cast(pl.Int32),
            (pl.col("pred") + 1) * 1000,
        )
        .drop(["from_yad_no", "to_yad_no"])
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
    transition_df = pl.read_parquet(cfg.exp.transision_path).filter(
        pl.col("from_yad_no") != pl.col("to_yad_no")
    )

    with utils.trace("eval"):
        oof_candidate_df = make_eval_df(cfg, other_oof_df, first_oof_df)
        logger.info(oof_candidate_df.head())
        metrics = calculate_metrics(
            oof_candidate_df, candidates_col="candidates", label_col="yad_no", k=[10]
        )
        logger.info(metrics)
        # seq_lenごとのmetrics
        train_log_df = load_log_data(Path(cfg.dir.data_dir), "train")
        seq_len_df = train_log_df.group_by("session_id").agg(
            (pl.col("seq_no").max() + 1).alias("seq_len")
        )
        oof_candidate_df = oof_candidate_df.join(seq_len_df, on="session_id")
        for i in range(1, 10):
            logger.info(i)
            metrics_list = calculate_metrics(
                oof_candidate_df.filter(pl.col("seq_len") == i),
                candidates_col="candidates",
                label_col="yad_no",
                k=10,
                is_print=False,
            )
            for metrics in metrics_list:
                metrics = {f"{k}/each_seq_len": v for k, v in metrics.items()}
                metrics["seq_len"] = i
                logger.info(metrics)

    with utils.trace("submission"):
        test_submission_df = make_submission(cfg, other_test_df, first_test_df)
        logger.info(test_submission_df.head())
        test_submission_df.write_csv(output_path / "submission.csv")

    with utils.trace("post process for eval"):
        oof_candidate_df = make_eval_df(
            cfg,
            other_oof_df,
            concat_label_pred(cfg, first_oof_df, transition_df, "train"),
        )
        logger.info(oof_candidate_df.head())
        metrics = calculate_metrics(
            oof_candidate_df, candidates_col="candidates", label_col="yad_no", k=[10]
        )
        logger.info(metrics)
        # seq_lenごとのmetrics
        train_log_df = load_log_data(Path(cfg.dir.data_dir), "train")
        seq_len_df = train_log_df.group_by("session_id").agg(
            (pl.col("seq_no").max() + 1).alias("seq_len")
        )
        oof_candidate_df = oof_candidate_df.join(seq_len_df, on="session_id")
        for i in range(1, 10):
            logger.info(i)
            metrics_list = calculate_metrics(
                oof_candidate_df.filter(pl.col("seq_len") == i),
                candidates_col="candidates",
                label_col="yad_no",
                k=10,
            )
            for metrics in metrics_list:
                metrics = {f"{k}/each_seq_len": v for k, v in metrics.items()}
                metrics["seq_len"] = i
                logger.info(metrics)

    with utils.trace("post process for submission"):
        test_submission_df = make_submission(
            cfg,
            other_test_df,
            concat_label_pred(cfg, first_test_df, transition_df, "test"),
        )
        logger.info(test_submission_df.head())
        test_submission_df.write_csv(output_path / "submission_pp.csv")


if __name__ == "__main__":
    main()
