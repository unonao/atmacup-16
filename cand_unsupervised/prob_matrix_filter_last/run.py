"""
logの出現遷移を使って遷移確率行列を作る
"""
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
import scipy.sparse as sparse
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix, eye
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

import utils
import wandb
from utils.load import load_label_data, load_log_data, load_session_data, load_yad_data
from utils.logger import get_logger
from utils.metrics import calculate_metrics


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    wandb.init(
        project="atmaCup16-candidate",
        name=exp_name,
        config=OmegaConf.to_container(cfg.exp, resolve=True),
        mode="disabled" if cfg.debug else "online",
    )

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.cand_unsupervised_dir) / exp_name
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    with utils.timer("load data"):
        train_log_df = load_log_data(Path(cfg.dir.data_dir), "train")
        test_log_df = load_log_data(Path(cfg.dir.data_dir), "test")
        all_log_df = pl.concat([train_log_df, test_log_df])

    """
    """
    with utils.timer("create ranking"):
        # 連番に変換

        all_log_cast_df = all_log_df.with_columns(
            pl.col("yad_no").cast(str).cast(pl.Categorical).to_physical().alias("yid"),
        )

        unique_df = all_log_cast_df.unique(["yad_no", "yid"])
        unique_yids = unique_df["yid"].to_numpy()
        unique_yad_nos = unique_df["yad_no"].to_list()
        yid2yad_no = dict(zip(unique_yids, unique_yad_nos))

        # 遷移を作成
        transition_dfs = []
        for rti in cfg.exp.range_transitions:
            if rti == 0:
                continue
            df = (
                all_log_cast_df.with_columns(
                    pl.col("yid").alias("from_id"),
                    pl.col("yid").shift(-(rti)).over("session_id").alias("to_id"),
                )
                .filter(~pl.col("to_id").is_null())
                .filter(pl.col("from_id") != pl.col("to_id"))  # 同じものへは遷移しない
                .select(["session_id", "from_id", "to_id"])
            )
            transition_dfs.append(df)
        transition_df = (
            pl.concat(transition_dfs)
            .unique(["session_id", "from_id", "to_id"])
            .drop("session_id")
        )

        # 疎行列の作成
        sparse_matrix = sparse.csr_matrix(
            (
                np.ones(len(transition_df)),
                (
                    transition_df["from_id"].to_numpy(),
                    transition_df["to_id"].to_numpy(),
                ),
            )
        )

        # 右確率行列にする
        sparse_matrix = normalize(sparse_matrix, norm="l1", axis=1)

        if cfg.exp.self_loop_prob is not None:
            # セルフループを追加
            sparse_matrix = (
                sparse_matrix * (1 - cfg.exp.self_loop_prob)
                + eye(sparse_matrix.shape[0]) * cfg.exp.self_loop_prob
            )
            # 一応 normalize (特に影響はないはずだがセルフループのみのやつもあるため)
            sparse_matrix = normalize(sparse_matrix, norm="l1", axis=1)

        for _ in range(cfg.exp.transition_times - 1):
            sparse_matrix *= sparse_matrix

        # 各行から上位K件の要素のindexとvalueを取得する関数
        def top_k_indices_per_row(matrix, K):
            top_k_indices = np.argsort(-matrix, axis=1)[:, :K]
            top_k_values = np.array(
                [matrix[i, top_k_indices[i]] for i in range(matrix.shape[0])]
            )
            return top_k_indices, top_k_values

        indices, values = top_k_indices_per_row(
            sparse_matrix.toarray(), cfg.exp.num_candidate
        )

        # from_yad_no,

        from_yad_no = [yid2yad_no[i] for i in range(len(indices))]
        to_yad_nos = [[yid2yad_no[c] for c in cs] for cs in indices]

        yad2yad_df = pl.DataFrame(
            {
                "from_yad_no": from_yad_no,  # unique_sids と同じ順番
                "to_yad_nos": to_yad_nos,
                "transition_prob": values,
            }
        )
        yad2yad_df = (
            yad2yad_df.explode(["to_yad_nos", "transition_prob"])
            .filter(pl.col("transition_prob") > 0)
            .rename({"to_yad_nos": "to_yad_no"})
        )

        print(yad2yad_df.head())
        print(yad2yad_df.shape)
        yad2yad_df.write_parquet(output_path / "yad2yad_feature.parquet")

    """
    候補の作成
    """
    # session_id ごとにランキングの上位10個を予測値とする submission を作成
    with utils.timer("load session data"):
        train_session_df = load_session_data(Path(cfg.dir.data_dir), "train")
        test_session_df = load_session_data(Path(cfg.dir.data_dir), "test")

    with utils.timer("make candidates"):

        def make_candidates(log_df, session_df, transition_df):
            # session_id ごとに最後の yad_no を取得する
            log_df = (
                log_df.group_by("session_id")
                .agg(pl.all().sort_by("seq_no").last())
                .sort(by="session_id")
                .with_columns(pl.col("yad_no").alias("from_yad_no"))
            ).select(["session_id", "from_yad_no"])

            candidate_df = (
                log_df.join(transition_df, on="from_yad_no")
                .group_by(["session_id", "to_yad_no"])
                .agg(
                    pl.sum("transition_prob").alias("transition_prob"),
                )
                .sort(by=["session_id", "transition_prob"], descending=True)
                .group_by(["session_id"])
                .agg(
                    pl.col("to_yad_no").alias("candidates"),
                )
            )
            candidate_df = session_df.join(
                candidate_df, on="session_id", how="left"
            ).with_columns(
                # candidates が null の場合は空のリストを入れておく
                pl.when(pl.col("candidates").is_null())
                .then(pl.Series("empty", [[]]))
                .otherwise(pl.col("candidates"))
                .alias("candidates")
            )
            return candidate_df

        train_candidate_df = make_candidates(train_log_df, train_session_df, yad2yad_df)
        test_candidate_df = make_candidates(test_log_df, test_session_df, yad2yad_df)

        print("train_candidate_df")
        print(train_candidate_df.head(5))
        train_candidate_df.write_parquet(output_path / "train_candidate.parquet")
        test_candidate_df.write_parquet(output_path / "test_candidate.parquet")

    """
    スコア計算
    """
    with utils.timer("calculate metrics"):
        train_label_df = load_label_data(Path(cfg.dir.data_dir), "train")
        train_candidate_df = train_candidate_df.with_columns(
            train_label_df.select("yad_no")
        )
        metrics_list = calculate_metrics(
            train_candidate_df,
            candidates_col="candidates",
            label_col="yad_no",
            k=cfg.exp.k,
        )
        for metrics in metrics_list:
            wandb.log(metrics)
            print(metrics)

    """
    seq_lenごとに求める
    """
    seq_len_df = train_log_df.group_by("session_id").agg(
        (pl.col("seq_no").max() + 1).alias("seq_len")
    )
    train_candidate_df = train_candidate_df.join(seq_len_df, on="session_id")
    for i in range(1, 10):
        print(i)
        metrics_list = calculate_metrics(
            train_candidate_df.filter(pl.col("seq_len") == i),
            candidates_col="candidates",
            label_col="yad_no",
            k=10,
        )
        for metrics in metrics_list:
            metrics = {f"{k}/each_seq_len": v for k, v in metrics.items()}
            metrics["seq_len"] = i
            wandb.log(metrics)
            print(metrics)


if __name__ == "__main__":
    my_app()
