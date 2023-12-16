"""
yad_no に対するグラフ特徴量を付加する
"""

import os
import sys
from pathlib import Path

import hydra
import igraph as ig
import numpy as np
import polars as pl
import scipy.sparse as sparse
from hydra.core.hydra_config import HydraConfig
from igraph import Graph
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix, eye

import utils
from utils.load import load_log_data, load_yad_data
from utils.metrics import calculate_metrics


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.cand_unsupervised_dir) / exp_name
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    with utils.timer("load data"):
        train_log_df = load_log_data(Path(cfg.dir.data_dir), "train")
        test_log_df = load_log_data(Path(cfg.dir.data_dir), "test")
        all_log_df = pl.concat([train_log_df, test_log_df])

    """
    グラフ作成
    """
    with utils.trace("making graph"):
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

        for rti in [-1, 1]:
            if rti == 0:
                continue
            df = (
                all_log_cast_df.with_columns(
                    pl.col("yid").alias("from_id"),
                    pl.col("yid").shift(-(rti)).over("session_id").alias("to_id"),
                )
                .filter(~pl.col("to_id").is_null())
                .filter(pl.col("from_id") != pl.col("to_id"))  # 同じものへは遷移しない
                .select(["from_id", "to_id"])
            )
            transition_dfs.append(df)
        transition_df = pl.concat(transition_dfs)

        # 行列の作成
        matrix = sparse.csr_matrix(
            (
                np.ones(len(transition_df)),
                (
                    transition_df["from_id"].to_numpy(),
                    transition_df["to_id"].to_numpy(),
                ),
            )
        ).toarray()

        graph = Graph.Adjacency(matrix)

    """
    頂点の特徴量を作成する
    """
    with utils.trace("making node features"):
        node_degrees = graph.outdegree()
        node_pageranks = graph.pagerank()
        node_clustering_coefs = graph.transitivity_local_undirected()

        yad_feature_df = pl.DataFrame(
            {
                "yad_no": [yid2yad_no[yid] for yid in range(len(node_pageranks))],
                "degree": node_degrees,
                "pagerank": node_pageranks,
                "clustering_coef": node_clustering_coefs,
            }
        )
        print(yad_feature_df.head())
        print(yad_feature_df.shape)
        yad_feature_df.write_parquet(output_path / "yad_feature.parquet")


if __name__ == "__main__":
    my_app()
