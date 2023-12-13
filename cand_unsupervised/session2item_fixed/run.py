"""
┌───────────────┬───────────────┬───────────────┬──────────────┬───┬──────────────┬──────────────┬──────────────┬──────────────┐
│ session_id    ┆ session_facto ┆ session_facto ┆ session_fact ┆ … ┆ session_fact ┆ session_fact ┆ session_fact ┆ session_fact │
│ ---           ┆ r_0           ┆ r_1           ┆ or_2         ┆   ┆ or_13        ┆ or_14        ┆ or_15        ┆ or_16        │
│ str           ┆ ---           ┆ ---           ┆ ---          ┆   ┆ ---          ┆ ---          ┆ ---          ┆ ---          │
│               ┆ f32           ┆ f32           ┆ f32          ┆   ┆ f32          ┆ f32          ┆ f32          ┆ f32          │
╞═══════════════╪═══════════════╪═══════════════╪══════════════╪═══╪══════════════╪══════════════╪══════════════╪══════════════╡
│ 000178c4d4d56 ┆ -0.056526     ┆ 0.000468      ┆ 0.040071     ┆ … ┆ 0.11476      ┆ 0.028699     ┆ 0.065785     ┆ 1.0          │
│ 7d4715331dd0c ┆               ┆               ┆              ┆   ┆              ┆              ┆              ┆              │
│ dab76c        ┆               ┆               ┆              ┆   ┆              ┆              ┆              ┆              │
│ 00162135a888f ┆ -0.064778     ┆ 0.087254      ┆ -0.166377    ┆ … ┆ 0.065091     ┆ 0.114852     ┆ 0.053998     ┆ 1.0          │
│ 3fddf855120f5 ┆               ┆               ┆              ┆   ┆              ┆              ┆              ┆              │
│ 6e8834        ┆               ┆               ┆              ┆   ┆              ┆              ┆              ┆              │
│ 00177461657db ┆ -0.012405     ┆ 0.093384      ┆ 0.376799     ┆ … ┆ 0.023636     ┆ 0.09102      ┆ -0.216416    ┆ 1.0          │
│ a2e30e7d97f2c ┆               ┆               ┆              ┆   ┆              ┆              ┆              ┆              │
│ 925bf4        ┆               ┆               ┆              ┆   ┆              ┆              ┆              ┆              │
│ 002248805d700 ┆ -0.059238     ┆ -0.000584     ┆ 0.043162     ┆ … ┆ -0.004842    ┆ 0.016494     ┆ 0.023197     ┆ 1.0          │
│ 62f562b237452 ┆               ┆               ┆              ┆   ┆              ┆              ┆              ┆              │
│ 88cb9b        ┆               ┆               ┆              ┆   ┆              ┆              ┆              ┆              │
│ 002a60849ac3a ┆ -0.051336     ┆ 0.034834      ┆ -0.046053    ┆ … ┆ -0.040264    ┆ -0.023008    ┆ 0.033096     ┆ 1.0          │
│ af2be1ccebd28 ┆               ┆               ┆              ┆   ┆              ┆              ┆              ┆              │
│ 081f9e        ┆               ┆               ┆              ┆   ┆              ┆              ┆              ┆              │
└───────────────┴───────────────┴───────────────┴──────────────┴───┴──────────────┴──────────────┴──────────────┴──────────────┘

┌──────────────────────────────────┬───────────────────────┬──────────────────────────────────┐
│ session_id                       ┆ candidates            ┆ session2item_fixed/bpr001/scores │
│ ---                              ┆ ---                   ┆ ---                              │
│ str                              ┆ list[i32]             ┆ list[f32]                        │
╞══════════════════════════════════╪═══════════════════════╪══════════════════════════════════╡
│ 000007603d533d30453cc45d0f3d119f ┆ [7373, 4271, … 9350]  ┆ [0.482072, 0.422179, … 0.32216]  │
│ 0000ca043ed437a1472c9d1d154eb49b ┆ [5483, 7373, … 6600]  ┆ [0.438193, 0.425272, … 0.326339] │
│ 0000d4835cf113316fe447e2f80ba1c8 ┆ [10871, 4271, … 4078] ┆ [0.419859, 0.410627, … 0.317624] │
│ 0000fcda1ae1b2f431e55a7075d1f500 ┆ [8475, 1282, … 5731]  ┆ [0.555531, 0.507757, … 0.352819] │
│ 000104bdffaaad1a1e0a9ebacf585f33 ┆ [898, 96, … 3631]     ┆ [1.381106, 1.151066, … 0.522688] │
└──────────────────────────────────┴───────────────────────┴──────────────────────────────────┘

session_id2candidate_score, train_candidate
┌──────────────────────────────────┬────────────────────────┬──────────────────────────────────┐
│ session_id                       ┆ candidates             ┆ session2item/base/scores         │
│ ---                              ┆ ---                    ┆ ---                              │
│ str                              ┆ list[i32]              ┆ list[f32]                        │
╞══════════════════════════════════╪════════════════════════╪══════════════════════════════════╡
│ 0007aa002adb32be360c7ca491a552d6 ┆ [9020, 11872, … 3052]  ┆ [1.165573, 0.979387, … 0.288602] │
│ 0014359abc1444b4f19f5225fa32cf9e ┆ [10610, 10573, … 2418] ┆ [0.365361, 0.35402, … 0.257573]  │
│ 001d2bfb0608cf3f31a10d536d6a2d34 ┆ [5149, 4215, … 13115]  ┆ [1.644428, 1.612176, … 0.313224] │
│ 003a3fbe80d8f4dbad0a2c78df6656eb ┆ [10610, 10573, … 9957] ┆ [0.366142, 0.354553, … 0.259129] │
│ 003d089ec3a3423c7370708b11454e08 ┆ [10610, 4807, … 13071] ┆ [0.354023, 0.344608, … 0.259409] │
└──────────────────────────────────┴────────────────────────┴──────────────────────────────────┘
"""

import os
import sys
from pathlib import Path

import hydra
import implicit
import numpy as np
import polars as pl
import scipy.sparse as sparse
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import utils
import wandb
from utils.load import load_label_data, load_log_data, load_session_data
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
    if cfg.debug:
        cfg.exp.num_candidate = 10

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.cand_unsupervised_dir) / exp_name
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    with utils.timer("load data"):
        train_log_df = load_log_data(Path(cfg.dir.data_dir), "train")
        test_log_df = load_log_data(Path(cfg.dir.data_dir), "test")
        all_log_df = pl.concat([train_log_df, test_log_df])

    """
    モデルの学習
    """
    with utils.timer("train model"):
        # session_id を連番に変換
        all_log_df = all_log_df.with_columns(
            pl.col("session_id").cast(pl.Categorical).to_physical().alias("sid"),
        )

        # sid と yad_no を対応させる dict
        unique_df = all_log_df.unique(["sid", "session_id"])
        unique_sids = unique_df["sid"].to_numpy()
        unique_session_ids = unique_df["session_id"].to_list()
        unique_yad_nos = unique_df["yad_no"].unique()

        sparse_item_user = sparse.csr_matrix(
            (
                np.ones(len(all_log_df)),
                (all_log_df["sid"].to_numpy(), all_log_df["yad_no"].to_numpy()),
            )
        )
        if cfg.exp.implicit.model == "bpr":
            from implicit.cpu.bpr import BayesianPersonalizedRanking

            model = BayesianPersonalizedRanking(
                **OmegaConf.to_container(cfg.exp.implicit.params, resolve=True)
            )
        elif cfg.exp.implicit.model == "als":
            from implicit.cpu.als import AlternatingLeastSquares

            model = AlternatingLeastSquares(
                **OmegaConf.to_container(cfg.exp.implicit.params, resolve=True)
            )

        model.fit(sparse_item_user)

    with utils.timer("save factor"):
        session_ids = unique_session_ids
        session_vectors = model.user_factors[unique_sids]
        session_factor_df = pl.DataFrame({"session_id": session_ids}).with_columns(
            pl.Series(name=f"session_factor_{i}", values=session_vectors[:, i])
            for i in range(session_vectors.shape[1])
        )
        print(session_factor_df.head())

        yad_ids = unique_yad_nos
        yad_vectors = model.item_factors[unique_yad_nos]
        yad_factor_df = pl.DataFrame({"yad_no": yad_ids}).with_columns(
            pl.Series(name=f"yad_factor_{i}", values=yad_vectors[:, i])
            for i in range(yad_vectors.shape[1])
        )
        print(yad_factor_df.head())

        session_factor_df.write_parquet(output_path / "session_factor.parquet")
        yad_factor_df.write_parquet(output_path / "yad_factor.parquet")

    with utils.timer("predict&save candidates"):
        train_session_df = load_session_data(Path(cfg.dir.data_dir), "train")
        test_session_df = load_session_data(Path(cfg.dir.data_dir), "test")

        # 少し時間がかかる
        candidates, scores = model.recommend(
            unique_sids,
            sparse_item_user[unique_sids],
            N=cfg.exp.num_candidate,
            filter_already_liked_items=False,
        )
        candidate_score_df = pl.DataFrame(
            {
                "session_id": unique_session_ids,  # unique_sids と同じ順番
                "candidates": candidates,
                f"{exp_name}/scores": scores,
            }
        )

        train_candidate_df = train_session_df.join(
            candidate_score_df, on="session_id", how="left"
        )
        test_candidate_df = test_session_df.join(
            candidate_score_df, on="session_id", how="left"
        )
        train_candidate_df.write_parquet(output_path / "train_candidate.parquet")
        test_candidate_df.write_parquet(output_path / "test_candidate.parquet")
        print(train_candidate_df.head())

    with utils.timer("calculate metrics"):
        train_label_df = load_label_data(Path(cfg.dir.data_dir), "train")
        train_candidate_df = train_candidate_df.with_columns(
            train_label_df.select("yad_no")
        )
        if cfg.debug:
            train_candidate_df = train_candidate_df.head(10000)
        metrics_list = calculate_metrics(
            train_candidate_df,
            candidates_col="candidates",
            label_col="yad_no",
            k=cfg.exp.k,
        )
        for metrics in metrics_list:
            wandb.log(metrics)
            print(metrics)


if __name__ == "__main__":
    my_app()
