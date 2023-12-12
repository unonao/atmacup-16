"""
┌──────────────────┬──────────────────┬─────────────────┬─────────────────┬───┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ session_id       ┆ session_factor_0 ┆ session_factor_ ┆ session_factor_ ┆ … ┆ session_factor_ ┆ session_factor_ ┆ session_factor_ ┆ session_factor_ │
│ ---              ┆ ---              ┆ 1               ┆ 2               ┆   ┆ 61              ┆ 62              ┆ 63              ┆ 64              │
│ str              ┆ f32              ┆ ---             ┆ ---             ┆   ┆ ---             ┆ ---             ┆ ---             ┆ ---             │
│                  ┆                  ┆ f32             ┆ f32             ┆   ┆ f32             ┆ f32             ┆ f32             ┆ f32             │
╞══════════════════╪══════════════════╪═════════════════╪═════════════════╪═══╪═════════════════╪═════════════════╪═════════════════╪═════════════════╡
│ 0007aa002adb32be ┆ 0.035359         ┆ 0.018994        ┆ -0.13692        ┆ … ┆ -0.00649        ┆ 0.032971        ┆ -0.071059       ┆ 1.0             │
│ 360c7ca491a552d6 ┆                  ┆                 ┆                 ┆   ┆                 ┆                 ┆                 ┆                 │
│ 0014359abc1444b4 ┆ -0.006221        ┆ 0.005648        ┆ 0.017916        ┆ … ┆ -0.010312       ┆ -0.00538        ┆ -0.002639       ┆ 1.0             │
│ f19f5225fa32cf9e ┆                  ┆                 ┆                 ┆   ┆                 ┆                 ┆                 ┆                 │
│ 001d2bfb0608cf3f ┆ 0.171373         ┆ 0.128937        ┆ 0.040572        ┆ … ┆ -0.022058       ┆ 0.140751        ┆ -0.011165       ┆ 1.0             │
│ 31a10d536d6a2d34 ┆                  ┆                 ┆                 ┆   ┆                 ┆                 ┆                 ┆                 │
│ 003a3fbe80d8f4db ┆ -0.001459        ┆ 0.007201        ┆ -0.009924       ┆ … ┆ -0.025909       ┆ -0.023814       ┆ 0.022478        ┆ 1.0             │
│ ad0a2c78df6656eb ┆                  ┆                 ┆                 ┆   ┆                 ┆                 ┆                 ┆                 │
│ 003d089ec3a3423c ┆ -0.013307        ┆ 0.012161        ┆ -0.005174       ┆ … ┆ 0.002522        ┆ -0.012138       ┆ -0.018984       ┆ 1.0             │
│ 7370708b11454e08 ┆                  ┆                 ┆                 ┆   ┆                 ┆                 ┆                 ┆                 │
└──────────────────┴──────────────────┴─────────────────┴─────────────────┴───┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘

┌────────┬──────────────┬──────────────┬──────────────┬───┬───────────────┬───────────────┬───────────────┬───────────────┐
│ yad_no ┆ yad_factor_0 ┆ yad_factor_1 ┆ yad_factor_2 ┆ … ┆ yad_factor_61 ┆ yad_factor_62 ┆ yad_factor_63 ┆ yad_factor_64 │
│ ---    ┆ ---          ┆ ---          ┆ ---          ┆   ┆ ---           ┆ ---           ┆ ---           ┆ ---           │
│ i64    ┆ f32          ┆ f32          ┆ f32          ┆   ┆ f32           ┆ f32           ┆ f32           ┆ f32           │
╞════════╪══════════════╪══════════════╪══════════════╪═══╪═══════════════╪═══════════════╪═══════════════╪═══════════════╡
│ 9020   ┆ 0.019135     ┆ -0.009186    ┆ -0.00715     ┆ … ┆ -0.005009     ┆ 0.008592      ┆ 0.008974      ┆ 1.0           │
│ 9490   ┆ -0.089488    ┆ -0.115775    ┆ -0.052715    ┆ … ┆ 0.133665      ┆ 0.115494      ┆ 0.06883       ┆ 1.0           │
│ 5149   ┆ -0.005133    ┆ -0.003591    ┆ -0.016209    ┆ … ┆ 0.002912      ┆ -0.00662      ┆ 0.00836       ┆ 1.0           │
│ 6944   ┆ -0.006431    ┆ -0.049941    ┆ 0.008194     ┆ … ┆ -0.061958     ┆ -0.000141     ┆ -0.086868     ┆ 1.0           │
│ 12921  ┆ -0.003595    ┆ 0.000706     ┆ 0.000358     ┆ … ┆ 0.036212      ┆ -0.019668     ┆ -0.010116     ┆ 1.0           │
└────────┴──────────────┴──────────────┴──────────────┴───┴───────────────┴───────────────┴───────────────┴───────────────┘

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
        yad_vectors = model.user_factors[unique_yad_nos]
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
