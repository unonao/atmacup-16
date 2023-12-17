"""
session内でのyad_noの遷移から yad_no の遷移確率を計算し、その確率が高いものを候補とする

yad2yad_feature
┌─────────────┬───────────┬───────────────────────────────────┐
│ from_yad_no ┆ to_yad_no ┆ transition_prob_transition_prob/… │
│ ---         ┆ ---       ┆ ---                               │
│ i64         ┆ i64       ┆ f64                               │
╞═════════════╪═══════════╪═══════════════════════════════════╡
│ 1           ┆ 1254      ┆ 0.25                              │
│ 1           ┆ 1503      ┆ 0.25                              │
│ 1           ┆ 4133      ┆ 0.25                              │
│ 1           ┆ 10352     ┆ 0.25                              │
│ 2           ┆ 3847      ┆ 0.25                              │
└─────────────┴───────────┴───────────────────────────────────┘

candidates
┌──────────────────────────────────┬───────────────────────┐
│ session_id                       ┆ candidates            │
│ ---                              ┆ ---                   │
│ str                              ┆ list[i64]             │
╞══════════════════════════════════╪═══════════════════════╡
│ 000007603d533d30453cc45d0f3d119f ┆ [11882, 2808, … 5289] │
│ 0000ca043ed437a1472c9d1d154eb49b ┆ [8253, 8747, … 4488]  │
│ 0000d4835cf113316fe447e2f80ba1c8 ┆ []                    │
│ 0000fcda1ae1b2f431e55a7075d1f500 ┆ [626, 755, … 7872]    │
│ 000104bdffaaad1a1e0a9ebacf585f33 ┆ [96, 3894, … 12338]   │
└──────────────────────────────────┴───────────────────────┘
"""

import os
import sys
from pathlib import Path

import hydra
import polars as pl
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

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.cand_unsupervised_dir) / exp_name
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    with utils.timer("load data"):
        train_log_df = load_log_data(Path(cfg.dir.data_dir), "train")
        test_log_df = load_log_data(Path(cfg.dir.data_dir), "test")
        # all_log_df = pl.concat([train_log_df, test_log_df])

    """
    遷移確率の作成
    """
    with utils.timer("create ranking"):

        def make_transition_df(all_log_df):
            transition_dfs = []

            # 遷移を作成
            for rti in range(-cfg.exp.range_transition, cfg.exp.range_transition):
                if rti == 0:
                    continue
                df = (
                    all_log_df.with_columns(
                        pl.col("yad_no").alias("from_yad_no"),
                        pl.col("yad_no")
                        .shift(-(rti + 1))
                        .over("session_id")
                        .alias("to_yad_no"),
                    )
                    .filter(~pl.col("to_yad_no").is_null())
                    .filter(
                        pl.col("from_yad_no") != pl.col("to_yad_no")
                    )  # 同じものへは遷移しない
                    .select(["from_yad_no", "to_yad_no"])
                )
                transition_dfs.append(df)
            transition_df = pl.concat(transition_dfs)
            # 集約して確率計算
            transition_df = (
                transition_df.group_by(["from_yad_no", "to_yad_no"])
                .agg(pl.col("from_yad_no").count().alias("from_to_count"))
                .with_columns(
                    pl.col("from_to_count")
                    .sum()
                    .over(["from_yad_no"])
                    .alias("from_count"),
                )
                .with_columns(
                    (pl.col("from_to_count") / pl.col("from_count")).alias(
                        "transition_prob"
                    )
                )
                .sort(by=["from_yad_no", "to_yad_no"])
                .select(["from_yad_no", "to_yad_no", "transition_prob"])
            )
            return transition_df

        # train と test で別々に計算
        train_transition_df = make_transition_df(train_log_df)
        test_transition_df = make_transition_df(test_log_df)

        # ランキングを保存
        save_df = train_transition_df.with_columns(
            pl.col("transition_prob").alias(f"transition_prob_{exp_name}"),
        ).select(["from_yad_no", "to_yad_no", f"transition_prob_{exp_name}"])
        save_df.write_parquet(output_path / "yad2yad_feature_train.parquet")
        save_df = test_transition_df.with_columns(
            pl.col("transition_prob").alias(f"transition_prob_{exp_name}"),
        ).select(["from_yad_no", "to_yad_no", f"transition_prob_{exp_name}"])
        save_df.write_parquet(output_path / "yad2yad_feature_test.parquet")

    """
    候補の作成
    """
    # session_id ごとにランキングの上位10個を予測値とする submission を作成
    with utils.timer("load session data"):
        train_session_df = load_session_data(Path(cfg.dir.data_dir), "train")
        test_session_df = load_session_data(Path(cfg.dir.data_dir), "test")

    with utils.timer("make candidates"):

        def make_candidates(log_df, session_df, transition_df):
            log_df = (
                log_df.sort(by="session_id").with_columns(
                    pl.col("yad_no").alias("from_yad_no")
                )
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

        train_candidate_df = make_candidates(
            train_log_df, train_session_df, train_transition_df
        )
        test_candidate_df = make_candidates(
            test_log_df, test_session_df, test_transition_df
        )

        print("train_candidate_df")
        print(train_candidate_df.head(5))
        # 保存
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
