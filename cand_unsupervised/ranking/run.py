"""
logの出現頻度のランキングを作成する

yad_feature
┌────────┬─────────────────────────┬───────────────────────┐
│ yad_no ┆ counts_001_ranking/base ┆ rank_001_ranking/base │
│ ---    ┆ ---                     ┆ ---                   │
│ i64    ┆ u32                     ┆ f64                   │
╞════════╪═════════════════════════╪═══════════════════════╡
│ 12350  ┆ 1606                    ┆ 1.0                   │
│ 719    ┆ 1520                    ┆ 2.0                   │
│ 3338   ┆ 1492                    ┆ 3.0                   │
│ 13468  ┆ 1373                    ┆ 4.0                   │
│ 10095  ┆ 1313                    ┆ 5.0                   │
└────────┴─────────────────────────┴───────────────────────┘

candidates
┌──────────────────────────────────┬─────────────────────┐
│ session_id                       ┆ candidates          │
│ ---                              ┆ ---                 │
│ str                              ┆ list[i64]           │
╞══════════════════════════════════╪═════════════════════╡
│ 000007603d533d30453cc45d0f3d119f ┆ [12350, 719, … 496] │
│ 0000ca043ed437a1472c9d1d154eb49b ┆ [12350, 719, … 496] │
│ 0000d4835cf113316fe447e2f80ba1c8 ┆ [12350, 719, … 496] │
│ 0000fcda1ae1b2f431e55a7075d1f500 ┆ [12350, 719, … 496] │
│ 000104bdffaaad1a1e0a9ebacf585f33 ┆ [12350, 719, … 496] │
└──────────────────────────────────┴─────────────────────┘
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
        mode="disabled",  # if cfg.debug else "online",
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
    ランクの作成
    """
    with utils.timer("create ranking"):
        count_df = (
            all_log_df.get_column("yad_no")
            .value_counts()
            .sort(by="counts", descending=True)
        ).with_columns(pl.col("counts").rank(descending=True).alias("rank"))

        # ランキングを保存
        save_df = count_df.with_columns(
            pl.col("counts").alias(f"counts_{exp_name}"),
            pl.col("rank").alias(f"rank_{exp_name}"),
        ).select(["yad_no", f"counts_{exp_name}", f"rank_{exp_name}"])
        print(save_df.head())
        print(save_df.shape)
        save_df.write_parquet(output_path / "yad_feature.parquet")

    """
    候補の作成
    """
    # session_id ごとにランキングの上位10個を予測値とする submission を作成
    with utils.timer("load session data"):
        train_session_df = load_session_data(Path(cfg.dir.data_dir), "train")
        test_session_df = load_session_data(Path(cfg.dir.data_dir), "test")

    with utils.timer("make candidates"):
        ## 上位num_candidate個の yad_no を取得
        yad_list = count_df.get_column("yad_no").to_list()[: cfg.exp.num_candidate]
        train_candidate_df = train_session_df.with_columns(
            pl.Series(
                name="candidates",
                values=[yad_list for _ in range(len(train_session_df))],
            )
        )
        test_candidate_df = test_session_df.with_columns(
            pl.Series(
                name="candidates",
                values=[yad_list for _ in range(len(test_session_df))],
            )
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
            train_candidate_df = train_candidate_df.head(100)
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
