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
        yad_df = load_yad_data(Path(cfg.dir.data_dir))
        # all_log_df = pl.concat([train_log_df, test_log_df])

    """
    遷移確率の作成
    """
    with utils.timer("create transition"):
        for target_col in ["sml_cd", "lrg_cd", "ken_cd", "wid_cd"]:

            def make_transition_df(all_log_df):
                all_log_df = all_log_df.join(
                    yad_df.select(["yad_no", target_col]), on="yad_no", how="left"
                )

                transition_dfs = []
                # 遷移を作成
                for rti in range(-cfg.exp.range_transition, cfg.exp.range_transition):
                    if rti == 0:
                        continue
                    df = (
                        all_log_df.with_columns(
                            pl.col(target_col).alias(f"from_{target_col}"),
                            pl.col(target_col)
                            .shift(-(rti + 1))
                            .over("session_id")
                            .alias(f"to_{target_col}"),
                        )
                        .filter(~pl.col(f"to_{target_col}").is_null())
                        .filter(
                            pl.col(f"from_{target_col}") != pl.col(f"to_{target_col}")
                        )  # 同じものへは遷移しない
                        .select([f"from_{target_col}", f"to_{target_col}"])
                    )
                    transition_dfs.append(df)
                transition_df = pl.concat(transition_dfs)
                # 集約して確率計算
                transition_df = (
                    transition_df.group_by([f"from_{target_col}", f"to_{target_col}"])
                    .agg(
                        pl.col(f"from_{target_col}").count().alias("from_to_count")
                    )  # 分子を計算(同じペア数)
                    .with_columns(
                        pl.col("from_to_count")
                        .sum()
                        .over([f"from_{target_col}"])
                        .alias("from_count"),
                    )  # 分母を計算
                    .with_columns(
                        (pl.col("from_to_count") / pl.col("from_count")).alias(
                            f"transition_prob_{target_col}"
                        )
                    )
                    .sort(by=[f"from_{target_col}", f"to_{target_col}"])
                    .select(
                        [
                            f"from_{target_col}",
                            f"to_{target_col}",
                            f"transition_prob_{target_col}",
                        ]
                    )
                )

                return transition_df

            # train と test で別々に計算
            train_transition_df = make_transition_df(train_log_df)
            test_transition_df = make_transition_df(test_log_df)

            # ランキングを保存
            save_df = train_transition_df.select(
                [
                    f"from_{target_col}",
                    f"to_{target_col}",
                    f"transition_prob_{target_col}",
                ]
            )
            print(save_df.head())
            print(save_df.shape)
            save_df.write_parquet(
                output_path / f"{target_col}2{target_col}_feature_train.parquet"
            )

            # ランキングを保存
            save_df = test_transition_df.select(
                [
                    f"from_{target_col}",
                    f"to_{target_col}",
                    f"transition_prob_{target_col}",
                ]
            )
            save_df.write_parquet(
                output_path / f"{target_col}2{target_col}_feature_test.parquet"
            )


if __name__ == "__main__":
    my_app()
