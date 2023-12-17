"""

yad_feature(地域ごとのカウントとランク)
┌────────┬────────────────────────────────┬──────────────────────────────┐
│ yad_no ┆ counts_ranking_location/ken_cd ┆ rank_ranking_location/ken_cd │
│ ---    ┆ ---                            ┆ ---                          │
│ i64    ┆ u32                            ┆ f64                          │
╞════════╪════════════════════════════════╪══════════════════════════════╡
│ 1      ┆ 30                             ┆ 363.5                        │
│ 2      ┆ 29                             ┆ 129.0                        │
│ 3      ┆ 210                            ┆ 32.0                         │
│ 4      ┆ 67                             ┆ 447.5                        │
│ 5      ┆ 30                             ┆ 59.0                         │
└────────┴────────────────────────────────┴──────────────────────────────┘

┌──────────────────────────────────┬────────────────────────┐
│ session_id                       ┆ candidates             │
│ ---                              ┆ ---                    │
│ str                              ┆ list[i64]              │
╞══════════════════════════════════╪════════════════════════╡
│ 000007603d533d30453cc45d0f3d119f ┆ [11499, 2445, … 12447] │
│ 0000ca043ed437a1472c9d1d154eb49b ┆ [3184, 7888, … 13749]  │
│ 0000d4835cf113316fe447e2f80ba1c8 ┆ [12350, 3338, … 13126] │
│ 0000fcda1ae1b2f431e55a7075d1f500 ┆ [385, 11398, … 10947]  │
│ 000104bdffaaad1a1e0a9ebacf585f33 ┆ [10827, 8985, … 13465] │
└──────────────────────────────────┴────────────────────────┘
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
from utils.load import load_label_data, load_log_data, load_session_data, load_yad_data
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
        yad_df = load_yad_data(Path(cfg.dir.data_dir))

    """
    ランクの作成
    """
    with utils.timer("create ranking"):

        def make_count_df(all_log_df):
            count_df = (
                all_log_df.get_column("yad_no")
                .value_counts()
                .sort(by="counts", descending=True)
            )
            yad_counts_df = yad_df.join(count_df, on="yad_no").with_columns(
                pl.col("counts")
                .rank(descending=True)
                .over(cfg.exp.location_col)
                .alias("rank")
            )
            return yad_counts_df

        # train と test で別々に計算
        train_count_df = make_count_df(train_log_df)
        test_count_df = make_count_df(test_log_df)

        # ランキングを保存
        save_df = train_count_df.with_columns(
            pl.col("counts").alias(f"counts_{exp_name}"),
            pl.col("rank").alias(f"rank_{exp_name}"),
        ).select(["yad_no", f"counts_{exp_name}", f"rank_{exp_name}"])
        print(save_df.head())
        print(save_df.shape)
        save_df.write_parquet(output_path / "yad_feature_train.parquet")

        save_df = test_count_df.with_columns(
            pl.col("counts").alias(f"counts_{exp_name}"),
            pl.col("rank").alias(f"rank_{exp_name}"),
        ).select(["yad_no", f"counts_{exp_name}", f"rank_{exp_name}"])
        print(save_df.head())
        print(save_df.shape)
        save_df.write_parquet(output_path / "yad_feature_test.parquet")

    """
    候補の作成
    """
    # session_id ごとにランキングの上位10個を予測値とする submission を作成
    with utils.timer("load session data"):
        train_session_df = load_session_data(Path(cfg.dir.data_dir), "train")
        test_session_df = load_session_data(Path(cfg.dir.data_dir), "test")

    with utils.timer("make candidates"):
        # location_col ごとにrankが高いものを候補とする
        location_candidates_df = (
            (
                train_count_df.sort(by="counts", descending=True)
                .group_by(cfg.exp.location_col)
                .agg(
                    [
                        pl.col("yad_no").alias("candidates"),
                    ]
                )
            )
            .select([cfg.exp.location_col, "candidates"])
            .sort(by=cfg.exp.location_col)
        )

        # それぞれの session_id の中で最頻値の location_col を取得
        train_session_mode_df = (
            train_log_df.join(yad_df, on="yad_no")
            .group_by("session_id")
            .agg(
                [
                    pl.col(cfg.exp.location_col)
                    .mode()
                    .first()
                    .alias(cfg.exp.location_col)
                ]
            )
        ).sort(by="session_id")
        print("train_session_mode_df")
        print(train_session_mode_df.head(5))

        # candidate
        train_candidate_df = (
            train_session_df.join(train_session_mode_df, on="session_id")
            .join(location_candidates_df, on=cfg.exp.location_col)
            .select(["session_id", "candidates"])
        ).sort(by="session_id")
        print("train_candidate_df")
        print(train_candidate_df.head(5))

        # test
        location_candidates_df = (
            (
                test_count_df.sort(by="counts", descending=True)
                .group_by(cfg.exp.location_col)
                .agg(
                    [
                        pl.col("yad_no").alias("candidates"),
                    ]
                )
            )
            .select([cfg.exp.location_col, "candidates"])
            .sort(by=cfg.exp.location_col)
        )
        test_session_mode_df = (
            test_log_df.join(yad_df, on="yad_no")
            .group_by("session_id")
            .agg(
                [
                    pl.col(cfg.exp.location_col)
                    .mode()
                    .first()
                    .alias(cfg.exp.location_col)
                ]
            )
        )
        test_candidate_df = (
            test_session_df.join(test_session_mode_df, on="session_id")
            .join(location_candidates_df, on=cfg.exp.location_col)
            .select(["session_id", "candidates"])
        )
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
