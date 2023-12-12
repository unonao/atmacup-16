"""
targetへの遷移確率を計算する

fold はtrainのみ
┌─────────────┬───────────┬─────────────────┬──────┐
│ from_yad_no ┆ to_yad_no ┆ transition_prob ┆ fold │
│ ---         ┆ ---       ┆ ---             ┆ ---  │
│ i64         ┆ i64       ┆ f64             ┆ i64  │
╞═════════════╪═══════════╪═════════════════╪══════╡
│ 2           ┆ 36        ┆ 0.058824        ┆ 0    │
│ 2           ┆ 299       ┆ 0.058824        ┆ 0    │
│ 2           ┆ 2200      ┆ 0.058824        ┆ 0    │
│ 2           ┆ 3847      ┆ 0.058824        ┆ 0    │
│ 2           ┆ 3860      ┆ 0.117647        ┆ 0    │
└─────────────┴───────────┴─────────────────┴──────┘

┌──────────────────────────────────┬──────────────────────┐
│ session_id                       ┆ candidates           │
│ ---                              ┆ ---                  │
│ str                              ┆ list[i64]            │
╞══════════════════════════════════╪══════════════════════╡
│ 000007603d533d30453cc45d0f3d119f ┆ [2808, 11882, 3324]  │
│ 0000ca043ed437a1472c9d1d154eb49b ┆ [8253, 1092, … 540]  │
│ 0000d4835cf113316fe447e2f80ba1c8 ┆ [9039, 6722, … 4355] │
│ 0000fcda1ae1b2f431e55a7075d1f500 ┆ [626, 755, … 2976]   │
│ 000104bdffaaad1a1e0a9ebacf585f33 ┆ [3894, 7749, … 9452] │
└──────────────────────────────────┴──────────────────────┘
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


def make_transition_prob(log_df, label_df, only_last=True):
    if only_last:
        log_df = (
            log_df.group_by("session_id")
            .agg(pl.all().sort_by("seq_no").last())
            .sort(by="session_id")
        )
    # labelを付与
    log_df = log_df.join(
        label_df.with_columns(pl.col("yad_no").alias("label")),
        on=["session_id"],
        how="left",
    ).with_columns(
        pl.col("yad_no").alias("from_yad_no"),
        pl.col("label").alias("to_yad_no"),
    )
    # 集約して確率計算
    transition_df = (
        log_df.group_by(["from_yad_no", "to_yad_no"])
        .agg(pl.col("from_yad_no").count().alias("from_to_count"))
        .with_columns(
            pl.col("from_to_count").sum().over(["from_yad_no"]).alias("from_count"),
        )
        .with_columns(
            (pl.col("from_to_count") / pl.col("from_count")).alias("transition_prob")
        )
        .sort(by=["from_yad_no", "to_yad_no"])
        .select(["from_yad_no", "to_yad_no", "transition_prob"])
    )
    return transition_df


def make_candidate(session_df, log_df, transition_df, mode: str, only_last=True):
    if only_last:
        log_df = (
            log_df.group_by("session_id")
            .agg(pl.all().sort_by("seq_no").last())
            .sort(by="session_id")
        )
    # probを付与
    if mode == "train":  # trainはfoldごとに異なる
        log_df = log_df.join(
            transition_df,
            left_on=["yad_no", "fold"],
            right_on=["from_yad_no", "fold"],
            how="inner",
        )
    elif mode == "test":
        log_df = log_df.join(
            transition_df,
            left_on=["yad_no"],
            right_on=["from_yad_no"],
            how="inner",
        )

    # 遷移確率を結合し、確率の降順に候補として生成する
    candidate_df = (
        log_df.group_by(
            ["session_id", "to_yad_no"]
        )  # all用に to_yad_noが複数あるときに対応するため集約
        .agg(pl.sum("transition_prob"))
        .sort(by=["session_id", "transition_prob"], descending=True)
        .group_by("session_id")
        .agg(pl.col("to_yad_no").alias("candidates"))
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
    output_path = Path(cfg.dir.cand_supervised_dir) / exp_name
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    with utils.timer("load data"):
        train_log_df = load_log_data(Path(cfg.dir.data_dir), "train")
        test_log_df = load_log_data(Path(cfg.dir.data_dir), "test")

        # fold を追加
        fold_df = pl.read_parquet(cfg.exp.fold_path)
        train_log_df = train_log_df.join(fold_df, on="session_id")

        train_label_df = load_label_data(Path(cfg.dir.data_dir), "train")
        train_session_df = load_session_data(Path(cfg.dir.data_dir), "train")
        test_session_df = load_session_data(Path(cfg.dir.data_dir), "test")

    """
    遷移確率の作成
    """
    with utils.timer("create ranking"):
        # train の遷移確率を作成
        train_transtion_dfs = []
        ## クロスバリデーションのfoldごとにtarget encodingをする
        for fold in range(train_log_df["fold"].n_unique()):
            train_fold_df = train_log_df.filter(pl.col("fold") != fold)
            # valid_fold_df = train_log_df.filter(pl.col("fold") == fold)

            # train_fold_df で、valid_fold_df 用の 遷移確率特徴と候補を生成する
            transition_df = make_transition_prob(train_fold_df, train_label_df)
            transition_df = transition_df.with_columns(
                pl.lit(fold).cast(pl.Int64).alias("fold")
            )  # 特定foldの特徴であることを明示する
            train_transtion_dfs.append(transition_df)
        train_trainsition_df = pl.concat(train_transtion_dfs)

        # test 用にtrain全体で遷移確率を作成
        test_transition_df = make_transition_prob(train_log_df, train_label_df)

        train_trainsition_df.write_parquet(output_path / "train_yad2yad.parquet")
        test_transition_df.write_parquet(output_path / "test_yad2yad.parquet")

        print(train_trainsition_df.head(5))

    """
    候補の作成
    """

    with utils.timer("make candidates"):
        train_candidate_df = make_candidate(
            train_session_df,
            train_log_df,
            train_trainsition_df,
            "train",
            only_last=cfg.exp.only_last,
        )
        print("train_candidate_df")
        print(train_candidate_df.head(5))

        test_candidate_df = make_candidate(
            test_session_df,
            test_log_df,
            test_transition_df,
            "test",
            only_last=cfg.exp.only_last,
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
