"""
logの出現頻度のランキングを作成する
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
        all_log_df = pl.concat([train_log_df, test_log_df])

    """
    ランクの作成
    """
    with utils.timer("create ranking"):
        count_df = (
            all_log_df.get_column("yad_no")
            .value_counts()
            .sort(by="counts", descending=True)
        ).with_row_count()
        print(count_df.head())
        print(count_df.shape)

        # ランキングを保存 "row_nr", "yad_no", "counts"
        count_df.write_parquet(output_path / "yad_feature_ranking.parquet")

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
