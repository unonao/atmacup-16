"""
与えられた画像の数を特徴量として用いる
┌────────┬──────┬──────┬──────────┬──────────┬────────┬───────────────┐
│ yad_no ┆ food ┆ room ┆ facility ┆ exterior ┆ others ┆ sum_image_num │
│ ---    ┆ ---  ┆ ---  ┆ ---      ┆ ---      ┆ ---    ┆ ---           │
│ i64    ┆ u32  ┆ u32  ┆ u32      ┆ u32      ┆ u32    ┆ u32           │
╞════════╪══════╪══════╪══════════╪══════════╪════════╪═══════════════╡
│ 2      ┆ 3    ┆ 1    ┆ 3        ┆ 3        ┆ 2      ┆ 12            │
│ 30     ┆ 3    ┆ 3    ┆ 3        ┆ 2        ┆ 3      ┆ 14            │
│ 71     ┆ 2    ┆ 1    ┆ 1        ┆ 2        ┆ 3      ┆ 9             │
│ 116    ┆ 3    ┆ 1    ┆ 3        ┆ 3        ┆ 3      ┆ 13            │
│ 128    ┆ 3    ┆ 3    ┆ 3        ┆ 3        ┆ 3      ┆ 15            │
└────────┴──────┴──────┴──────────┴──────────┴────────┴───────────────┘
"""

import os
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import utils
from utils.load import (
    load_image_embeddings,
)


def seed_everything(seed: int = 42):
    """seedを固定するための関数
    Args:
        seed (int, optional): seedの値. Defaults to 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"
    output_path = Path(cfg.dir.cand_unsupervised_dir) / exp_name
    os.makedirs(output_path, exist_ok=True)
    print(f"exp_name: {exp_name}")
    print(f"output_path: {output_path}")

    with utils.timer("load data"):
        image_df = load_image_embeddings(
            Path(cfg.dir.data_dir), columns=["yad_no", "category"]
        )

    with utils.timer("nearest neibour"):
        image_count_df = (
            image_df.group_by(["yad_no", "category"])
            .agg(pl.col("category").count().alias("counts"))
            .pivot(values="counts", index="yad_no", columns="category")
            .with_columns(
                pl.sum_horizontal(
                    ["room", "others", "exterior", "food", "facility"]
                ).alias("sum_image_num")
            )
        )
        print(image_count_df.head())

        image_count_df.write_parquet(output_path / "yad2image_count.parquet")


if __name__ == "__main__":
    my_app()
