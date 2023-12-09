"""
作成した候補を結合してデータセットを作成する
trainに対しては抜けているラベルのデータがあれば追加しておく（ただし、検証時にはこのデータは使わない）
"""

import os
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

import utils
from utils.load import load_label_data


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.datasets_dir) / exp_name
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    with utils.timer("load data"):
        train_label_df = load_label_data(Path(cfg.dir.data_dir))

    skf = StratifiedKFold(
        n_splits=cfg.exp.n_splits, shuffle=True, random_state=cfg.seed
    )

    X = train_label_df.get_column("session_id").to_numpy()
    y = train_label_df.get_column("yad_no").to_numpy()
    folds = np.zeros(train_label_df.height)
    for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
        folds[valid_index] = fold

    train_fold_df = train_label_df.with_columns(
        pl.Series(folds).cast(pl.Int64).alias("fold")
    ).select(["session_id", "fold"])

    train_fold_df.write_parquet(output_path / "train_fold.parquet")
    print(f"train_fold_df: {train_fold_df.shape}")
    print(train_fold_df.head(5))


if __name__ == "__main__":
    my_app()
