import logging
import os
import pickle
import sys
import time
from pathlib import Path

import hydra
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import utils
import wandb
from utils.load import load_label_data, load_session_data
from utils.logger import get_logger
from utils.metrics import calculate_metrics
from wandb.lightgbm import log_summary, wandb_callback

logger = None


def train_one_fold(
    cfg: DictConfig, train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> lgb.Booster:
    unuse_cols = cfg.exp.lgbm.unuse_cols
    feature_cols = [col for col in train_df.columns if col not in unuse_cols]
    label_col = cfg.exp.lgbm.label_col

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[label_col]

    lgb_train_dataset = lgb.Dataset(
        X_train,
        label=np.array(y_train),
        feature_name=feature_cols,
    )
    lgb_valid_dataset = lgb.Dataset(
        X_valid,
        label=np.array(y_valid),
        feature_name=feature_cols,
    )

    bst = lgb.train(
        OmegaConf.to_container(cfg.exp.lgbm.params, resolve=True),
        lgb_train_dataset,
        valid_sets=[lgb_train_dataset, lgb_valid_dataset],
        valid_names=["train", "valid"],
        categorical_feature=cfg.exp.lgbm.cat_cols,
        verbose_eval=cfg.exp.lgbm.verbose_eval,
        callbacks=[wandb_callback()],
    )
    log_summary(bst, save_model_checkpoint=True)
    logger.info(
        f"best_itelation: {bst.best_iteration}, train: {bst.best_score['train']}, valid: {bst.best_score['valid']}"
    )
    return bst


def predict_one_fold(
    cfg: DictConfig, bst: lgb.Booster, test_df: pd.DataFrame
) -> pd.DataFrame:
    unuse_cols = cfg.exp.lgbm.unuse_cols
    feature_cols = [col for col in test_df.columns if col not in unuse_cols]

    X_test = test_df[feature_cols]
    y_pred = bst.predict(X_test)
    return y_pred


def save_model(cfg: DictConfig, bst: lgb.Booster, output_path: Path, fold: int) -> None:
    with open(output_path / f"model_dict_{fold}.pkl", "wb") as f:
        pickle.dump({"model": bst}, f)

    # save feature importance
    fig, ax = plt.subplots(figsize=(10, 20))
    ax = lgb.plot_importance(bst, importance_type="gain", ax=ax, max_num_features=100)
    fig.tight_layout()
    fig.savefig(output_path / f"importance_{fold}.png")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    output_path = Path(cfg.dir.exp_dir) / exp_name
    os.makedirs(output_path, exist_ok=True)

    global logger
    logger = get_logger(__name__, file_path=output_path / "run.log")

    logger.info(f"exp_name: {exp_name}")
    logger.info(f"ouput_path: {output_path}")
    logger.info(OmegaConf.to_yaml(cfg))

    wandb.init(
        project="atmaCup16-exp",
        name=exp_name,
        config=OmegaConf.to_container(cfg.exp, resolve=True),
        mode="disabled" if cfg.debug else "online",
    )

    # load datasets
    with utils.trace("load datasets"):
        train_df = pl.read_parquet(Path(cfg.exp.datasets_dir) / "train.parquet")
        test_df = pl.read_parquet(Path(cfg.exp.datasets_dir) / "test.parquet")
        num_folds = train_df["fold"].n_unique()
        index_array = np.arange(len(train_df))
        if cfg.debug:
            train_df = train_df.head(2000)
            test_df = test_df.head(2000)
            index_array = index_array[:2000]

    oof = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    # train
    for fold in range(num_folds):
        logger.info(f"fold: {fold}")
        train_fold_df = train_df.filter(pl.col("fold") != fold)
        valid_fold_df = train_df.filter(pl.col("fold") == fold)

        # train の負例をダウンサンプリング
        if cfg.exp.lgbm.downsampling_rate:
            train_positive_fold_df = train_fold_df.filter(
                pl.col(cfg.exp.lgbm.label_col) == 1
            )
            logger.info(f"train_positive_fold_df: {train_positive_fold_df.shape}")
            train_negative_fold_df = train_fold_df.filter(
                pl.col(cfg.exp.lgbm.label_col) == 0
            ).sample(fraction=cfg.exp.lgbm.downsampling_rate, seed=cfg.seed)
            logger.info(f"train_negative_fold_df: {train_negative_fold_df.shape}")
            train_fold_df = pl.concat([train_positive_fold_df, train_negative_fold_df])

        logger.info(f"train_fold_df: {train_fold_df.shape}")
        logger.info(f"valid_fold_df: {valid_fold_df.shape}")

        bst = train_one_fold(cfg, train_fold_df, valid_fold_df)
        save_model(cfg, bst, output_path, fold)

        # valid
        y_valid_pred = predict_one_fold(cfg, bst, valid_fold_df)
        y_pred = predict_one_fold(cfg, bst, test_df)
        oof[index_array[train_df["fold"].to_numpy() == fold]] = y_valid_pred
        test_preds += y_pred / num_folds

        #

    # make submission
    def make_submission(pred_df: pl.DataFrame, mode: str):
        session_df = load_session_data(Path(cfg.dir.data_dir), mode)
        pred_candidates_df = pred_df.group_by("session_id").agg(pl.col("candidates"))
        submission_df = (
            session_df.join(
                pred_candidates_df.with_columns(
                    [
                        pl.col("candidates").list.get(i).alias(f"predict_{i}")
                        for i in range(10)
                    ]
                ).drop("candidates"),
                on="session_id",
                how="left",
            )
            .fill_null(-1)
            .drop("session_id")
        )
        return submission_df, pred_candidates_df

    with utils.trace("save prediction"):
        np.save(output_path / "oof.npy", oof)
        np.save(output_path / "test_preds.npy", test_preds)

    with utils.trace("make oof submission & scoring"):
        oof_pred_df = (
            train_df.select(
                ["session_id", "candidates"],
            )
            .with_columns(pl.Series(name="pred", values=oof))
            .sort(by=["session_id", "pred"], descending=True)
        )
        _, oof_candidaates_df = make_submission(oof_pred_df, "train")
        train_label_df = load_label_data(Path(cfg.dir.data_dir))
        candidaates_df = train_label_df.join(
            oof_candidaates_df, on="session_id", how="left"
        ).with_columns(pl.col("candidates").fill_null([]))
        metrics = calculate_metrics(
            candidaates_df, candidates_col="candidates", label_col="yad_no", k=[10]
        )
        logger.info(f"metrics: {metrics}")
        for metric in metrics:
            wandb.log(metric)

    with utils.trace("make test submission"):
        test_pred_df = (
            test_df.select(
                ["session_id", "candidates"],
            )
            .with_columns(pl.Series(name="pred", values=test_preds))
            .sort(by=["session_id", "pred"], descending=True)
        )
        test_submission_df, _ = make_submission(test_pred_df, "test")
        test_submission_df.write_csv(output_path / "submission.csv")


if __name__ == "__main__":
    main()
