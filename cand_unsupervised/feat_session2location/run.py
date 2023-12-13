"""

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
from utils.load import load_label_data, load_log_data, load_session_data, load_yad_data
from utils.metrics import calculate_metrics


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    if cfg.debug:
        cfg.exp.num_candidate = 10

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.cand_unsupervised_dir) / exp_name
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    with utils.timer("load data"):
        yad_df = load_yad_data(Path(cfg.dir.data_dir))
        train_log_df = load_log_data(Path(cfg.dir.data_dir), "train")
        test_log_df = load_log_data(Path(cfg.dir.data_dir), "test")
        all_log_df = pl.concat([train_log_df, test_log_df])
        all_log_df = all_log_df.join(yad_df, on="yad_no", how="left")

        if cfg.debug:
            all_log_df = all_log_df.head(10000)
    """
    モデルの学習
    """
    with utils.timer("train model"):
        all_log_df = all_log_df.with_columns(
            # session_id を連番に変換
            pl.col("session_id").cast(pl.Categorical).to_physical().alias("sid"),
            # location を連番に変換
            pl.col(cfg.exp.location)
            .cast(pl.Categorical)
            .to_physical()
            .alias(cfg.exp.location + "_id"),
        )

        unique_df = all_log_df.unique(["sid", "session_id"])
        unique_sids = unique_df["sid"].to_numpy()
        unique_session_ids = unique_df["session_id"].to_list()

        unique_df = all_log_df.unique([cfg.exp.location, cfg.exp.location + "_id"])
        unique_location_ids = unique_df[cfg.exp.location + "_id"].to_numpy()
        unique_locations = unique_df[cfg.exp.location].to_list()
        loc_id2loc = dict(zip(unique_location_ids, unique_locations))

        sparse_item_user = sparse.csr_matrix(
            (
                np.ones(len(all_log_df)),
                (
                    all_log_df["sid"].to_numpy(),
                    all_log_df[cfg.exp.location + "_id"].to_numpy(),
                ),
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

        locations = unique_locations
        location_vectors = model.item_factors[unique_location_ids]
        location_factor_df = pl.DataFrame({cfg.exp.location: locations}).with_columns(
            pl.Series(
                name=f"{cfg.exp.location}_factor_{i}", values=location_vectors[:, i]
            )
            for i in range(location_vectors.shape[1])
        )
        print(location_factor_df.head())

        session_factor_df.write_parquet(output_path / "session_factor.parquet")
        location_factor_df.write_parquet(
            output_path / f"{cfg.exp.location}_factor.parquet"
        )

    # locationの順番的なものを生成
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

        # candidates を location に変換
        candidates = [[loc_id2loc[c] for c in cs] for cs in candidates]
        candidate_score_df = pl.DataFrame(
            {
                "session_id": unique_session_ids,  # unique_sids と同じ順番
                "candidates": candidates,
                f"{exp_name}/scores": scores,
            }
        )

        train_candidate_df = train_session_df.join(
            candidate_score_df, on="session_id", how="left"
        ).with_columns(pl.col("candidates").fill_null([]))
        test_candidate_df = test_session_df.join(
            candidate_score_df, on="session_id", how="left"
        ).with_columns(pl.col("candidates").fill_null([]))
        train_candidate_df.write_parquet(output_path / "train_candidate.parquet")
        test_candidate_df.write_parquet(output_path / "test_candidate.parquet")
        print(train_candidate_df.head())

    # 参考値としてラベルのlocation id とのスコアを計算

    with utils.timer("calculate metrics"):
        train_label_df = load_label_data(Path(cfg.dir.data_dir), "train")
        train_candidate_df = train_candidate_df.join(
            train_label_df.join(yad_df, on="yad_no"),
            on="session_id",
        )
        if cfg.debug:
            train_candidate_df = train_candidate_df.head(10000)
        metrics_list = calculate_metrics(
            train_candidate_df,
            candidates_col="candidates",
            label_col=cfg.exp.location,
            k=cfg.exp.k,
        )
        for metrics in metrics_list:
            print(metrics)


if __name__ == "__main__":
    my_app()
