"""
node2vecにより候補頂点と類似度を計算する
"""

import os
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import Node2Vec
from tqdm.auto import tqdm

import utils
import wandb
from utils.load import load_label_data, load_log_data, load_session_data
from utils.metrics import calculate_metrics

try:
    from cuml.neighbors import NearestNeighbors
except ModuleNotFoundError:
    from sklearn.neighbors import NearestNeighbors


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
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"
    output_path = Path(cfg.dir.cand_unsupervised_dir) / exp_name
    os.makedirs(output_path, exist_ok=True)
    print(f"exp_name: {exp_name}")
    print(f"output_path: {output_path}")

    seed_everything(cfg.seed)

    wandb.init(
        project="atmaCup16-candidate",
        name=exp_name,
        config=OmegaConf.to_container(cfg.exp, resolve=True),
        mode="disabled" if cfg.debug else "online",
    )

    with utils.timer("load data"):
        train_log_df = load_log_data(Path(cfg.dir.data_dir), "train")
        test_log_df = load_log_data(Path(cfg.dir.data_dir), "test")
        all_log_df = pl.concat([train_log_df, test_log_df])

    with utils.timer("make dataset"):
        # ラベルエンコーディング
        le = LabelEncoder()
        yad_nos = np.unique(all_log_df.get_column("yad_no").to_list())
        yad_nos = np.sort(yad_nos)
        le.fit(yad_nos)

        # shift して遷移先の yad_no を取得
        all_log_df = all_log_df.with_columns(
            pl.col("yad_no").alias("from_yad_no"),
            pl.col("yad_no").shift(-1).alias("to_yad_no"),
        )

        # ラベルエンコーディング
        all_log_df = all_log_df.with_columns(
            pl.Series(
                name="from_yad_no", values=le.transform(all_log_df["from_yad_no"])
            ),
            pl.Series(name="to_yad_no", values=le.transform(all_log_df["to_yad_no"])),
        )

        edges = all_log_df.select(["from_yad_no", "to_yad_no"]).to_numpy()

    with utils.timer("prepare train"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Node2Vec(
            torch.from_numpy(edges).T,
            sparse=True,
            *OmegaConf.to_container(cfg.exp.node2vec.params, resolve=True),
        ).to(device)
        num_workers = 0 if cfg.debug else 4
        loader = model.loader(
            batch_size=cfg.exp.node2vec.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        optimizer = torch.optim.SparseAdam(model.parameters(), lr=cfg.exp.node2vec.lr)
        num_epochs = 1 if cfg.debug else cfg.exp.node2vec.num_epochs
        for epoch in range(num_epochs):
            loss = train()
            wandb.log({"loss/node2vec": loss, "epoch": epoch})

        vectors = model(torch.arange(model.num_nodes).to(device)).detach().cpu().numpy()
        vectors = vectors / np.linalg.norm(vectors)
        vector_df = pl.DataFrame(
            pl.Series(
                name="yad_no", values=le.inverse_transform(np.arange(len(vectors)))
            ),
            pl.Series(name="vector", values=vectors.tolist()),
        )
        vector_df.write_parquet(output_path / "vector.parquet")

    with utils.timer("nearest neibour"):
        knn = NearestNeighbors(n_neighbors=cfg.exp.num_candidate, metric="cosine")
        knn.fit(vectors)
        dist, indices = knn.kneighbors(vectors)
        yad_nos = le.inverse_transform(np.arange(len(vectors)))
        neighbor_yad_nos = yad_nos[indices]
        cosine_similarity = 1 - dist
        yad2cand_df = pl.DataFrame(
            pl.Series(name="yad_no", values=yad_nos),
            pl.Series(name="neighbor_yad_nos", values=neighbor_yad_nos.tolist()),
            pl.Series(name="cosine_similarity", values=cosine_similarity.tolist()),
        )
        yad2cand_df.write_parquet(output_path / "yad2candidate.parquet")

        print(yad2cand_df.head())

    """
    候補の作成
    """
    # session_id ごとにランキングの上位10個を予測値とする submission を作成
    with utils.timer("load session data"):
        train_session_df = load_session_data(Path(cfg.dir.data_dir), "train")
        test_session_df = load_session_data(Path(cfg.dir.data_dir), "test")

    with utils.timer("make candidates"):

        def make_candidates(log_df, session_df, yad2cand_df):
            last_log_df = (
                log_df.group_by("session_id")
                .agg(pl.all().sort_by("seq_no").last())
                .sort(by="session_id")
            )
            candidate_df = (
                last_log_df.join(yad2cand_df, on="yad_no")
                .group_by("session_id")
                .agg(
                    pl.col("neighbor_yad_nos").alias("candidates"),
                    # pl.col("cosine_similarity").alias("cosine_similarity"),
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
            train_log_df, train_session_df, yad2cand_df
        )
        test_candidate_df = make_candidates(test_log_df, test_session_df, yad2cand_df)

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
