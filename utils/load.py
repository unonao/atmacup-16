from pathlib import Path

import polars as pl


def load_log_data(data_dir: Path, mode: str) -> pl.DataFrame:
    """
    logデータを読み込む
    """
    # ファイルパスを取得
    log_file_path = data_dir / f"{mode}_log.csv"

    # データを読み込む
    df = pl.read_csv(log_file_path)

    return df


def load_label_data(data_dir: Path, mode: str = "train") -> pl.DataFrame:
    """
    labelデータを読み込む
    """
    # ファイルパスを取得
    label_file_path = data_dir / f"{mode}_label.csv"

    # データを読み込む
    df = pl.read_csv(label_file_path)

    return df


def load_session_data(data_dir: Path, mode: str) -> pl.DataFrame:
    """
    sessionデータを読み込む
    """
    df = None
    if mode == "test":
        session_file_path = data_dir / "test_session.csv"
        df = pl.read_csv(session_file_path)
    elif mode == "train":
        session_file_path = data_dir / "train_label.csv"
        df = pl.read_csv(session_file_path).drop("yad_no")
    assert df is not None
    return df
