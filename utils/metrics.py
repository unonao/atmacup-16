import polars as pl


def calculate_precision(recommended_items: list[int], actual_item: int, k: int):
    """Calculate precision for a single row"""
    return int(actual_item in recommended_items[:k]) / k


def calculate_recall(recommended_items: list[int], actual_item: int, k: int):
    """Calculate recall for a single row"""
    return int(actual_item in recommended_items[:k])


def calculate_average_precision_at_k(
    recommended_items: list[int], actual_item: int, k: int
):
    """Calculate MAP@K for a single row"""

    if actual_item not in recommended_items[:k]:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(recommended_items):
        if p == actual_item and p not in recommended_items[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score


def calculate_metrics(
    candidate_df: pl.DataFrame,
    candidates_col: str,
    label_col: str,
    k: int | list[int] = 10,
    is_print=True,
) -> float:
    """
    recall, precision, map@k を計算
    """
    metrics_list = []

    if isinstance(k, int):
        k = [k]

    for k_ in k:
        metrics = {}
        # label_df の yad_no をリストに変換
        avg_num_candidates = (
            candidate_df.to_pandas()[candidates_col].apply(lambda x: len(x[:k_])).mean()
        )

        recall = (
            candidate_df.select(candidates_col, label_col)
            .map_rows(lambda row: calculate_recall(row[0], row[1], k_))
            .to_numpy()
            .mean()
        )
        precision = (
            candidate_df.select(candidates_col, label_col)
            .map_rows(lambda row: calculate_precision(row[0], row[1], k_))
            .to_numpy()
            .mean()
        )

        map_at_k = (
            candidate_df.select(candidates_col, label_col)
            .map_rows(lambda row: calculate_average_precision_at_k(row[0], row[1], k_))
            .to_numpy()
            .mean()
        )

        metrics = {
            "k": k_,
            "avg_num_candidates": avg_num_candidates,
            "recall": recall,
            "precision": precision,
            "map@k": map_at_k,
        }
        if is_print:
            print(metrics)
        metrics_list.append(metrics)

    return metrics_list
