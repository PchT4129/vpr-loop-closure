import argparse
import re
from pathlib import Path

import torch

from src.retrieve import load_feature_file, retrieve_top_k


def image_index_from_path(path: str) -> int:
    match = re.search(r"Image(\d+)\.jpg$", path)
    if match is None:
        raise ValueError(f"Could not parse image index from path: {path}")

    return int(match.group(1))


def is_correct_match(query_path: str, database_path: str, tolerance: int) -> bool:
    query_index = image_index_from_path(query_path)
    database_index = image_index_from_path(database_path)

    return abs(query_index - database_index) <= tolerance


def evaluate_retrieval(
    query_paths: list[str],
    database_paths: list[str],
    top_indices: torch.Tensor,
    recall_ks: list[int],
    precision_k: int,
    tolerance: int,
):
    metrics = {}

    for k in recall_ks:
        num_success = 0

        for query_idx, query_path in enumerate(query_paths):
            retrieved_indices = top_indices[query_idx, :k]

            hit = any(
                is_correct_match(
                    query_path=query_path,
                    database_path=database_paths[db_idx.item()],
                    tolerance=tolerance,
                )
                for db_idx in retrieved_indices
            )

            if hit:
                num_success += 1

        metrics[f"recall@{k}"] = num_success / len(query_paths)

    total_precision = 0.0

    for query_idx, query_path in enumerate(query_paths):
        retrieved_indices = top_indices[query_idx, :precision_k]

        num_correct = sum(
            is_correct_match(
                query_path=query_path,
                database_path=database_paths[db_idx.item()],
                tolerance=tolerance,
            )
            for db_idx in retrieved_indices
        )

        total_precision += num_correct / precision_k

    metrics[f"precision@{precision_k}"] = total_precision / len(query_paths)

    return metrics


def select_query_indices(
    query_paths: list[str],
    split_name: str,
    min_index: int | None = None,
    max_index: int | None = None,
) -> list[int]:
    selected_indices = []

    for index, path in enumerate(query_paths):
        if f"/{split_name}/" not in path:
            continue

        image_index = image_index_from_path(path)
        if min_index is not None and image_index < min_index:
            continue
        if max_index is not None and image_index > max_index:
            continue

        selected_indices.append(index)

    return selected_indices


def print_metrics(
    split_name: str,
    num_queries: int,
    tolerance: int,
    metrics: dict[str, float],
    min_index: int | None = None,
    max_index: int | None = None,
):
    print("=" * 80)
    print(f"Split: {split_name}")
    print(f"Num queries: {num_queries}")
    print(f"Tolerance: ±{tolerance} frames")

    if min_index is not None or max_index is not None:
        min_label = "*" if min_index is None else str(min_index)
        max_label = "*" if max_index is None else str(max_index)
        print(f"Index range: {min_label}-{max_label}")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--min-index", type=int, default=None)
    parser.add_argument("--max-index", type=int, default=None)
    args = parser.parse_args()

    database_features, database_paths = load_feature_file(args.database)
    query_features, query_paths = load_feature_file(args.query)

    _, top_indices = retrieve_top_k(
        query_features=query_features,
        database_features=database_features,
        k=args.top_k,
    )

    split_names = [args.split_name] if args.split_name is not None else [
        "day_right",
        "night_right",
    ]

    for split_name in split_names:
        split_indices = select_query_indices(
            query_paths=query_paths,
            split_name=split_name,
            min_index=args.min_index,
            max_index=args.max_index,
        )

        if len(split_indices) == 0:
            raise ValueError(
                f"No queries found for split={split_name}, "
                f"min_index={args.min_index}, max_index={args.max_index}"
            )

        split_query_paths = [query_paths[index] for index in split_indices]
        split_top_indices = top_indices[split_indices]

        metrics = evaluate_retrieval(
            query_paths=split_query_paths,
            database_paths=database_paths,
            top_indices=split_top_indices,
            recall_ks=[1, 5, 10],
            precision_k=5,
            tolerance=args.tolerance,
        )

        print_metrics(
            split_name=split_name,
            num_queries=len(split_query_paths),
            tolerance=args.tolerance,
            metrics=metrics,
            min_index=args.min_index,
            max_index=args.max_index,
        )


if __name__ == "__main__":
    main()