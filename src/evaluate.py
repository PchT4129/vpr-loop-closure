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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tolerance", type=int, default=3)
    args = parser.parse_args()

    database_features, database_paths = load_feature_file(args.database)
    query_features, query_paths = load_feature_file(args.query)

    top_scores, top_indices = retrieve_top_k(
        query_features=query_features,
        database_features=database_features,
        k=args.top_k,
    )

    for split_name in ["day_right", "night_right"]:
        split_indices = [
            index
            for index, path in enumerate(query_paths)
            if f"/{split_name}/" in path
        ]

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

        print("=" * 80)
        print(f"Split: {split_name}")
        print(f"Num queries: {len(split_query_paths)}")
        print(f"Tolerance: ±{args.tolerance} frames")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()