import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.retrieve import load_feature_file, retrieve_top_k


def load_rgb_image(path: str | Path):
    return Image.open(path).convert("RGB")


def visualize_retrieval(
    query_path: str,
    database_paths: list[str],
    scores: torch.Tensor,
    output_path: str | None = None,
):
    num_results = len(database_paths)

    fig, axes = plt.subplots(
        1,
        num_results + 1,
        figsize=(4 * (num_results + 1), 4),
    )

    query_image = load_rgb_image(query_path)
    axes[0].imshow(query_image)
    axes[0].set_title("Query")
    axes[0].axis("off")

    for i, database_path in enumerate(database_paths):
        image = load_rgb_image(database_path)
        score = scores[i].item()

        axes[i + 1].imshow(image)
        axes[i + 1].set_title(f"Top {i + 1}\nscore={score:.3f}")
        axes[i + 1].axis("off")

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    database_features, database_paths = load_feature_file(args.database)
    query_features, query_paths = load_feature_file(args.query)

    top_scores, top_indices = retrieve_top_k(
        query_features=query_features,
        database_features=database_features,
        k=args.top_k,
    )

    query_path = query_paths[args.query_index]
    result_indices = top_indices[args.query_index]

    result_paths = [
        database_paths[index.item()]
        for index in result_indices
    ]

    result_scores = top_scores[args.query_index]

    print(f"Query: {query_path}")
    for rank, (path, score) in enumerate(zip(result_paths, result_scores), start=1):
        print(f"Top {rank}: score={score.item():.4f} | {path}")

    visualize_retrieval(
        query_path=query_path,
        database_paths=result_paths,
        scores=result_scores,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()