import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import ResNet18FeatureExtractor
from src.triplet_dataset import TripletPlaceDataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        anchor = batch["anchor"].to(device)
        positive = batch["positive"].to(device)
        negative = batch["negative"].to(device)

        anchor_features = model(anchor)
        positive_features = model(positive)
        negative_features = model(negative)

        loss = criterion(anchor_features, positive_features, negative_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * anchor.size(0)

    average_loss = total_loss / len(loader.dataset)

    return average_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", type=str, required=True)
    parser.add_argument("--database-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--positive-tolerance", type=int, default=3)
    parser.add_argument("--negative-gap", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TripletPlaceDataset(
        anchor_dir=args.anchor_dir,
        database_dir=args.database_dir,
        positive_tolerance=args.positive_tolerance,
        negative_gap=args.negative_gap,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    model = ResNet18FeatureExtractor(pretrained=True).to(device)

    criterion = torch.nn.TripletMarginLoss(
        margin=args.margin,
        p=2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        average_loss = train_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {average_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs,
            "lr": args.lr,
            "margin": args.margin,
        },
        output_path,
    )

    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()