from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import RadarMotionNet
from radar_motion_dataset import RadarPointCloudDataset


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, leave=False):
        batch = batch_to_device(batch, device)
        logits = model(batch["points"], batch["point_mask"], batch["frame_mask"])
        loss = criterion(logits, batch["label"])

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch["label"].size(0)
        correct += (logits.argmax(dim=1) == batch["label"]).sum().item()
        total += batch["label"].size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".", help="Folder containing samples or one sample folder.")
    parser.add_argument("--val-root", default=None, help="Optional validation split folder.")
    parser.add_argument("--test-root", default=None, help="Optional test split folder.")
    parser.add_argument("--point-file", default="pointclouds.npy")
    parser.add_argument("--label-mode", choices=("fine", "coarse"), default="fine")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-points", type=int, default=128)
    parser.add_argument("--output-dir", default="runs/pointnet_transformer")
    args = parser.parse_args()

    dataset = RadarPointCloudDataset(
        args.data_root,
        point_file=args.point_file,
        max_frames=args.max_frames,
        max_points=args.max_points,
        label_mode=args.label_mode,
    )
    if len(dataset.label_to_id) < 2:
        raise ValueError(
            "Training needs at least two motion classes. This folder currently contains "
            f"{len(dataset)} sample(s) and labels={dataset.label_to_id}."
        )

    test_set = None
    if args.val_root:
        train_set = dataset
        val_set = RadarPointCloudDataset(
            args.val_root,
            point_file=args.point_file,
            max_frames=args.max_frames,
            max_points=args.max_points,
            label_mode=args.label_mode,
            labels=dataset.label_to_id,
        )
    else:
        val_size = max(1, int(len(dataset) * 0.2))
        train_size = len(dataset) - val_size
        if train_size < 1:
            raise ValueError("Need at least two samples to create a train/validation split.")

        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    if args.test_root:
        test_set = RadarPointCloudDataset(
            args.test_root,
            point_file=args.point_file,
            max_frames=args.max_frames,
            max_points=args.max_points,
            label_mode=args.label_mode,
            labels=dataset.label_to_id,
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = (
        DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if test_set is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarMotionNet(
        num_classes=len(dataset.label_to_id),
        max_frames=args.max_frames,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels.json").write_text(json.dumps(dataset.label_to_id, indent=2), encoding="utf-8")

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "label_to_id": dataset.label_to_id,
                    "label_mode": args.label_mode,
                    "max_frames": args.max_frames,
                    "max_points": args.max_points,
                },
                output_dir / "best.pt",
            )

    if test_loader is not None:
        checkpoint = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
        print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")

if __name__ == "__main__":
    main()
