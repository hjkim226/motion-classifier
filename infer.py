from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import RadarMotionNet
from radar_motion_dataset import RadarPointCloudDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--point-file", default="pointclouds.npy")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    label_to_id = checkpoint["label_to_id"]
    label_mode = checkpoint.get("label_mode", "fine")
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    dataset = RadarPointCloudDataset(
        Path(args.sample_dir),
        point_file=args.point_file,
        max_frames=checkpoint["max_frames"],
        max_points=checkpoint["max_points"],
        label_mode=label_mode,
        labels=label_to_id,
    )
    batch = dataset[0]
    model = RadarMotionNet(
        num_classes=len(label_to_id),
        frame_dim=checkpoint.get("frame_dim", 256),
        temporal_layers=checkpoint.get("temporal_layers", 3),
        temporal_heads=checkpoint.get("temporal_heads", 4),
        dropout=checkpoint.get("dropout", 0.1),
        max_frames=checkpoint["max_frames"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        logits = model(
            batch["points"].unsqueeze(0),
            batch["point_mask"].unsqueeze(0),
            batch["frame_mask"].unsqueeze(0),
        )
        probs = logits.softmax(dim=1)[0]

    top = torch.topk(probs, k=min(5, probs.numel()))
    for score, class_id in zip(top.values.tolist(), top.indices.tolist()):
        print(f"{id_to_label[class_id]}: {score:.4f}")


if __name__ == "__main__":
    main()
