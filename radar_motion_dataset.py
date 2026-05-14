from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from label_utils import COARSE_LABEL_TO_ID, classify_coarse_label

@dataclass(frozen=True)
class RadarSample:
    path: Path
    label: str
    fine_label: str
    coarse_label: str


def read_prompt_fields(sample_dir: Path) -> dict[str, str]:
    prompt_path = sample_dir / "prompt.txt"
    if not prompt_path.exists():
        return {}

    fields: dict[str, str] = {}
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return fields


def sample_label(sample_dir: Path, label_mode: str) -> tuple[str, str, str]:
    fields = read_prompt_fields(sample_dir)
    fine_label = fields.get("name") or sample_dir.name
    coarse_label = classify_coarse_label(
        fine_label,
        desc=fields.get("desc", ""),
        env=fields.get("env", ""),
    )
    if label_mode == "fine":
        return fine_label, fine_label, coarse_label
    if label_mode == "coarse":
        return coarse_label, fine_label, coarse_label
    raise ValueError(f"Unsupported label_mode={label_mode!r}")


def discover_samples(root: Path, label_mode: str = "fine") -> list[RadarSample]:
    """Find sample folders that contain point-cloud radar data."""
    root = root.expanduser().resolve()
    candidates: Iterable[Path]
    if (root / "pointclouds.npy").exists() or (root / "radarllm_6d.npy").exists():
        candidates = [root]
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_dir())

    samples: list[RadarSample] = []
    for sample_dir in candidates:
        if (sample_dir / "pointclouds.npy").exists() or (sample_dir / "radarllm_6d.npy").exists():
            label, fine_label, coarse_label = sample_label(sample_dir, label_mode)
            samples.append(
                RadarSample(
                    path=sample_dir,
                    label=label,
                    fine_label=fine_label,
                    coarse_label=coarse_label,
                )
            )
    if not samples:
        raise FileNotFoundError(f"No sample folders with pointclouds.npy or radarllm_6d.npy under {root}")
    return samples


class RadarPointCloudDataset(Dataset):
    """
    Loads one motion clip per folder.

    Output tensors:
      points: [T, N, C]
      point_mask: [T, N], True where a real point exists
      frame_mask: [T], True where a real frame exists
      label: scalar class id
    """

    def __init__(
        self,
        root,
        point_file: str = "pointclouds.npy",
        max_frames: int = 64,
        max_points: int = 128,
        label_mode: str = "fine",
        labels: Optional[dict[str, int]] = None,
    ) -> None:
        self.root = Path(root)
        self.label_mode = label_mode
        self.samples = discover_samples(self.root, label_mode=label_mode)
        self.point_file = point_file
        self.max_frames = max_frames
        self.max_points = max_points

        if labels is None:
            if label_mode == "coarse":
                labels = dict(COARSE_LABEL_TO_ID)
            else:
                names = sorted({sample.label for sample in self.samples})
                labels = {name: idx for idx, name in enumerate(names)}
        self.label_to_id = labels
        self.id_to_label = {idx: name for name, idx in labels.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        point_path = sample.path / self.point_file
        if not point_path.exists():
            point_path = sample.path / "radarllm_6d.npy"

        frames = np.load(point_path, allow_pickle=True)
        points, point_mask, frame_mask = self._pack_frames(frames)

        return {
            "points": torch.from_numpy(points),
            "point_mask": torch.from_numpy(point_mask),
            "frame_mask": torch.from_numpy(frame_mask),
            "label": torch.tensor(self.label_to_id.get(sample.label, -1), dtype=torch.long),
        }

    def _pack_frames(self, frames: np.ndarray):
        valid_frames = min(len(frames), self.max_frames)
        feature_dim = int(frames[0].shape[1]) if valid_frames and len(frames[0]) else 6

        points = np.zeros((self.max_frames, self.max_points, feature_dim), dtype=np.float32)
        point_mask = np.zeros((self.max_frames, self.max_points), dtype=bool)
        frame_mask = np.zeros((self.max_frames,), dtype=bool)

        for frame_idx in range(valid_frames):
            frame = np.asarray(frames[frame_idx], dtype=np.float32)
            if frame.ndim != 2 or frame.shape[0] == 0:
                continue
            count = min(frame.shape[0], self.max_points)
            points[frame_idx, :count] = frame[:count]
            point_mask[frame_idx, :count] = True
            frame_mask[frame_idx] = True

        return points, point_mask, frame_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--label-mode", choices=("fine", "coarse"), default="fine")
    args = parser.parse_args()

    dataset = RadarPointCloudDataset(args.root, label_mode=args.label_mode)
    print(f"samples: {len(dataset)}")
    print(f"labels: {dataset.label_to_id}")
    item = dataset[0]
    for key, value in item.items():
        print(key, tuple(value.shape) if hasattr(value, "shape") else value)


if __name__ == "__main__":
    main()
