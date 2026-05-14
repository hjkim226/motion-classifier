from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    name: str
    path: Path
    group: str


def read_prompt(sample_dir: Path) -> dict[str, str]:
    prompt_path = sample_dir / "prompt.txt"
    if not prompt_path.exists():
        return {}

    fields: dict[str, str] = {}
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return fields


def read_scenario_groups(scenario_path: Path) -> dict[str, str]:
    if not scenario_path.exists():
        return {}

    items = json.loads(scenario_path.read_text(encoding="utf-8"))
    return {item["name"]: item.get("env", "unknown") for item in items}


def discover_samples(source: Path, scenario_groups: dict[str, str]) -> list[Sample]:
    samples: list[Sample] = []
    for sample_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        prompt = read_prompt(sample_dir)
        name = prompt.get("name") or sample_dir.name
        group = prompt.get("env") or scenario_groups.get(name) or scenario_groups.get(sample_dir.name) or "unknown"
        samples.append(Sample(name=sample_dir.name, path=sample_dir, group=group))
    if not samples:
        raise FileNotFoundError(f"No sample folders found under {source}")
    return samples


def allocate_counts(count: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if count < 3:
        return count, 0, 0

    test_count = max(1, round(count * test_ratio))
    val_count = max(1, round(count * val_ratio))
    train_count = count - val_count - test_count
    if train_count < 1:
        train_count = 1
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
    return train_count, val_count, test_count


def stratified_split(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.group, []).append(sample)

    splits = {"train": [], "validation": [], "test": []}
    for group_samples in groups.values():
        group_samples = list(group_samples)
        rng.shuffle(group_samples)
        train_count, val_count, test_count = allocate_counts(len(group_samples), val_ratio, test_ratio)

        splits["train"].extend(group_samples[:train_count])
        splits["validation"].extend(group_samples[train_count : train_count + val_count])
        splits["test"].extend(group_samples[train_count + val_count : train_count + val_count + test_count])

    for split_samples in splits.values():
        split_samples.sort(key=lambda sample: sample.name)
    return splits


def link_or_copy_sample(sample: Sample, split_dir: Path, mode: str) -> None:
    destination = split_dir / sample.name
    if mode == "copy":
        shutil.copytree(sample.path, destination)
        return

    target = Path("../..") / sample.path
    destination.symlink_to(target, target_is_directory=True)


def write_split(
    splits: dict[str, list[Sample]],
    source: Path,
    output: Path,
    mode: str,
    force: bool,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> None:
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists. Pass --force to replace it.")
        shutil.rmtree(output)

    output.mkdir(parents=True)
    manifest = {
        "source": str(source),
        "mode": mode,
        "seed": seed,
        "ratios": {"train": 1.0 - val_ratio - test_ratio, "validation": val_ratio, "test": test_ratio},
        "splits": {},
    }

    for split_name, split_samples in splits.items():
        split_dir = output / split_name
        split_dir.mkdir()
        for sample in split_samples:
            link_or_copy_sample(sample, split_dir, mode)
        manifest["splits"][split_name] = [
            {"name": sample.name, "group": sample.group} for sample in split_samples
        ]

    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/validation/test folders from sample directories.")
    parser.add_argument("--source", default="dataset", help="Folder containing one subfolder per sample.")
    parser.add_argument("--output", default="data", help="Output split folder.")
    parser.add_argument("--scenario", default="scenario.json", help="Scenario metadata used for stratification.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--force", action="store_true", help="Replace the output folder if it already exists.")
    args = parser.parse_args()

    source = Path(args.source)
    scenario_groups = read_scenario_groups(Path(args.scenario))
    samples = discover_samples(source, scenario_groups)
    splits = stratified_split(samples, args.val_ratio, args.test_ratio, args.seed)
    write_split(splits, source, Path(args.output), args.mode, args.force, args.seed, args.val_ratio, args.test_ratio)

    print(f"wrote {args.output}")
    for split_name, split_samples in splits.items():
        print(f"{split_name}: {len(split_samples)}")


if __name__ == "__main__":
    main()