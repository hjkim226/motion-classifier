# Radar Motion Recognition Baseline

This is a PointNet + temporal Transformer baseline for classifying motion from radar point-cloud clips.

Each training sample is expected to be one folder containing:

- `pointclouds.npy` or `radarllm_6d.npy`: object array of `T` frames, each frame shaped `[N, 6]`
- `prompt.txt`: optional metadata. The `name:` field is used as the fine-grained class label.

Example sample:

```text
basting_meat/
  pointclouds.npy
  radarllm_6d.npy
  prompt.txt
```

For many motions, place folders under one dataset root:

```text
dataset/
  basting_meat_001/
  chopping_001/
  stirring_001/
```

Create a reproducible train/validation/test split:

```bash
python3 split_dataset.py --source dataset --output data --force
```

This writes `data/train`, `data/validation`, and `data/test` as symlinks by default, plus
`data/split_manifest.json`. Pass `--mode copy` if you need physical copies instead.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Inspect a dataset:

```bash
python3 radar_motion_dataset.py /path/to/dataset
python3 radar_motion_dataset.py /path/to/dataset --label-mode coarse
```

Train:

```bash
python3 train.py --data-root /path/to/dataset --epochs 50
```

Train the coarse `cooking` / `eating` / `cleanup` / `other` task with the prepared split:

```bash
python3 train.py \
  --data-root data/train \
  --val-root data/validation \
  --test-root data/test \
  --label-mode coarse \
  --epochs 50
```


Run inference:

```bash
python3 infer.py /path/to/sample --checkpoint runs/pointnet_transformer/best.pt
```
