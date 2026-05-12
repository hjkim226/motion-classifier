# Radar Motion Recognition Baseline

This is a PointNet + temporal Transformer baseline for classifying motion from radar point-cloud clips.

Each training sample is expected to be one folder containing:

- `pointclouds.npy` or `radarllm_6d.npy`: object array of `T` frames, each frame shaped `[N, 6]`
- `prompt.txt`: optional metadata. The `name:` field is used as the class label.

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

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Inspect a dataset:

```bash
python3 radar_motion_dataset.py /path/to/dataset
```

Train:

```bash
python3 train.py --data-root /path/to/dataset --epochs 50
```

Run inference:

```bash
python3 infer.py /path/to/sample --checkpoint runs/pointnet_transformer/best.pt
```

The current folder has only one label, `basting_meat`, so it is useful for checking loading but not enough to train a classifier. Add at least two motion classes before running training.
