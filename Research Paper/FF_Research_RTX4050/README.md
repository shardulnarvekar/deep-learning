# FF Research — RTX 4050 Local Training

This project is a clean local implementation for CIFAR-100 built around the
published HCL-FF training principles, with a detached multi-layer fusion readout.

## What it preserves

- 17-layer residual CW-Conv Forward-Forward backbone
- class-wise mean-goodness local loss
- strict layer-wise detachment
- class-subset GroupNorm goodness decoupling
- coarse-to-fine hierarchy
- supervised contrastive loss on decoupled features
- strong CIFAR-100 augmentation
- validation-only goodness interval selection

## What this research variant adds

- separate paths for goodness / propagation / readout
- detached GAP features from multiple layers
- BN + dropout + linear fusion classifier
- validation-only ensemble calibration
- balanced mini-batches so SupCon still sees positives on a 6 GB GPU

The fusion head cannot send gradients back into the convolutional backbone.
That makes its effect easy to measure against the goodness baseline in the same run.

## Accuracy

The published HCL-FF paper reports ~70% on CIFAR-100 and a stronger ablation
around 70.76%. That does NOT guarantee this exact local 4050 run will exceed 70%.
This implementation is designed to preserve the strong parts of HCL while adding
a detached readout that may improve final prediction.

## Your environment

You already installed:
- Python 3.11
- torch 2.11.0+cu128
- torchvision 0.26.0+cu128
- RTX 4050 CUDA support

No Jupyter is required.

## First: smoke test

From the project folder, with `(.venv)` visible:

```bat
python train.py --epochs 2 --eval-every 1 --max-steps 2 --run-dir runs\smoke
```

This only verifies that:
- CIFAR-100 downloads
- the hierarchy downloads
- the model fits in VRAM
- forward/backward works
- checkpointing works

Do NOT judge accuracy from the smoke test.

## Then: full run

```bat
python train.py --epochs 1000 --eval-every 10 --run-dir runs\full_4050
```

The default physical batch is 128:
32 classes × 4 samples/class.

If VRAM usage is comfortable, later we can test a 256 batch:

```bat
python train.py --epochs 1000 --classes-per-batch 64 --samples-per-class 4 --eval-every 10 --run-dir runs\full_4050_b256
```

Do not change to batch 256 until the 128-batch smoke test succeeds.

## Resume

The script automatically resumes from:

`runs\<run-name>\last.pt`

So if Windows restarts or training is stopped, rerun the same command.

To intentionally start over:

```bat
python train.py ... --no-resume
```

## Outputs

Each run directory contains:

- `last.pt` — latest checkpoint
- `best.pt` — best validation checkpoint
- `metrics.csv` — training/validation history
- `config.json` — exact run configuration
- `final_test.json` — held-out test result after training completes

## Important research hygiene

The test set is not used to choose:
- goodness layer interval
- ensemble weights
- best checkpoint

Those choices are made using the validation split only.
The held-out CIFAR-100 test result is produced at the end.
