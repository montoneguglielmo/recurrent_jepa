# Recurrent JEPA

A latent world model for robotic manipulation based on the **Joint-Embedding Predictive Architecture (JEPA)**. The model learns to predict future latent representations of observations and is trained with the **SIGReg** regularizer from [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://huggingface.co/collections/quentinll/lewm), which keeps the embedding space well-distributed by penalising deviations from an isotropic Gaussian.

## Architecture

```
Observations (pixels + proprioception)
        │
        ▼
 ┌─────────────────────────────────────┐
 │  Raw Encoders                       │
 │  • ViT-tiny (vision, 96×96 → 192d) │
 │  • MLP      (proprio → 10d)         │
 └──────────────┬──────────────────────┘
                │ concat (202d)
                ▼
 ┌─────────────────────────────────────┐
 │  State Encoder  (MLP, 400d, BN)    │
 └──────────────┬──────────────────────┘
                │ z_t  (B, T, 400d)
       ┌────────┴────────┐
       ▼                 ▼
 ┌──────────────┐  ┌─────────────────────────────┐
 │ Predictor    │  │ Auxiliary classifiers        │
 │ (diffusion,  │  │ • agent position  (→ 2d)     │
 │  100 steps)  │  │ • block pos+angle (→ 3d)     │
 └──────┬───────┘  └─────────────────────────────┘
        │ ẑ_{t+1}
        ▼
 ┌─────────────────────┐
 │ Action Decoder      │
 │ (MLP, concat z_t    │
 │  and ẑ_{t+1} → 10d)│
 └─────────────────────┘
```

**Training objective**: MSE between predicted and actual next-state embeddings, regularised with SIGReg.

## Installation

Requires **Python 3.10**. Install dependencies with [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
# with uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt

# with pip
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Datasets use the HDF5 format for fast loading. Download from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:

```bash
export STABLEWM_HOME=/path/to/your/storage
```

The dataset name and loading config are in [config/train/data/pusht.yaml](config/train/data/pusht.yaml):

```yaml
dataset:
  name: pusht_expert_train   # matches the .h5 filename (without extension)
  keys_to_load: [pixels, action, proprio, state]
  num_steps: 4               # history (3) + predictions (1)
  frameskip: 5
```

## Training

```bash
python train.py
```

Configuration is managed with [Hydra](https://hydra.cc). The main config is [config/train/lewm.yaml](config/train/lewm.yaml). Key overrides:

```bash
# Resume from a checkpoint
python train.py resume_from=/path/to/ckpt.ckpt

# Disable W&B logging
python train.py wandb.enabled=False

# Limit batches per epoch (useful for debugging)
python train.py trainer.limit_batches=10
```

Checkpoints are saved to `~/.cache/stable_worldmodel/outputs/<date>/<time>/checkpoints/`.

### Hyperparameter sweeps (W&B)

```bash
wandb sweep sweep_config.yaml
wandb agent <sweep-id>
```

## Planning

Three planning modes are available, selected via the `policy` key in [config/eval/pusht.yaml](config/eval/pusht.yaml).

### 1. Random planning
Useful as a baseline for benchmarking. No model required.

```yaml
policy: random
```

### 2. CEM planning (LeWorldModel-style)
Follows the approach from the LeWorldModel paper. The CEM solver samples many action sequences, rolls them forward through the world model, and iteratively refines the distribution around the best candidates.

```yaml
policy: le-wm
```

To use this mode with a custom model, the model must implement:

```python
def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
    ...
```

- `info_dict` — current observation dict (must contain `"pixels"` and `"goal"` keys)
- `action_candidates` — sampled action sequences of shape `(Batch, N_samples, Horizon, action_dim)`
- returns a cost tensor of shape `(Batch, N_samples)` — lower is better (MSE to goal embedding is the natural choice)

`stable_worldmodel` discovers the method automatically by scanning the checkpoint for any `nn.Module` that has `get_cost`. `JEPA` does not implement this yet.

### 3. Custom policy (`get_action`)
Load any checkpoint as a policy. `eval.py` will call `policy.get_action(info)` at each step, where `info` contains the current observation and goal.

```yaml
policy: /path/to/best_object.ckpt
```

`JEPA.get_action` is already implemented in [jepa.py](jepa.py): it samples candidate next states from the predictor, selects the one closest to the goal embedding, and decodes an action via the action decoder.

## Evaluation

```bash
python eval.py
```

Config: [config/eval/pusht.yaml](config/eval/pusht.yaml). Set the policy path before running:

```yaml
policy:
  path: /path/to/best_object.ckpt
```

## File overview

| File | Description |
|---|---|
| [jepa.py](jepa.py) | `JEPA` model — forward pass, loss, and policy inference |
| [module.py](module.py) | Neural network building blocks: `MLP`, `TimeWrapper`, `ConditionalDiffusionPredictor`, `VisionTransformer` / `vit_tiny`, `SIGReg`, `CNNNet` |
| [train.py](train.py) | Training loop (Hydra entry point) |
| [eval.py](eval.py) | Policy evaluation on the PushT environment |
| [utils.py](utils.py) | Device setup, image preprocessing, column normalizers |
| [test.py](test.py) | Offline validation: loads best checkpoint and reports MSE / Pearson correlation for auxiliary classifiers |
| [sweep_config.yaml](sweep_config.yaml) | W&B Bayesian hyperparameter sweep config |
