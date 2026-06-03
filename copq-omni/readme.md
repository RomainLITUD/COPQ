# COP-Q OmniSafe Experiments

This folder contains the OmniSafe-based implementation used for the COP-Q
Safety-Gymnasium experiments.

## What is included

- `omnisafe/algorithms/off_policy/copq.py`: COP-Q implementation.
- `omnisafe/configs/off-policy/COPQ.yaml`: default COP-Q configuration.
- `offpolicy_benchmark.py`: COP-Q off-policy benchmark entrypoint.
- `onpolicy_benchmark.py`: retained on-policy baseline benchmark entrypoint.

The code is based on OmniSafe. Unused OmniSafe baselines were removed for a
cleaner release, while the COP-Q implementation and paper-relevant comparison
methods are kept.

## Retained Methods

Off-policy:

- `COPQ`
- `SACLag`
- `SACUCB`
- `CAL`
- `ORAC`

On-policy:

- `RCPO`
- `CUP`
- `PPOSaute`
- `TRPOPID`

## Requirements

Install PyTorch, Safety-Gymnasium, and common OmniSafe dependencies. For
example:

```bash
pip install torch gymnasium safety-gymnasium numpy pyyaml rich typer pandas seaborn tensorboard wandb
```

Install the PyTorch build that matches your machine. A CPU build is enough for
testing; CUDA-enabled PyTorch is required for GPU training.

## Run

Run commands from this directory:

```bash
cd copq-omni
```

COP-Q on Safety-Gymnasium with CPU:

```bash
python offpolicy_benchmark.py \
  --env-id SafetyPointGoal2-v0 \
  --num-seeds 6 \
  --num-pool 6
```

COP-Q with GPUs:

```bash
python offpolicy_benchmark.py \
  --env-id SafetyPointGoal2-v0 \
  --gpu-ids 0,1 \
  --num-pool 6
```

When `--gpu-ids` is provided, experiment workers are assigned to `cuda:<id>`
round-robin by `ExperimentGrid`. If `--gpu-ids` is omitted, the YAML/default
device is used, which is CPU in this cleaned release.

Run retained on-policy baselines:

```bash
python onpolicy_benchmark.py \
  --env-id SafetyPointGoal2-v0 \
  --num-seeds 6 \
  --num-pool 6
```

## Common Environments

Paper safe-navigation tasks include:

- `SafetyPointGoal2-v0`
- `SafetyPointButton2-v0`
- `SafetyCarGoal2-v0`
- `SafetyCarButton2-v0`

Use `--env-id` to select one environment at a time.

## Outputs

Experiment outputs are written under:

```text
exp-x/<experiment-name>/
```

Each run stores the expanded config and logs using OmniSafe's logger.
