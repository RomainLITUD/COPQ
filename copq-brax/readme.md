# COP-Q Brax Experiments

This folder contains the JAX/Brax implementation used for the COP-Q locomotion
experiments in the paper.

## What is included

- `saferl.py`: main experiment entrypoint.
- `sac/`: modified SAC learner with multi-objective critics and COP-Q losses.
- `custom_envs/`: Brax locomotion environments with reward/safety signals.
- `vis_utils.py`: plotting utilities for saved experiment outputs.

The customized Brax environments used by the paper are provided in
`custom_envs/`. They define the task-specific reward and safety signals for
`hopper`, `walker2d`, `ant`, and `humanoid`, so no separate environment code
needs to be downloaded beyond installing Brax itself.

The critic predicts safety/cost and reward values as a vector. COP-Q estimates
the ensemble covariance, applies Cholesky-ordered projection, and uses the
projected value in both critic and actor updates.

## Requirements

This code requires Brax and JAX. They are not included in this repository.

Typical packages:

```bash
pip install brax jax jaxlib flax optax numpy matplotlib
```

Use the JAX installation command appropriate for your machine if you need GPU
support.

## Run

Run commands from this directory:

```bash
cd copq-brax
```

Hard-safety locomotion task examples:

```bash
python saferl.py \
  --env humanoid \
  --task-name lrl \
  --learning-method cop \
  --model-mode mo \
  --ensemble-size 3 \
  --beta 1.0 \
  --save-file humanoid_lrl_cop
```

Safe-velocity constrained task examples:

```bash
python saferl.py \
  --env humanoid \
  --task-name crl \
  --learning-method cop \
  --model-mode mo \
  --ensemble-size 3 \
  --beta 1.0 \
  --save-file humanoid_crl_cop
```

The default device is CPU. To select a GPU after installing GPU-enabled JAX:

```bash
python saferl.py --device gpu --gpu-id 0
```

Outputs are saved under `results/<env>/<save-file>.npz`. The file contains:

- `x`: evaluation environment steps
- `y`: evaluation metrics
- `replay_rewards`: reward/cost samples from replay

## Useful options

- `--env`: one of `hopper`, `walker2d`, `ant`, `humanoid`
- `--num-seeds`: number of seeds, default `10`
- `--env-steps`: training steps, default `3072000`
- `--output-dir`: result directory, default `results`
- `--save-policy`: also save final policy parameters
- `--exploration --explore-method boundary`: enable the optional exploration policy

## Baseline branches

The loss code also keeps baseline branches used during development:

- `worst_of_both`: objective-wise conservative double-Q
- `scalar`: scalarized double-Q for hard-safety locomotion
- `saclag`, `saclag_ucb`, `cal`: constrained RL baselines

For the paper COP-Q path, use `--learning-method cop`.
