# Cholesky-Ordered Projection Q-learning

This repository contains the reproduction code for Cholesky-Ordered Projection
Q-learning (COP-Q), a safe reinforcement learning method that uses an ensemble
of multi-objective critics and a Cholesky-ordered projection to form
conservative value estimates for policy learning.

The code is organized as two independent implementations:

- `copq-brax/`: JAX/Brax locomotion experiments with customized Brax
  environments for reward and safety signals.
- `copq-omni/`: OmniSafe-based Safety-Gymnasium experiments, including COP-Q
  and the paper-relevant comparison methods.

Each folder has its own `readme.md` with dependencies, runnable commands,
environment names, outputs, and implementation notes. There is no root-level
training entrypoint; run experiments from the corresponding implementation
folder.

## Quick Start

Choose the codebase that matches the experiment you want to reproduce:

```bash
cd copq-brax   # Brax locomotion experiments
```

or

```bash
cd copq-omni   # Safety-Gymnasium / OmniSafe experiments
```

Then follow the folder-specific README. The cleaned release keeps the COP-Q
implementations intact while removing unused caches, backup files, and
irrelevant baseline code.

## Notes

- GPU training is supported when the correct JAX or PyTorch CUDA build is
  installed, but CPU execution is sufficient for code inspection and basic
  command checks.
- The two implementations are independent and should be installed/run in
  separate environments when possible.
