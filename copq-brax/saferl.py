import argparse
import functools
import os
from pathlib import Path
import random

import numpy as np


CONSTRAINTS = {
    "hopper": 74.02,
    "walker2d": 234.15,
    "ant": 262.22,
    "humanoid": 141.19,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COP-Q Brax experiments.")
    parser.add_argument(
        "--env",
        default="humanoid",
        choices=sorted(CONSTRAINTS),
        help="Brax locomotion environment.",
    )
    parser.add_argument("--num-obj", dest="num_obj", type=int, default=2)
    parser.add_argument(
        "--save-policy",
        dest="save_policy",
        action="store_true",
        help="Save final policy parameters.",
    )
    parser.add_argument(
        "--model-mode",
        dest="model_mode",
        default="mo",
        choices=["mo", "independent", "gaussian"],
        help="Critic output mode. Use 'mo' for joint reward/safety critics.",
    )
    parser.add_argument(
        "--task-name",
        dest="task_name",
        default="lrl",
        choices=["lrl", "crl"],
        help="'lrl' for hard-safety locomotion, 'crl' for safe-velocity CRL.",
    )
    parser.add_argument(
        "--learning-method",
        dest="learning_method",
        default="cop",
        choices=["cop", "cop-q", "worst_of_both", "scalar", "saclag", "saclag_ucb", "cal"],
        help="Loss branch used by sac/losses.py.",
    )
    parser.add_argument(
        "--explore-method",
        dest="explore_method",
        default="boundary",
        choices=["boundary", "orac"],
    )
    parser.add_argument("--exploration", action="store_true")
    parser.add_argument("--redq", "--reqd", dest="redq", action="store_true")
    parser.add_argument("--save-file", dest="save_file", default="copq")
    parser.add_argument("--output-dir", dest="output_dir", default="results")
    parser.add_argument("--num-seeds", dest="num_seeds", type=int, default=10)
    parser.add_argument("--seed-stride", dest="seed_stride", type=int, default=5)
    parser.add_argument("--ensemble-size", dest="ensemble_size", type=int, default=3)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--convex", type=float, default=10.0)
    parser.add_argument("--env-steps", dest="env_steps", type=int, default=3_072_000)
    parser.add_argument("--nb-layers", dest="nb_layers", type=int, default=2)
    parser.add_argument("--hidden-size", dest="hidden_size", type=int, default=256)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--budget", type=float, default=2.5)
    parser.add_argument("--budget-st", dest="budget_st", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="JAX platform. Default is CPU so the script is usable without a GPU.",
    )
    parser.add_argument(
        "--gpu-id",
        dest="gpu_id",
        default=None,
        help="CUDA device id(s), used only with --device gpu.",
    )

    return parser.parse_args()


def configure_jax_runtime(args: argparse.Namespace) -> None:
    os.environ["JAX_DETERMINISTIC"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
    if args.device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def main():
    args = parse_args()
    configure_jax_runtime(args)

    import jax
    from custom_envs.env_construct import get_environment
    from sac import train as sac

    random.seed(1)
    np.random.seed(1)

    seeds = [args.seed_stride * i for i in range(args.num_seeds)]
    output_dir = Path(args.output_dir) / args.env
    output_dir.mkdir(parents=True, exist_ok=True)

    train_fn = functools.partial(sac.train, num_timesteps=args.env_steps, num_evals=101, task_name=args.task_name,
                  episode_length=1000, normalize_observations=True, action_repeat=1, target_delay=1,
                  discounting=0.99, learning_rate=3e-4, q_learning_rate=3e-4, num_envs=64, 
                  batch_size=args.batch_size, num_eval_envs=10, multiplier_learning_rate=1e-5,
                  grad_updates_per_step=64, max_devices_per_host=1, max_replay_size=args.env_steps, 
                  min_replay_size=16384, network_size = (args.hidden_size,)*args.nb_layers, 
                  )

    X, Y, R = [], [], []
    for i, seed in enumerate(seeds):
        print(f"seed {i + 1}/{len(seeds)}: {seed}")
        env = get_environment(args.env, backend="generalized")
        x, y, rew, params = train_fn(environment=env,
                                     name=args.env, 
                                     mode=args.model_mode,
                                     num_obj=args.num_obj,
                                    ensemble_size=args.ensemble_size,
                                    seed=seed,
                                    method = args.learning_method,
                                    cost_limit=CONSTRAINTS[args.env],
                                    budget = args.budget,
                                    budget_st = args.budget_st,
                                    convex_coeff= args.convex,
                                    beta = args.beta,
                                    exploration_strategy = args.exploration,
                                    exploration_method = args.explore_method,
                                    redq=args.redq,
                                    delta = 4.,
                                    )
        if args.save_policy:
            from brax.io import model

            model.save_params(output_dir / f"{args.save_file}_policy", params)
        
        jax.clear_caches()
        X.append(x)
        Y.append(y)
        R.append(rew)

    X = np.array(X)
    Y = np.array(Y)
    R = np.array(R)

    print(Y.shape)

    np.savez(output_dir / f"{args.save_file}.npz", x=X, y=Y, replay_rewards=R)

if __name__ == "__main__":
    main()
