import argparse


def parse_gpu_ids(value: str | None) -> list[int] | None:
    if value is None or value.strip().lower() in {"", "cpu", "none"}:
        return None
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COP-Q off-policy Safety-Gymnasium experiments.")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU IDs, e.g. 0 or 0,1. Default: CPU.")
    parser.add_argument("--num-pool", type=int, default=12, help="Number of parallel experiment workers.")
    parser.add_argument("--env-id", default="SafetyPointGoal2-v0", help="Safety-Gymnasium environment ID.")
    parser.add_argument("--num-seeds", type=int, default=6)
    parser.add_argument("--seed-stride", type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    from omnisafe.common.experiment_grid import ExperimentGrid
    from omnisafe.utils.exp_grid_tools import train

    eg = ExperimentGrid(exp_name='copq-offpolicy')

    eg.add('algo', ['COPQ'])

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])

    eg.add('algo_cfgs:convex', [10.])
    eg.add('algo_cfgs:steps_per_epoch', [2000])
    eg.add('train_cfgs:total_steps', [2000 * 600])
    eg.add('train_cfgs:torch_threads', [2])
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('lagrange_cfgs:cost_limit', [10.])
    eg.add('lagrange_cfgs:lagrangian_multiplier_init', [0.001])
    eg.add('logger_cfgs:window_lens', [1])
    eg.add('algo_cfgs:gamma', [0.975])
    eg.add('algo_cfgs:warmup_epochs', [0])
    eg.add('algo_cfgs:budget', [1.])
    eg.add('algo_cfgs:alpha', [0.00001])
    eg.add('algo_cfgs:auto_alpha', [False])


    gpu_id = parse_gpu_ids(args.gpu_ids)

    # set up the environments.
    eg.add('env_id', [args.env_id])
    eg.add('seed', [i * args.seed_stride for i in range(args.num_seeds)])
    eg.run(train, num_pool=args.num_pool, gpu_id=gpu_id)
