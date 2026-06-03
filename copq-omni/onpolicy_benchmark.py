import argparse


def parse_gpu_ids(value: str | None) -> list[int] | None:
    if value is None or value.strip().lower() in {"", "cpu", "none"}:
        return None
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run on-policy Safety-Gymnasium baselines.")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU IDs, e.g. 0 or 0,1. Default: CPU.")
    parser.add_argument("--num-pool", type=int, default=24, help="Number of parallel experiment workers.")
    parser.add_argument("--env-id", default="SafetyPointPush1-v0", help="Safety-Gymnasium environment ID.")
    parser.add_argument("--num-seeds", type=int, default=6)
    parser.add_argument("--seed-stride", type=int, default=5)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    from omnisafe.common.experiment_grid import ExperimentGrid
    from omnisafe.utils.exp_grid_tools import train

    eg = ExperimentGrid(exp_name='copq-onpolicy')

    # set up the algorithms.
    on_policys = ['TRPOPID', 'RCPO', 'PPOSaute', 'CUP']
    eg.add(
        'algo',
        on_policys
    )

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])

    # the default configs here are as follows:
    # eg.add('algo_cfgs:steps_per_epoch', [20000])
    # eg.add('train_cfgs:total_steps', [20000 * 500])
    # which can reproduce results of 1e7 steps.

    # if you want to reproduce results of 1e6 steps, using
    eg.add('algo_cfgs:steps_per_epoch', [2000])
    eg.add('train_cfgs:total_steps', [2000 * 600])
    eg.add('train_cfgs:torch_threads', [1])
    eg.add('algo_cfgs:gamma', [0.975])
    eg.add('algo_cfgs:cost_gamma', [0.975])

    gpu_id = parse_gpu_ids(args.gpu_ids)

    # set up the environment.
    eg.add('env_id', [args.env_id])
    eg.add('seed', [args.seed_stride * i for i in range(args.num_seeds)])

    # total experiment num must can be divided by num_pool.
    # meanwhile, users should decide this value according to their machine.
    eg.run(train, num_pool=args.num_pool, gpu_id=gpu_id)
