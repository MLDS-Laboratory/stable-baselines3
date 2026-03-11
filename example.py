import argparse
import json
from datetime import datetime

import minigrid
import novgrid

import wandb

# Import custom environments - this also registers thems
from envs import env_configs
from envs.guarded_maze_wandb import GuardedMazeWandbCallback
from stable_baselines3 import A2C, CPPO, MG, MVPI, PPO, XPO  # noqa: F401
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


def parse_xy(xy: str | None):
    if xy is None:
        return None
    values = [v.strip() for v in xy.split(',')]
    if len(values) != 2:
        raise ValueError("Expected coordinates as 'x,y'")
    return int(values[0]), int(values[1])


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--project', type=str, default=None)
parser.add_argument('--algorithm', type=str, default='ppo')
parser.add_argument('--algo_name', type=str, default='ppo')
parser.add_argument('--env_id', type=str, default='RiskyInvertedPendulum-v0')
parser.add_argument('--rollout_buffer_kwargs', type=json.loads, default={})
parser.add_argument('--gini_coef', type=float, default=None)
parser.add_argument('--steps', type=float, default=2e6)
parser.add_argument('--maze_log_freq', type=int, default=10_000)
parser.add_argument('--maze_non_cumulative', action='store_true')
parser.add_argument('--maze_fixed_start', action='store_true')
parser.add_argument('--maze_init_state', type=str, default=None)
parser.add_argument('--maze_goal_state', type=str, default=None)
args, _ = parser.parse_known_args()

algo_name = args.algorithm.upper()
try:
    AlgorithmClass = globals()[algo_name]
except KeyError:
    raise ValueError(  # noqa: B904
        f"Unknown algorithm '{args.algorithm}'. "
        f"Available: {[k for k,v in globals().items() if isinstance(v, type)]}"
    )


set_random_seed(args.seed)
project_name = args.project or f"{args.env_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
run = wandb.init(
    project=project_name,
    sync_tensorboard=True,
    config=vars(args)
)

monitor_kwargs = {'info_keywords': ('viol',)} if 'Risky' in args.env_id else {}


if args.env_id in env_configs:
    wrappers = [
        minigrid.wrappers.FlatObsWrapper] if 'MiniGrid' in args.env_id else []
    env = novgrid.NoveltyEnv(
        env_configs=env_configs[args.env_id],
        novelty_step=args.steps // 2,
        monitor_kwargs=monitor_kwargs,
        wrappers=wrappers
    )
else:
    wrapper_class = minigrid.wrappers.FlatObsWrapper if 'MiniGrid' in args.env_id else None
    env_kwargs = {}
    if 'GuardedMaze' in args.env_id:
        maze_init_state = parse_xy(args.maze_init_state)
        maze_goal_state = parse_xy(args.maze_goal_state)
        if maze_init_state is not None:
            env_kwargs['init_state'] = maze_init_state
            env_kwargs['fixed_reset'] = True
        if maze_goal_state is not None:
            env_kwargs['goal_state'] = maze_goal_state
        if args.maze_fixed_start:
            env_kwargs['fixed_reset'] = True

    env = make_vec_env(
        args.env_id,
        env_kwargs=env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_class=wrapper_class,
    )

kwargs = {}
if args.gini_coef:
    kwargs['gini_coef'] = args.gini_coef

model = AlgorithmClass(
    'MlpPolicy',
    env,
    rollout_buffer_kwargs=args.rollout_buffer_kwargs,
    verbose=1,
    tensorboard_log=f'runs/{run.id}',
    **kwargs
)

callbacks = []
if 'GuardedMaze' in args.env_id:
    callbacks.append(
        GuardedMazeWandbCallback(
            log_freq=args.maze_log_freq,
            cumulative=not args.maze_non_cumulative,
        )
    )

model.learn(
    total_timesteps=args.steps,
    callback=callbacks if callbacks else None,
)
