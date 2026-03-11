import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, RSPPO, A2C
from stable_baselines3.common.utils import set_random_seed
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.registration import register
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, RSPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from numpy.linalg import norm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import argparse

class CustomGravityWrapper(gym.Wrapper):
    def __init__(self, env, gravity=[0, 0, -9.81]):
        super().__init__(env)
        if hasattr(self.env.unwrapped, "model"):
            self.env.unwrapped.model.opt.gravity[:] = gravity
        else:
            raise AttributeError("This environment doesn't have a MuJoCo model.")
        

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
parser.add_argument('env_id', type=str)
parser.add_argument('beta', type=float)

args, _ = parser.parse_known_args()

if args.beta < 0:
    beta_str = f'N{abs(args.beta):.3f}'.replace('.', '')
else:
    beta_str = f'{args.beta:.3f}'.replace('.', '')
run = wandb.init(project=f'{args.env_id}-TD(lambda)**', sync_tensorboard=True, name=f'{args.env_id}-{args.seed}-{beta_str}')

set_random_seed(args.seed)
env = make_vec_env(args.env_id)

eval_env = make_vec_env(args.env_id)

eval_callback= EvalCallback(
    eval_env, best_model_save_path=f'./logs/{run.id}',
    log_path=f'./logs/{run.id}', eval_freq=10000,
    n_eval_episodes=100, deterministic=False, render=False
)

if not args.beta:
    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=f'runs/{run.id}')
else:   
    model = RSPPO('MlpPolicy', env, verbose=0, tensorboard_log=f'runs/{run.id}', rollout_buffer_kwargs={'beta': args.beta})
model.learn(total_timesteps=1e6, callback=[eval_callback])
model = PPO.load(f'./logs/{run.id}/best_model')

if 'LunarLander' in args.env_id:
    model_var = [2, 3, 4, 5, 6, 7]
    for var in model_var:
        env_kwargs = {'gravity': -10 + var}
        env = make_vec_env(args.env_id, env_kwargs=env_kwargs)
        mean_reward, std_reward = evaluate_policy(model, env, deterministic=False, n_eval_episodes=100)
        wandb.log({f'eval/{env_kwargs["gravity"]}': mean_reward})
else:
    model_var = [-6, -4, -2, 0, 2, 4, 6]
    for var in model_var:
        gravity = np.array([0, 0, -9.81 + var])
        env = make_vec_env(lambda: CustomGravityWrapper(
            gym.make(args.env_id), gravity=gravity)
        )
        mean_reward, std_reward = evaluate_policy(model, env, deterministic=False, n_eval_episodes=100)
        wandb.log({f'eval/{gravity[-1]}': mean_reward})
wandb.log({'seed': args.seed})
wandb.log({'beta': args.beta})



