import json
import re
import subprocess
from uuid import uuid4

import gymnasium as gym
import ray

from envs import env_configs  # noqa: F401  # Import registers custom envs


@ray.remote(num_cpus=1)
def run_trial(args):
    cmd = ["python", "example.py", *args]
    subprocess.run(cmd, check=True)


def sanitize_wandb_project_name(name: str) -> str:
    sanitized = re.sub(r"[/\\#?%:]", "_", name)
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "sb3_project"


steps = "1e6"
seeds = [0, 10, 20, 30, 40]
# seeds = [0, 10, 20]
# envs = [
#     "Nom2PerturbWalker-v0",
#     # "Nom2FrictionWalker-v0",
#     # "Nom2NoiseWalker-v0",
#     "Nom2PerturbCartpole-v0",
#     # "Nom2DampingCartpole-v0",
#     # "Nom2NoiseCartpole-v0",
#     "Nom2PerturbQuadruped-v0",
#     # "Nom2DampingQuadruped-v0",
#     # "Nom2NoiseQuadruped-v0",
# ]
# envs = [
#     "Safe2Risk1MiniGrid-7x7-v0",
#     "Safe2Risk2MiniGrid-7x7-v0",
#     "Safe2Risk3MiniGrid-7x7-v0",
#     # "Safe2Risk1Pendulum-v0",
#     # "Safe2Risk2Pendulum-v0",
#     # "Safe2Risk3Pendulum-v0",
# ]
# envs = ['Safe2RiskyMiniGrid-7x7-v0', 'Safe2RiskyPendulum-v0']


def resolve_walker_env() -> str:
    candidates = [
        "dm_control/walker-run-v0",
        "walker_run-v0",
        "walker_realworld_walk-v0",
    ]
    for env_id in candidates:
        try:
            env = gym.make(env_id)
            env.close()
            return env_id
        except Exception:
            continue

    raise RuntimeError(
        "No walker environment found. Tried: "
        "dm_control/walker-run-v0, walker_run-v0, walker_realworld_walk-v0. "
        "Install dm_control-compatible env registration or use one of the local env IDs."
    )


envs = [resolve_walker_env()]
identifier = uuid4()

configs = {
    "ppo": [
        {"--algo_name": "ppo"},  # no extra params, just the algorithm name
    ],
    "rpo": [
        {"--algo_name": "rpo"}
    ],
    # "mg": [
    #     {"--gini_coef": 0.8, "--algo_name": "mg_0.8"},
    #     {"--gini_coef": 1.0, "--algo_name": "mg_1.0"},
    #     {"--gini_coef": 1.2, "--algo_name": "mg_1.2"},
    # ],
    # "xpo": [
    #     {"--rollout_buffer_kwargs": {"beta": -0.001}, "--algo_name": "xpo_-0.001"},
    #     {"--rollout_buffer_kwargs": {"beta": -0.005}, "--algo_name": "xpo_-0.005"},
    #     {"--rollout_buffer_kwargs": {"beta": -0.01},  "--algo_name": "xpo_-0.01"},
    # ],
    # "mvpi": [
    #     {"--rollout_buffer_kwargs": {"lam": 0.2}, "--algo_name": "mvpi_0.2"},
    #     {"--rollout_buffer_kwargs": {"lam": 0.4}, "--algo_name": "mvpi_0.4"},
    #     {"--rollout_buffer_kwargs": {"lam": 0.6}, "--algo_name": "mvpi_0.6"},
    # ],
}


trials = []
for env in envs:
    for seed in seeds:
        for algorithm, hyperparam_list in configs.items():
            for hyperparams in hyperparam_list:
                project_name = sanitize_wandb_project_name(
                    f"{env}-{identifier}")
                args = [
                    "--project", project_name,
                    "--seed", str(seed),
                    "--steps", steps,
                    "--env_id", env,
                    "--algorithm", algorithm,
                ]

                if "GuardedMaze" in env:
                    args += [
                        "--maze_fixed_start",
                        "--maze_init_state", "1,1",
                    ]

                for flag, value in hyperparams.items():
                    if isinstance(value, dict):
                        args += [flag, json.dumps(value)]
                    else:
                        args += [flag, str(value)]

                trials.append(args)

futures = [run_trial.remote(args) for args in trials]
ray.get(futures)
