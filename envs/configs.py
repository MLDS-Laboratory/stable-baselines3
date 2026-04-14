from gymnasium.envs.registration import register

from .guarded_maze import GuardedMazeEnv
from .realworld_control import REALWORLD_ENV_CONFIGS, RealWorldControlEnv
from .risky_minigrid import RiskyMiniGridEnv
from .risky_mujoco import RiskyHalfCheetahEnv, RiskyInvertedPendulumEnv, RiskySwimmerEnv

# from .risky_pendulum import RiskyInvertedPendulumEnv


env_configs = {
    # MiniGrid environments
    'Safe2Risk1MiniGrid-7x7-v0': [
        {"env_id": "RiskyMiniGrid-7x7-v0", "is_risky": False},
        {"env_id": "RiskyMiniGrid-7x7-v0", "risk_level": 1}
    ],
    'Safe2Risk2MiniGrid-7x7-v0': [
        {"env_id": "RiskyMiniGrid-7x7-v0", "is_risky": False},
        {"env_id": "RiskyMiniGrid-7x7-v0", "risk_level": 2}
    ],
    'Safe2Risk3MiniGrid-7x7-v0': [
        {"env_id": "RiskyMiniGrid-7x7-v0", "is_risky": False},
        {"env_id": "RiskyMiniGrid-7x7-v0", "risk_level": 3}
    ],

    # Pendulum environments
    'Safe2Risk1Pendulum-v0': [
        {"env_id": "RiskyInvertedPendulum-v0", "is_risky": False},
        {"env_id": "RiskyInvertedPendulum-v0", "risk_level": 1}
    ],
    'Safe2Risk2Pendulum-v0': [
        {"env_id": "RiskyInvertedPendulum-v0", "is_risky": False},
        {"env_id": "RiskyInvertedPendulum-v0", "risk_level": 2}
    ],
    'Safe2Risk3Pendulum-v0': [
        {"env_id": "RiskyInvertedPendulum-v0", "is_risky": False},
        {"env_id": "RiskyInvertedPendulum-v0", "risk_level": 3}
    ],

    # RealWorld transitions (nominal -> novelty)
    'Nom2PerturbWalker-v0': [
        {"env_id": "walker_realworld_walk-v0"},
        {"env_id": "walker_perturb_novelty-v0"}
    ],
    'Nom2FrictionWalker-v0': [
        {"env_id": "walker_realworld_walk-v0"},
        {"env_id": "walker_friction_novelty-v0"}
    ],
    'Nom2NoiseWalker-v0': [
        {"env_id": "walker_realworld_walk-v0"},
        {"env_id": "walker_noise_novelty-v0"}
    ],
    'Nom2PerturbCartpole-v0': [
        {"env_id": "cartpole_realworld_balance-v0"},
        {"env_id": "cartpole_perturb_novelty-v0"}
    ],
    'Nom2DampingCartpole-v0': [
        {"env_id": "cartpole_realworld_balance-v0"},
        {"env_id": "cartpole_damping_novelty-v0"}
    ],
    'Nom2NoiseCartpole-v0': [
        {"env_id": "cartpole_realworld_balance-v0"},
        {"env_id": "cartpole_noise_novelty-v0"}
    ],
    'Nom2PerturbQuadruped-v0': [
        {"env_id": "quadruped_realworld_walk-v0"},
        {"env_id": "quadruped_perturb_novelty-v0"}
    ],
    'Nom2DampingQuadruped-v0': [
        {"env_id": "quadruped_realworld_walk-v0"},
        {"env_id": "quadruped_damping_novelty-v0"}
    ],
    'Nom2NoiseQuadruped-v0': [
        {"env_id": "quadruped_realworld_walk-v0"},
        {"env_id": "quadruped_noise_novelty-v0"}
    ],

}


def register_all_envs():
    """Register all custom environments with gymnasium."""
    register(
        id="RiskySwimmer-v0",
        entry_point=RiskySwimmerEnv,
        max_episode_steps=1000
    )
    register(
        id="RiskyHalfCheetah-v0",
        entry_point=RiskyHalfCheetahEnv,
        max_episode_steps=1000
    )
    register(
        id="RiskyInvertedPendulum-v0",
        entry_point=RiskyInvertedPendulumEnv,
        max_episode_steps=200
    )
    register(
        id="RiskyMiniGrid-7x7-v0",
        entry_point=RiskyMiniGridEnv,
    )
    register(
        id="GuardedMaze-8x8-v0",
        entry_point=GuardedMazeEnv,
        kwargs={"mode": 1},
    )
    register(
        id="GuardedMazePaper-8x8-v0",
        entry_point=GuardedMazeEnv,
        kwargs={
            "mode": 1,
            "continuous": True,
            "max_steps": 161,
            "guard_prob": 0.2,
            "guard_cost": 32,
            "rand_cost": True,
            "limit_step_penalty": False,
        },
    )
    register(
        id="GuardedMaze-16x16-v0",
        entry_point=GuardedMazeEnv,
        kwargs={"mode": 2},
    )
    register(
        id="GuardedMazePaper-16x16-v0",
        entry_point=GuardedMazeEnv,
        kwargs={
            "mode": 2,
            "continuous": True,
            "guard_prob": 0.2,
            "guard_cost": 32,
            "rand_cost": True,
            "limit_step_penalty": False,
        },
    )

    base_realworld_ids = {
        config["env_id"]
        for config in REALWORLD_ENV_CONFIGS.values()
    }
    for base_env_id in base_realworld_ids:
        register(
            id=f"{base_env_id}-v0",
            entry_point=RealWorldControlEnv,
            kwargs={"name": base_env_id},
            max_episode_steps=1000,
        )

    for config_name in REALWORLD_ENV_CONFIGS:
        register(
            id=f"{config_name}-v0",
            entry_point=RealWorldControlEnv,
            kwargs={"name": config_name},
            max_episode_steps=1000,
        )
