import gymnasium as gym
import numpy as np

REALWORLD_ENV_CONFIGS = {
    "walker_perturb_novelty": {
        "env_id": "walker_realworld_walk",
        "perturb_spec": {
            "enable": True,
            "param": "thigh_length",
            "scheduler": "constant",
            "start": 0.55,
            "min": 0.55,
            "max": 0.55,
            "std": 0,
        },
    },
    "walker_friction_novelty": {
        "env_id": "walker_realworld_walk",
        "perturb_spec": {
            "enable": True,
            "param": "contact_friction",
            "scheduler": "constant",
            "start": 6,
            "min": 6,
            "max": 6,
            "std": 0,
        },
    },
    "walker_noise_novelty": {
        "env_id": "walker_realworld_walk",
        "noise_spec": {
            "gaussian": {
                "enable": True,
                "actions": 0.0,
                "observations": 5.0,
            }
        },
    },
    "cartpole_perturb_novelty": {
        "env_id": "cartpole_realworld_balance",
        "perturb_spec": {
            "enable": True,
            "param": "pole_mass",
            "scheduler": "constant",
            "start": 10,
            "min": 10,
            "max": 10,
            "std": 0,
        },
    },
    "cartpole_damping_novelty": {
        "env_id": "cartpole_realworld_balance",
        "perturb_spec": {
            "enable": True,
            "param": "joint_damping",
            "scheduler": "constant",
            "start": 4e-6,
            "min": 4e-6,
            "max": 4e-6,
            "std": 0,
        },
    },
    "cartpole_noise_novelty": {
        "env_id": "cartpole_realworld_balance",
        "noise_spec": {
            "gaussian": {
                "enable": True,
                "actions": 0.5,
                "observations": 0.5,
            }
        },
    },
    "quadruped_perturb_novelty": {
        "env_id": "quadruped_realworld_walk",
        "perturb_spec": {
            "enable": True,
            "param": "shin_length",
            "scheduler": "constant",
            "start": 4.0,
            "min": 4.0,
            "max": 4.0,
            "std": 0,
        },
    },
    "quadruped_damping_novelty": {
        "env_id": "quadruped_realworld_walk",
        "perturb_spec": {
            "enable": True,
            "param": "joint_damping",
            "scheduler": "constant",
            "start": 10.0,
            "min": 10.0,
            "max": 10.0,
            "std": 0,
        },
    },
    "quadruped_noise_novelty": {
        "env_id": "quadruped_realworld_walk",
        "noise_spec": {
            "gaussian": {
                "enable": True,
                "actions": 5.0,
                "observations": 5.0,
            }
        },
    },
}


class RealWorldControlEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, name: str, action_repeat: int = 1, seed: int = 0, render_mode: str = None):
        if render_mode not in (None, "rgb_array"):
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. Supported modes: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        config = dict(REALWORLD_ENV_CONFIGS.get(name, {"env_id": name}))
        env_id = config.pop("env_id")

        domain, task = env_id.split("_", 1)
        if domain == "cup":
            domain = "ball_in_cup"

        try:
            import realworldrl_suite.environments as rwrl
        except ImportError as error:
            raise ImportError(
                "realworldrl_suite is required for RealWorldControlEnv. "
                "Install it with: pip install realworldrl_suite"
            ) from error

        self._env = rwrl.load(
            domain,
            task,
            random=seed,
            **config,
        )
        self._action_repeat = action_repeat

        action_spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            action_spec.minimum, action_spec.maximum, dtype=np.float32)

        self._obs_keys = list(self._env.observation_spec().keys())
        low = []
        high = []
        for key in self._obs_keys:
            spec = self._env.observation_spec()[key]
            shape = (1,) if len(spec.shape) == 0 else spec.shape
            size = int(np.prod(shape))
            low.append(np.full(size, -np.inf, dtype=np.float32))
            high.append(np.full(size, np.inf, dtype=np.float32))
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(low),
            high=np.concatenate(high),
            dtype=np.float32,
        )

    def _flatten_obs(self, obs_dict):
        flat_obs = []
        for key in self._obs_keys:
            value = np.asarray(obs_dict[key], dtype=np.float32)
            if value.shape == ():
                value = value.reshape(1)
            flat_obs.append(value.reshape(-1))
        return np.concatenate(flat_obs, dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not np.isfinite(action).all():
            raise ValueError(f"Action has non-finite values: {action}")

        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += float(time_step.reward or 0.0)
            if time_step.last():
                break

        obs = self._flatten_obs(time_step.observation)
        terminated = bool(time_step.last())
        truncated = False
        info = {
            "discount": np.array(time_step.discount, dtype=np.float32),
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            try:
                self._env.task.random.seed(seed)
            except AttributeError:
                pass
        time_step = self._env.reset()
        obs = self._flatten_obs(time_step.observation)
        return obs, {}

    def render(self):
        if self.render_mode is None:
            return None
        return self._env.physics.render(64, 64, camera_id=0)

    def close(self):
        if hasattr(self._env, "physics") and hasattr(self._env.physics, "free"):
            self._env.physics.free()
        if hasattr(self._env, "close"):
            self._env.close()
