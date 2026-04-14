import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from envs import env_configs  # noqa: F401 (import registers custom envs)
from stable_baselines3 import CPPO, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


class CVaREvalCallback(BaseCallback):
    def __init__(
        self,
        env_id: str,
        env_kwargs: dict,
        alpha: float,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
        seed: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.alpha = alpha
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.seed = seed

        self.timesteps: list[int] = []
        self.cvars: list[float] = []
        self.mean_returns: list[float] = []
        self._last_eval_timestep = 0

    @staticmethod
    def _compute_cvar(returns: np.ndarray, alpha: float) -> float:
        sorted_returns = np.sort(returns)
        tail_count = max(1, int(np.ceil(alpha * len(sorted_returns))))
        return float(np.mean(sorted_returns[:tail_count]))

    def _evaluate(self) -> None:
        eval_env = gym.make(self.env_id, **self.env_kwargs)
        returns = []

        for episode_idx in range(self.n_eval_episodes):
            obs, _ = eval_env.reset(seed=self.seed + episode_idx)
            done = False
            episode_return = 0.0

            while not done:
                action, _ = self.model.predict(
                    obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_return += float(reward)
                done = bool(terminated or truncated)

            returns.append(episode_return)

        eval_env.close()
        returns_array = np.asarray(returns, dtype=np.float32)

        cvar = self._compute_cvar(returns_array, self.alpha)
        mean_return = float(np.mean(returns_array))

        self.timesteps.append(self.num_timesteps)
        self.cvars.append(cvar)
        self.mean_returns.append(mean_return)

        if self.verbose > 0:
            print(
                f"[Eval] step={self.num_timesteps} "
                f"mean_return={mean_return:.3f} cvar@{self.alpha:.2f}={cvar:.3f}"
            )

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_timestep >= self.eval_freq:
            self._evaluate()
            self._last_eval_timestep = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if len(self.timesteps) == 0 or self.timesteps[-1] != self.num_timesteps:
            self._evaluate()


def parse_seeds(seeds: str) -> list[int]:
    return [int(x.strip()) for x in seeds.split(",") if x.strip()]


def parse_algorithms(algorithms: str) -> list[str]:
    return [x.strip().lower() for x in algorithms.split(",") if x.strip()]


def apply_paper_cgm_preset(args: argparse.Namespace) -> None:
    if not args.paper_cgm_preset:
        return
    if args.env_id == "GuardedMaze-8x8-v0":
        args.env_id = "GuardedMazePaper-8x8-v0"
    if args.total_timesteps == 200_000:
        args.total_timesteps = 1_000_000
    if args.cvar_alpha == 0.2:
        args.cvar_alpha = 0.05
    args.maze_fixed_start = True
    if args.cppo_minimum_return_cap is None:
        args.cppo_minimum_return_cap = -151.0


def build_env_kwargs(args: argparse.Namespace) -> dict:
    env_kwargs: dict = {}
    if "GuardedMaze" not in args.env_id:
        return env_kwargs
    if args.maze_init_state is not None:
        x_str, y_str = [v.strip() for v in args.maze_init_state.split(",")]
        env_kwargs["init_state"] = (int(x_str), int(y_str))
    if args.maze_fixed_start:
        env_kwargs["fixed_reset"] = True
    return env_kwargs


def build_model_kwargs(args: argparse.Namespace) -> tuple[dict[str, float | int], dict[str, float | int]]:
    if not args.paper_cgm_preset:
        return {}, {}

    ppo_model_kwargs: dict[str, float | int] = {
        "learning_rate": 1e-3,
        "n_steps": 10_000,
        "n_epochs": 6,
        "batch_size": 50,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 1e-5,
    }
    cppo_model_kwargs: dict[str, float | int] = {
        **ppo_model_kwargs,
        "cap_alpha": args.cvar_alpha,
        "cap_tau": 0.2,
        "initial_return_cap": (
            args.cppo_minimum_return_cap
            if args.cppo_minimum_return_cap is not None
            else -0.1
        ),
        "minimum_return_cap": (
            args.cppo_minimum_return_cap
            if args.cppo_minimum_return_cap is not None
            else -0.1
        ),
    }
    return ppo_model_kwargs, cppo_model_kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GuardedMaze-8x8-v0")
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=30)
    parser.add_argument("--cvar_alpha", type=float, default=0.2)
    parser.add_argument("--seeds", type=str, default="0,10,20")
    parser.add_argument("--algorithms", type=str, default="ppo,cppo")
    parser.add_argument("--deterministic_eval", action="store_true")
    parser.add_argument("--output_dir", type=str,
                        default="results/cvar_guarded_maze")
    parser.add_argument("--maze_fixed_start",
                        action="store_true", default=True)
    parser.add_argument("--maze_init_state", type=str, default="1,1")
    parser.add_argument("--paper_cgm_preset", action="store_true")
    parser.add_argument("--cppo_minimum_return_cap", type=float, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    apply_paper_cgm_preset(args)

    if not (0 < args.cvar_alpha <= 1.0):
        raise ValueError("cvar_alpha must be in (0, 1].")

    seeds = parse_seeds(args.seeds)
    selected_algorithms = parse_algorithms(args.algorithms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = build_env_kwargs(args)

    available_algorithms = {
        "ppo": PPO,
        "cppo": CPPO,
    }

    unknown_algorithms = [
        algo for algo in selected_algorithms if algo not in available_algorithms]
    if unknown_algorithms:
        raise ValueError(
            f"Unknown algorithms: {unknown_algorithms}. "
            f"Available: {sorted(available_algorithms.keys())}"
        )

    algorithms = {algo: available_algorithms[algo]
                  for algo in selected_algorithms}

    ppo_model_kwargs, cppo_model_kwargs = build_model_kwargs(args)

    raw_results: dict[str, dict[int, dict[str, list[float] | list[int]]]] = {
        algo: {} for algo in algorithms
    }

    for algo_name, algo_cls in algorithms.items():
        for seed in seeds:
            if args.verbose > 0:
                print(f"\n=== Training {algo_name.upper()} | seed={seed} ===")

            set_random_seed(seed)
            train_env = make_vec_env(
                args.env_id,
                seed=seed,
                env_kwargs=env_kwargs,
            )

            eval_callback = CVaREvalCallback(
                env_id=args.env_id,
                env_kwargs=env_kwargs,
                alpha=args.cvar_alpha,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=args.deterministic_eval,
                seed=seed,
                verbose=args.verbose,
            )

            model = algo_cls(
                "MlpPolicy",
                train_env,
                seed=seed,
                verbose=0,
                **(cppo_model_kwargs if algo_name == "cppo" else ppo_model_kwargs),
            )

            model.learn(total_timesteps=args.total_timesteps,
                        callback=eval_callback)
            train_env.close()

            raw_results[algo_name][seed] = {
                "timesteps": eval_callback.timesteps,
                "cvar": eval_callback.cvars,
                "mean_return": eval_callback.mean_returns,
            }

    # Align and aggregate curves across seeds per algorithm
    aggregated: dict[str, dict[str, list[float] | list[int]]] = {}
    for algo_name in algorithms:
        seed_results = raw_results[algo_name]
        if not seed_results:
            continue

        first_seed = seeds[0]
        timesteps = seed_results[first_seed]["timesteps"]
        cvar_matrix = []
        for seed in seeds:
            result = seed_results[seed]
            if result["timesteps"] != timesteps:
                raise RuntimeError(
                    f"Mismatched eval timesteps for {algo_name} seed={seed}."
                )
            cvar_matrix.append(result["cvar"])

        cvar_array = np.asarray(cvar_matrix, dtype=np.float32)
        aggregated[algo_name] = {
            "timesteps": timesteps,
            "cvar_mean": np.mean(cvar_array, axis=0).tolist(),
            "cvar_std": np.std(cvar_array, axis=0).tolist(),
        }

    with (output_dir / "raw_cvar_results.json").open("w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)

    with (output_dir / "aggregated_cvar_results.json").open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    plt.figure(figsize=(8, 5), dpi=120)
    for algo_name in algorithms:
        if algo_name not in aggregated:
            continue
        timesteps = np.asarray(
            aggregated[algo_name]["timesteps"], dtype=np.int32)
        mean_curve = np.asarray(
            aggregated[algo_name]["cvar_mean"], dtype=np.float32)
        std_curve = np.asarray(
            aggregated[algo_name]["cvar_std"], dtype=np.float32)
        plt.plot(timesteps, mean_curve, label=algo_name.upper())
        plt.fill_between(timesteps, mean_curve - std_curve,
                         mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel(f"CVaR@{args.cvar_alpha:.1f} (episodic return)")
    plt.title(f"{args.env_id}: PPO vs CPPO")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "cvar_curve.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"\nSaved plot to: {plot_path}")
    print(f"Saved raw results to: {output_dir / 'raw_cvar_results.json'}")
    print(
        f"Saved aggregated results to: {output_dir / 'aggregated_cvar_results.json'}")


if __name__ == "__main__":
    main()
