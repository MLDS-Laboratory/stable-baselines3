import argparse
import json
import subprocess
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import ray


@ray.remote(num_cpus=1)
def run_trial(args: list[str]) -> None:
    cmd = ["python", "experiment_cvar.py", *args]
    subprocess.run(cmd, check=True)


def parse_seeds(seeds: str) -> list[int]:
    return [int(x.strip()) for x in seeds.split(",") if x.strip()]


def parse_algorithms(algorithms: str) -> list[str]:
    return [x.strip().lower() for x in algorithms.split(",") if x.strip()]


def apply_paper_cgm_preset(args: argparse.Namespace) -> None:
    if not args.paper_cgm_preset:
        return
    if args.env_id == "GuardedMaze-8x8-v0":
        args.env_id = "GuardedMazePaper-8x8-v0"
    if args.total_timesteps == 1_000_000:
        args.total_timesteps = 1_000_000
    if args.cvar_alpha == 0.2:
        args.cvar_alpha = 0.05
    args.maze_fixed_start = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="GuardedMaze-8x8-v0")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=30)
    parser.add_argument("--cvar_alpha", type=float, default=0.2)
    parser.add_argument("--seeds", type=str, default="0,10,20,30,40")
    parser.add_argument("--algorithms", type=str, default="ppo,cppo")
    parser.add_argument("--deterministic_eval", action="store_true")
    parser.add_argument("--output_dir", type=str,
                        default="results/cvar_guarded_maze_ray")
    parser.add_argument("--maze_fixed_start",
                        action="store_true", default=True)
    parser.add_argument("--maze_init_state", type=str, default="1,1")
    parser.add_argument("--paper_cgm_preset", action="store_true")
    parser.add_argument("--cppo_minimum_return_cap", type=float, default=None)
    parser.add_argument("--num_cpus", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    apply_paper_cgm_preset(args)

    if not (0 < args.cvar_alpha <= 1.0):
        raise ValueError("cvar_alpha must be in (0, 1].")

    seeds = parse_seeds(args.seeds)
    algorithms = parse_algorithms(args.algorithms)

    run_id = str(uuid4())[:8]
    output_dir = Path(args.output_dir)
    trial_root = output_dir / f"trials_{run_id}"
    trial_root.mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus)

    futures = []
    trial_paths: dict[str, dict[int, Path]] = {algo: {} for algo in algorithms}

    for algo in algorithms:
        for seed in seeds:
            trial_dir = trial_root / f"{algo}_seed{seed}"
            trial_paths[algo][seed] = trial_dir

            trial_args = [
                "--env_id",
                args.env_id,
                "--total_timesteps",
                str(args.total_timesteps),
                "--eval_freq",
                str(args.eval_freq),
                "--n_eval_episodes",
                str(args.n_eval_episodes),
                "--cvar_alpha",
                str(args.cvar_alpha),
                "--seeds",
                str(seed),
                "--algorithms",
                algo,
                "--output_dir",
                str(trial_dir),
                "--verbose",
                str(args.verbose),
            ]

            if args.deterministic_eval:
                trial_args.append("--deterministic_eval")
            if args.maze_fixed_start:
                trial_args.append("--maze_fixed_start")
            if args.maze_init_state is not None:
                trial_args.extend(["--maze_init_state", args.maze_init_state])
            if args.paper_cgm_preset:
                trial_args.append("--paper_cgm_preset")
            if args.cppo_minimum_return_cap is not None:
                trial_args.extend([
                    "--cppo_minimum_return_cap",
                    str(args.cppo_minimum_return_cap),
                ])

            futures.append(run_trial.remote(trial_args))

    ray.get(futures)

    raw_results: dict[str, dict[int, dict[str, list[float] | list[int]]]] = {
        algo: {} for algo in algorithms
    }

    for algo in algorithms:
        for seed in seeds:
            trial_dir = trial_paths[algo][seed]
            trial_raw_path = trial_dir / "raw_cvar_results.json"
            with trial_raw_path.open("r", encoding="utf-8") as f:
                trial_raw = json.load(f)

            run_data = trial_raw[algo][str(seed)]
            raw_results[algo][seed] = {
                "timesteps": run_data["timesteps"],
                "cvar": run_data["cvar"],
                "mean_return": run_data["mean_return"],
            }

    aggregated: dict[str, dict[str, list[float] | list[int]]] = {}
    for algo in algorithms:
        first_seed = seeds[0]
        timesteps = raw_results[algo][first_seed]["timesteps"]

        cvar_matrix = []
        for seed in seeds:
            seed_timesteps = raw_results[algo][seed]["timesteps"]
            if seed_timesteps != timesteps:
                raise RuntimeError(
                    f"Mismatched eval timesteps for {algo} seed={seed}."
                )
            cvar_matrix.append(raw_results[algo][seed]["cvar"])

        cvar_array = np.asarray(cvar_matrix, dtype=np.float32)
        aggregated[algo] = {
            "timesteps": timesteps,
            "cvar_mean": np.mean(cvar_array, axis=0).tolist(),
            "cvar_std": np.std(cvar_array, axis=0).tolist(),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "raw_cvar_results.json").open("w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)

    with (output_dir / "aggregated_cvar_results.json").open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    plt.figure(figsize=(8, 5), dpi=120)
    for algo in algorithms:
        timesteps = np.asarray(aggregated[algo]["timesteps"], dtype=np.int32)
        mean_curve = np.asarray(
            aggregated[algo]["cvar_mean"], dtype=np.float32)
        std_curve = np.asarray(aggregated[algo]["cvar_std"], dtype=np.float32)

        plt.plot(timesteps, mean_curve, label=algo.upper())
        plt.fill_between(timesteps, mean_curve - std_curve,
                         mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel(f"CVaR@{args.cvar_alpha:.1f} (episodic return)")
    plt.title(f"{args.env_id}: {' vs '.join(a.upper() for a in algorithms)}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "cvar_curve.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot to: {plot_path}")
    print(f"Saved raw results to: {output_dir / 'raw_cvar_results.json'}")
    print(
        f"Saved aggregated results to: {output_dir / 'aggregated_cvar_results.json'}")


if __name__ == "__main__":
    main()
