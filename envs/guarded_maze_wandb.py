from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import wandb
from stable_baselines3.common.callbacks import BaseCallback


class GuardedMazeWandbCallback(BaseCallback):
    def __init__(
        self,
        log_freq: int = 10_000,
        cumulative: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.log_freq = log_freq
        self.cumulative = cumulative
        self._counts: Optional[np.ndarray] = None
        self._rows: Optional[int] = None
        self._cols: Optional[int] = None
        self._maze_map: Optional[np.ndarray] = None
        self._goal_map: Optional[np.ndarray] = None
        self._free_mask: Optional[np.ndarray] = None
        self._guard_mask: Optional[np.ndarray] = None
        self._last_logged_step = 0
        self._visited_xy: list[np.ndarray] = []
        self._start_xy: Optional[np.ndarray] = None
        self._prev_in_guard: Optional[np.ndarray] = None
        self._guard_entries_total = 0
        self._guard_samples_total = 0
        self._guard_entries_window = 0
        self._guard_samples_window = 0

    def _on_training_start(self) -> None:
        rows = int(self.training_env.get_attr("rows")[0])
        cols = int(self.training_env.get_attr("cols")[0])
        maze_map = np.array(self.training_env.get_attr("map")[0], dtype=float)
        goal_map = np.array(self.training_env.get_attr("goal")[0], dtype=float)

        self._rows = rows
        self._cols = cols
        self._maze_map = maze_map
        self._goal_map = goal_map
        self._free_mask = maze_map <= 0
        self._guard_mask = maze_map < 0
        self._counts = np.zeros((rows, cols), dtype=np.float64)

        start_obs = np.array(self.training_env.get_attr(
            "state_xy"), dtype=np.float32)
        start_obs = self._extract_xy(start_obs)
        if start_obs.shape[0] > 0:
            self._start_xy = start_obs[0].copy()
        self._prev_in_guard = self._xy_in_guard(start_obs)

    def _xy_in_guard(self, xy_batch: np.ndarray) -> np.ndarray:
        if self._guard_mask is None:
            return np.zeros((xy_batch.shape[0],), dtype=bool)
        x = np.clip(np.round(xy_batch[:, 0]).astype(int), 0, self._rows - 1)
        y = np.clip(np.round(xy_batch[:, 1]).astype(int), 0, self._cols - 1)
        return self._guard_mask[x, y]

    def _extract_xy(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if obs.ndim == 1:
            obs = obs[None, :]
        if obs.shape[-1] != 2:
            return np.empty((0, 2), dtype=np.float32)
        return obs[:, :2]

    def _on_step(self) -> bool:
        if self._counts is None:
            return True

        obs = self.locals.get("new_obs", self.locals.get("obs", None))

        if obs is not None:
            xy_batch = self._extract_xy(obs)
            if xy_batch.size > 0:
                xy_batch = xy_batch.astype(np.float32)
                xy_batch[:, 0] = np.clip(xy_batch[:, 0], 0, self._rows - 1)
                xy_batch[:, 1] = np.clip(xy_batch[:, 1], 0, self._cols - 1)

                hist2d, _, _ = np.histogram2d(
                    xy_batch[:, 0],
                    xy_batch[:, 1],
                    bins=[self._rows, self._cols],
                    range=[[-0.5, self._rows - 0.5], [-0.5, self._cols - 0.5]],
                )
                self._counts += hist2d
                self._visited_xy.extend(xy_batch)

                infos = self.locals.get("infos", None)
                dones = self.locals.get("dones", None)

                transition_xy = xy_batch.copy()
                if infos is not None and dones is not None:
                    dones_arr = np.asarray(dones, dtype=bool)
                    for i, done in enumerate(dones_arr):
                        if done and i < len(infos):
                            terminal_obs = infos[i].get(
                                "terminal_observation", None)
                            if terminal_obs is not None:
                                terminal_xy = self._extract_xy(
                                    np.asarray(terminal_obs))
                                if terminal_xy.shape[0] > 0:
                                    transition_xy[i] = terminal_xy[0]

                curr_in_guard_transition = self._xy_in_guard(transition_xy)
                if self._prev_in_guard is None or self._prev_in_guard.shape[0] != curr_in_guard_transition.shape[0]:
                    self._prev_in_guard = np.zeros_like(
                        curr_in_guard_transition, dtype=bool)

                entered_guard = (
                    ~self._prev_in_guard) & curr_in_guard_transition
                entry_count = int(np.sum(entered_guard))
                sample_count = int(curr_in_guard_transition.size)
                self._guard_entries_total += entry_count
                self._guard_samples_total += sample_count
                self._guard_entries_window += entry_count
                self._guard_samples_window += sample_count

                self._prev_in_guard = self._xy_in_guard(xy_batch)

        if self.num_timesteps - self._last_logged_step >= self.log_freq:
            self._log_scatter()
            self._last_logged_step = self.num_timesteps
            if not self.cumulative:
                self._counts.fill(0)
                self._visited_xy.clear()

        return True

    def _log_scatter(self) -> None:
        if self._counts is None or self._free_mask is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=120)

        if self._maze_map is not None:
            bg = np.zeros((self._rows, self._cols, 3), dtype=np.float32)

            walls = self._maze_map > 0
            bg[walls] = np.array([0.55, 0.55, 0.55], dtype=np.float32)

            if self._guard_mask is not None:
                bg[self._guard_mask] = np.array(
                    [0.85, 0.0, 0.0], dtype=np.float32)

            if self._goal_map is not None:
                goal_cells = self._goal_map > 0
                bg[goal_cells] = np.array([0.0, 0.8, 0.0], dtype=np.float32)

            ax.imshow(bg.swapaxes(0, 1), origin="lower",
                      interpolation="nearest")

        if len(self._visited_xy) > 0:
            visited = np.asarray(self._visited_xy, dtype=np.float32)
            ax.scatter(
                visited[:, 0],
                visited[:, 1],
                s=10,
                c="yellow",
                alpha=0.08,
                edgecolors="none",
                linewidths=0,
                rasterized=True,
            )

        if self._start_xy is not None:
            ax.scatter(
                self._start_xy[0],
                self._start_xy[1],
                s=160,
                marker="^",
                c="yellow",
                edgecolors="black",
                linewidths=1.0,
                alpha=1.0,
                zorder=5,
            )

        ax.set_xlim(-0.5, self._rows - 0.5)
        ax.set_ylim(-0.5, self._cols - 0.5)
        ax.set_aspect("equal")
        ax.set_title("Guarded Maze trajectory dots (density by opacity)")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        free_cells = int(np.sum(self._free_mask))
        visited_cells = int(np.sum((self._counts > 0) & self._free_mask))
        coverage = visited_cells / max(1, free_cells)
        guard_entry_rate = self._guard_entries_total / \
            max(1, self._guard_samples_total)
        guard_entry_rate_window = self._guard_entries_window / \
            max(1, self._guard_samples_window)

        wandb.log(
            {
                "guarded_maze/trajectory_scatter": wandb.Image(fig),
                "guarded_maze/coverage": coverage,
                "guarded_maze/visited_cells": visited_cells,
                "guarded_maze/free_cells": free_cells,
                "guarded_maze/guard_entry_count": self._guard_entries_total,
                "guarded_maze/guard_entry_rate": guard_entry_rate,
                "guarded_maze/guard_entry_rate_window": guard_entry_rate_window,
                "global_step": self.num_timesteps,
            },
            step=self.num_timesteps,
        )
        plt.close(fig)

        if not self.cumulative:
            self._guard_entries_window = 0
            self._guard_samples_window = 0

    def _on_training_end(self) -> None:
        self._log_scatter()
