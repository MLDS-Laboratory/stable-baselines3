import warnings
from typing import Any

import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils import seeding

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

MAZE_SIZE = {1: 8, 2: 16}
MAX_STEPS = {1: 10, 2: 60}


class GuardedMazeEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mode: int = 1,
        rows: int | None = None,
        cols: int | None = None,
        action_noise: float = 0.2,
        max_steps: int | None = None,
        guard_prob: float = 0.05,
        guard_cost: float = 4,
        rand_guard: bool = False,
        rand_cost: bool = False,
        seed: int | None = None,
        detailed_r: bool = False,
        force_motion: float = 0,
        collect: bool = False,
        fixed_reset: bool = False,
        init_state: tuple[int, int] | None = None,
        goal_state: tuple[int, int] | None = None,
        continuous: bool = True,
    ):
        self.mode = mode
        if rows is None:
            rows = MAZE_SIZE[self.mode]
        if cols is None:
            cols = rows
        if cols != rows:
            raise ValueError("Only square boards are currently supported.")
        if goal_state is None:
            goal_state = (-2, rows // 3)

        self.rng = np.random.RandomState(seed)
        self.continuous = continuous

        self.guard_prob = guard_prob
        self.guard_cost = guard_cost
        self.rand_guard = rand_guard
        self.rand_cost = rand_cost
        self.detailed_r = detailed_r
        self.rows, self.cols = rows, cols
        self.L = 2 * self.rows
        if max_steps is None:
            max_steps = MAX_STEPS[self.mode] * self.L
        self.max_steps = max_steps
        self.force_motion = force_motion

        self.action_space = spaces.Discrete(4)
        if self.continuous:
            self.observation_space = spaces.Box(
                low=np.float32(0),
                high=np.float32(self.rows),
                shape=(2,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.float32(0),
                high=np.float32(1),
                shape=(3, self.rows, self.cols),
                dtype=np.float32,
            )

        self.directions = [
            np.array((-1, 0)),
            np.array((1, 0)),
            np.array((0, -1)),
            np.array((0, 1)),
        ]
        self.action_noise = action_noise

        self.map = self._randomize_walls()
        self.goal_cell, self.goal = self._random_from_map(
            goal_state, 0.5 * (self.mode - 1))
        self.goal_cell = np.round(self.goal_cell).astype(int) % self.rows

        self.state_xy, self.state = self._random_from_map(init_state)
        self.state_cell = np.round(self.state_xy).astype(int)

        self.fixed_reset = fixed_reset
        if fixed_reset:
            self.reset_state_cell = self.state_cell.copy()
            self.reset_state = self.state.copy()
        else:
            self.reset_state_cell, self.reset_state = None, None

        self.curr_guard: bool | float | None = None
        self.curr_cost: float | None = None
        self.spotted = False
        self.nsteps = 0
        self.is_long_path = 0
        self.tot_reward = 0.0
        self.episode_count = 0
        self._nep: list[int] = []
        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._terminals: list[bool] = []
        self.collect = collect

    def seed(self, seed: int | None = None):
        self.np_random, seed = seeding.np_random(seed)
        self.rng = np.random.RandomState(seed)
        return [seed]

    def one_hot_encoding(self):
        m = np.zeros((self.rows, self.rows), dtype=float)
        x, y = self.state_xy
        i0, j0 = self.state_xy.astype(int)
        i1, j1 = i0 + 1, j0 + 1
        dx, dy = i1 - x, j1 - y
        m[i0, j0] = dx * dy
        m[i0, j1] = dx * (1 - dy)
        m[i1, j0] = (1 - dx) * dy
        m[i1, j1] = (1 - dx) * (1 - dy)
        return m.reshape(-1)

    def get_obs(self):
        if self.continuous:
            return self.state_xy.astype(np.float32)
        return self._im_from_state().astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        options = options or {}
        guard = options.get("guard", None)
        guard_cost = options.get("guard_cost", None)
        init_state = options.get("init_state", None)

        if guard is None:
            guard = self.rng.random() < self.guard_prob
        if guard_cost is None:
            guard_cost = self.rng.exponential(
                self.guard_cost) if self.rand_cost else self.guard_cost
        self.curr_guard = guard
        self.curr_cost = guard_cost

        if self.fixed_reset:
            self.state_cell = self.reset_state_cell
            self.state = self.reset_state
            self.state_xy = self.state_cell.copy().astype(float)
        else:
            self.state_xy, self.state = self._random_from_map(init_state)
            self.state_cell = np.round(self.state_xy).astype(int)

        self.state_traj = [np.reshape(self.state_xy, (1, self.state_xy.size))]
        self.nsteps = 0
        self.tot_reward = 0.0
        self.is_long_path = 0
        self.spotted = False

        if self.collect:
            self.episode_count += 1
            self._nep.append(self.episode_count)
            self._rewards.append(self.get_reward(False))
            self._terminals.append(False)
            self._states.append(np.reshape(
                self.state_cell, (1, self.state_cell.size)))

        obs = self.get_obs()
        info = {"guard": self.curr_guard, "guard_cost": self.curr_cost}
        return obs, info

    def collect_on(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._terminals = []
        self._nep = []
        self.episode_count = 0
        self.collect = True

    def collect_off(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._terminals = []
        self._nep = []
        self.episode_count = 0
        self.collect = False

    def get_collected(self):
        import pandas as pd

        return pd.DataFrame(
            np.concatenate(
                (
                    np.array(self._nep)[:, np.newaxis],
                    np.concatenate(self._states, axis=0),
                    np.array(self._actions)[:, np.newaxis],
                    np.array(self._rewards)[:, np.newaxis],
                    np.array(self._terminals)[:, np.newaxis],
                ),
                axis=1,
            ),
            columns=("episode", "x", "y", "a", "r", "terminal"),
        )

    def get_reward(self, done=False, moved=True, mid_cell=None):
        if self.detailed_r:
            if done:
                r = 0
            else:
                r = -np.linalg.norm(np.abs(self.state_cell - self.goal_cell), ord=1) / (
                    self.rows + self.cols
                )
        else:
            r = 1 * self.L if done else (-1 if self.nsteps < 2 * self.L else 0)

        if self.force_motion and not moved:
            r -= self.force_motion

        if not self.spotted:
            kz = max(self.is_kill_zone(), self.is_kill_zone(mid_cell))
            if kz == 1:
                self.is_long_path = -1
            if kz:
                if (self.rand_guard and self.rng.random() < self.curr_guard) or (
                    (not self.rand_guard) and self.curr_guard
                ):
                    r -= self.curr_cost * kz * self.L
                    self.spotted = True

        return float(r)

    def step(self, action: int):
        next_xy = self.state_xy + self.directions[action]
        if self.action_noise:
            next_xy = next_xy + self.rng.normal(0, self.action_noise, size=2)
        next_xy = np.clip(next_xy, 0, self.rows - 1)
        next_cell = np.round(next_xy).astype(int)
        mid_xy = 0.5 * (self.state_xy + next_xy)
        mid_cell = np.round(mid_xy).astype(int)

        moved = False
        if self.map[next_cell[0], next_cell[1]] != 1 and self.map[mid_cell[0], mid_cell[1]] != 1:
            moved = True
            self.state_xy = next_xy
            self.state_cell = next_cell
            self.state = np.zeros_like(self.map)
            self.state[self.state_cell[0], self.state_cell[1]] = 1

        self.state_traj.append(np.reshape(
            self.state_xy, (1, self.state_xy.size)))
        terminated = bool(self.goal[self.state_cell[0], self.state_cell[1]])
        if terminated and self.is_long_path >= 0:
            self.is_long_path = 1

        obs = self.get_obs()
        reward = self.get_reward(terminated, moved, mid_cell)

        truncated = self.nsteps >= self.max_steps
        done = terminated or truncated

        if self.collect:
            self._nep.append(self.episode_count)
            self._states.append(np.reshape(
                self.state_xy, (1, self.state_xy.size)))
            self._actions.append(action)
            self._rewards.append(reward)
            self._terminals.append(done)
            if done:
                self._actions.append(-1)

        self.tot_reward += reward
        self.nsteps += 1

        info = {"path": self.is_long_path}
        if done:
            info = {"r": self.tot_reward, "l": self.nsteps,
                    "path": self.is_long_path}

        return obs, reward, terminated, truncated, info

    def get_local_map(self, rad=1):
        pad = max(0, rad - 1)
        n = self.rows + 2 * pad
        padded_map = np.ones((n, n))
        padded_map[pad: n - pad, pad: n - pad] = self.map
        x = pad + self.state_cell[0]
        y = pad + self.state_cell[1]
        return padded_map[x - rad: x + rad + 1, y - rad: y + rad + 1]

    def _random_from_map(self, xy=None, radius=0.0):
        if xy is None:
            cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
            while self.map[cell[0], cell[1]] != 0:
                cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
            xy = np.array(cell).copy()
        else:
            xy = np.array(xy)
            xy = xy % self.rows
            cell = np.round(xy).astype(int)

        goal_map = np.zeros_like(self.map)
        goal_map[
            int(cell[0] - radius): int(cell[0] + radius + 1),
            int(cell[1] - radius): int(cell[1] + radius + 1),
        ] = 1

        return xy, goal_map

    def _im_from_state(self, for_plot=False):
        maze_map = self.map
        if for_plot:
            maze_map = np.array([[0.0, 0.0, 1.0][int(x)] for x in maze_map.reshape(-1)]).reshape(
                maze_map.shape
            )

        im_list = [maze_map, self.goal, self.state]
        image = np.stack(im_list, axis=0)

        if for_plot:
            walls = np.array([[0, 0.7, 0][int(x)] for x in self.map.reshape(-1)]).reshape(
                self.map.shape
            )
            walls = np.stack(3 * [walls], axis=0)
            image += walls

        return image

    def _show_state(self, ax=None, color_scale=0.7, show_traj=True, traj_col="w"):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1)
        image = self._im_from_state(for_plot=True).swapaxes(0, 2)
        image[self.state_cell[1], self.state_cell[0], 2] = 0
        ax.imshow(color_scale * image)
        ax.invert_yaxis()
        if show_traj:
            track = np.concatenate(self.state_traj)
            ax.plot(track[:, 0], track[:, 1], f"{traj_col}.-")
            ax.plot(track[:1, 0], track[:1, 1], f"{traj_col}>", markersize=12)
            ax.plot(track[-1:, 0], track[-1:, 1],
                    f"{traj_col}s", markersize=10)
        if self.rows == 16:
            ticks = [0, 5, 10, 15]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        return ax

    def is_kill_zone(self, cell=None):
        if cell is None:
            cell = self.state_cell
        k = self.map[cell[0], cell[1]]
        if k < 0:
            return -k
        return 0

    def _randomize_walls(self):
        W = self.cols
        H = self.rows
        maze_map = np.zeros((H, W))

        maze_map[0, :] = 1
        maze_map[:, 0] = 1
        maze_map[-1:, :] = 1
        maze_map[:, -1:] = 1

        if self.mode == 1:
            maze_map[W // 4 + 1: 3 * W // 4, 3 * H // 4 - 1: 3 * H // 4] = 1
            maze_map[3 * W // 4 - 1: 3 * W // 4, H // 4: 3 * H // 4] = 1
        elif self.mode == 2:
            maze_map[: W // 2 - 1, H // 2 - 1: H // 2 + 1] = 1
            maze_map[W // 4 + 1: 3 * W // 4 + 1,
                     3 * H // 4 - 1: 3 * H // 4 + 1] = 1
            maze_map[3 * W // 4 - 1: 3 * W // 4 + 1, H // 4: 3 * H // 4] = 1

        if self.guard_prob:
            if self.mode == 1:
                maze_map[3 * W // 4 - 1: 3 * W // 4, 1: H // 4] = -1
            elif self.mode == 2:
                maze_map[3 * W // 4 - 1: 3 * W // 4 + 1, 1: H // 4] = -1

        return maze_map

    def render(self):
        return 0


class GuardedMaze(GuardedMazeEnv):
    pass


def evaluate_strategies(
    n=8,
    max_cost=2,
    goal_val=1,
    guard_prob=0.05,
    guard_cost=4,
    short_dist=1,
):
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(2, 2)
    axs = axs.ravel()

    kp = np.arange(0, 1.01, 0.01)
    p = np.arange(0, 1, 0.01)
    kc = -guard_cost * np.log(1 - p)

    L = 2 * n
    max_cost *= L
    goal_val *= L

    for a, k in enumerate((kp, kc)):
        stay = max_cost + 0 * k
        late_reach = max_cost - goal_val + 0 * k
        long = (4 - short_dist) * (n - 2) * 1 - goal_val + 0 * k

        if a == 0:
            greed = short_dist * (n - 2) * 1 - goal_val + L * guard_cost * k
        else:
            greed = short_dist * (n - 2) * 1 - goal_val + L * k * guard_prob

        axs[a].plot(k, stay, label="stay")
        axs[a].plot(k, late_reach, label="late long")
        axs[a].plot(k, long, label="long")
        axs[a].plot(k, greed, label="greedy")
        axs[a].legend()

    axs[0].set_xlabel("guard prob")
    axs[0].set_ylabel(f"E[loss | cost={guard_cost}]")
    axs[1].set_xlabel("guard cost")
    axs[1].set_ylabel(f"E[loss | prob={guard_prob}]")
    plt.tight_layout()

    return axs
