import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv


class RiskyMiniGridEnv(MiniGridEnv):

    @staticmethod
    def mission_func():
        return "avoid risky cells"

    def __init__(
        self,
        size=7,
        max_steps=50,
        is_risky=True,
        risk_level=1,
        **kwargs,
    ):
        self.is_risky = is_risky
        self.risk_level = risk_level
        self.total_violations = 0

        super().__init__(
            mission_space=MissionSpace(
                mission_func=RiskyMiniGridEnv.mission_func),
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Fixed agent and goal positions
        agent_pos = (1, height - 2)
        goal_pos = (width - 2, height - 2)

        # Place agent
        self.agent_pos = agent_pos
        self.agent_dir = 0  # facing right

        # Place goal
        self.put_obj(Goal(), *goal_pos)

        # Risky cells along the direct Manhattan path
        risky_positions = []

        # Horizontal segment
        for x in range(agent_pos[0] + 1, goal_pos[0]):
            risky_positions.append((x, agent_pos[1]))
        self.risky_positions = risky_positions

        # Place lava (risky states)
        for x, y in risky_positions:
            self.grid.set(x, y, Lava())

        self.mission = "reach the goal while crossing risky states"

    def reset(self, **kwargs):
        self.total_violations = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        violation = tuple(self.agent_pos) in self.risky_positions
        if violation:
            if self.risk_level == 1:
                reward += np.random.normal(0, 1) * int(self.is_risky)
            elif self.risk_level == 2:
                reward += -abs(10.0 * np.random.randn() * int(self.is_risky))
            elif self.risk_level == 3:
                reward += -0.1 * int(self.is_risky)
            terminated = False
        self.total_violations += int(violation)
        info['viol'] = self.total_violations / self.step_count

        return obs, reward, terminated, truncated, info
