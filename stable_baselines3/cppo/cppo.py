from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO

SelfCPPO = TypeVar("SelfCPPO", bound="CPPO")


class CPPO(PPO):
    """
    Return-capped PPO implementation.

    This follows the PPO optimization loop from SB3 and modifies rollout
    collection by capping cumulative episodic return with a moving cap.
    The cap is updated after each rollout from a lower-tail quantile of
    uncapped episodic returns.
    """

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = RolloutBuffer,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        cap_return: bool = True,
        cap_alpha: float = 0.2,
        cap_tau: float = 0.1,
        initial_return_cap: float = -0.1,
        minimum_return_cap: float = -0.1,
    ):
        if rollout_buffer_kwargs is None:
            rollout_buffer_kwargs = {}

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.cap_return = cap_return
        self.cap_alpha = cap_alpha
        self.cap_tau = cap_tau
        self.return_cap = initial_return_cap
        self.minimum_return_cap = minimum_return_cap

        self._uncapped_episode_returns = np.zeros(
            self.n_envs, dtype=np.float32)
        self._capped_episode_returns = np.zeros(self.n_envs, dtype=np.float32)
        self._last_batch_quantile_return: Optional[float] = None

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        completed_uncapped_returns: list[float] = []

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions_np = actions.cpu().numpy()
            values_np = values.clone().cpu().numpy().flatten()

            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(
                        clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions_np, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            uncapped_rewards = rewards.copy()

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            self._uncapped_episode_returns += uncapped_rewards

            if self.cap_return:
                rewards = np.minimum(
                    self._uncapped_episode_returns, self.return_cap) - self._capped_episode_returns
            self._capped_episode_returns += rewards

            if np.any(dones):
                done_idx = np.where(dones)[0]
                completed_uncapped_returns.extend(
                    self._uncapped_episode_returns[done_idx].tolist())
                self._uncapped_episode_returns[done_idx] = 0.0
                self._capped_episode_returns[done_idx] = 0.0

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            # SMDP support: extract action durations from info (default 1.0 for standard MDP)
            if len(infos) == env.num_envs:
                action_durations = np.array(
                    [info.get("action_duration", 1.0) for info in infos], dtype=np.float32
                )
            else:
                action_durations = np.ones(env.num_envs, dtype=np.float32)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            terminal_obs)[0]  # type: ignore[arg-type]
                    # SMDP: use gamma^tau for the bootstrap discount
                    rewards[idx] += (self.gamma **
                                     action_durations[idx]) * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions_np,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                action_duration=action_durations,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(
                new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones)

        if self.cap_return and len(completed_uncapped_returns) > 0:
            batch_quantile_return = float(np.quantile(
                completed_uncapped_returns, self.cap_alpha))

            self.return_cap += self.cap_tau * \
                (batch_quantile_return - self.return_cap)
            self.return_cap = max(self.return_cap, self.minimum_return_cap)
            self._last_batch_quantile_return = batch_quantile_return

            self.logger.record("train/return_cap", self.return_cap)
            self.logger.record("rollout/quantile_return",
                               batch_quantile_return)
            self.logger.record("rollout/completed_episodes",
                               len(completed_uncapped_returns))

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def learn(
        self: SelfCPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "CPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfCPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
