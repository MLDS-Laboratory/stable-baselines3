import numpy as np
import pytest

from stable_baselines3 import COPG


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
@pytest.mark.parametrize("clip_range_vf", [None, 0.2, -0.2])
def test_copg(env_id, clip_range_vf):
    if clip_range_vf is not None and clip_range_vf < 0:
        with pytest.raises(AssertionError):
            COPG(
                "MlpPolicy",
                env_id,
                seed=0,
                policy_kwargs=dict(net_arch=[16]),
                clip_range_vf=clip_range_vf,
            )
    else:
        model = COPG(
            "MlpPolicy",
            env_id,
            n_steps=128,
            batch_size=64,
            n_epochs=1,
            seed=0,
            policy_kwargs=dict(net_arch=[16]),
            clip_range_vf=clip_range_vf,
        )
        model.learn(total_timesteps=256)
        loss = model.logger.name_to_value["train/loss"]
        assert not np.isnan(loss)
