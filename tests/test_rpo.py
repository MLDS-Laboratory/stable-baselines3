import numpy as np
import pytest

from stable_baselines3 import RPO


@pytest.mark.parametrize("rpo_alpha", [0.0, 0.1])
def test_rpo_smoke(rpo_alpha):
    model = RPO(
        "MlpPolicy",
        "Pendulum-v1",
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        rpo_alpha=rpo_alpha,
    )
    model.learn(total_timesteps=128)
    loss = model.logger.name_to_value["train/loss"]
    assert not np.isnan(loss)


def test_rpo_invalid_alpha():
    with pytest.raises(AssertionError, match="`rpo_alpha` must be non-negative"):
        RPO("MlpPolicy", "Pendulum-v1", rpo_alpha=-0.1)
