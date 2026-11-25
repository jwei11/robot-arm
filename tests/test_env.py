import numpy as np
import pytest

from so101_rl.config import DomainRandomization, EnvConfig
from so101_rl.env import OBS_DIM, OBS_LAYOUT, SO101ReachEnv
from so101_rl.kinematics import ARM_JOINTS, Kinematics


@pytest.fixture
def env():
    e = SO101ReachEnv()
    yield e
    e.close()


def test_model_matches_official_joints(env):
    kin = Kinematics()
    assert ARM_JOINTS == ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    np.testing.assert_allclose(np.degrees(kin.limits[:3]), [[-110, 110], [-100, 100], [-96.83, 96.83]], atol=0.01)


def test_reset_and_step_shapes(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (5,)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (OBS_DIM,) and isinstance(r, float)
    assert term is False and trunc is False
    assert {"distance", "is_success"} <= set(info)


def test_episode_truncates_at_fixed_length(env):
    env.reset(seed=0)
    for t in range(env.cfg.episode_steps):
        *_, term, trunc, _ = env.step(np.zeros(5))
        assert not term
        assert trunc == (t == env.cfg.episode_steps - 1)


def test_observation_contents(env):
    obs, _ = env.reset(seed=3)
    for _ in range(5):
        obs, *_ = env.step(env.action_space.sample())
    q = env.data.qpos[env.qpos_adr]
    np.testing.assert_allclose(obs[OBS_LAYOUT["qpos"]], q, atol=1e-6)
    np.testing.assert_allclose(obs[OBS_LAYOUT["ee_pos"]], env.ee_pos(), atol=1e-6)  # no noise: FK == true
    np.testing.assert_allclose(obs[OBS_LAYOUT["target_pos"]], env.target_pos, atol=1e-6)
    np.testing.assert_allclose(obs[OBS_LAYOUT["ee_to_target"]], (env.target_pos - env.ee_pos()) * 5.0, atol=1e-5)
    np.testing.assert_allclose(obs[OBS_LAYOUT["prev_action"]], env.prev_action, atol=1e-6)


def test_action_is_delta_clipped_and_clamped(env):
    env.reset(seed=0)
    q0 = env.q_target.copy()
    env.step(np.full(5, 10.0))  # out-of-range action is clipped to +1
    np.testing.assert_allclose(env.q_target - q0, np.minimum(env.cfg.max_delta, env.q_high - q0), atol=1e-12)
    for _ in range(200):
        env.step(np.ones(5))
    assert np.all(env.q_target <= env.q_high + 1e-12)


def test_reward_higher_when_closer(env):
    a = np.zeros(5)
    assert env.compute_reward(0.05, a) > env.compute_reward(0.10, a)
    assert env.compute_reward(0.01, a) > env.compute_reward(0.03, a) + env.cfg.success_bonus * 0.5
    assert env.compute_reward(0.05, a) > env.compute_reward(0.05, np.ones(5))


def test_reward_closer_in_rollout(env):
    """Placing the target at the current tip gives more reward than a far target."""
    env.reset(seed=0)
    env.reset(seed=0, options={"target_pos": env.ee_pos()})
    _, r_near, *_ = env.step(np.zeros(5))
    env.reset(seed=0, options={"target_pos": env.ee_pos() + [0.0, 0.0, 0.15]})
    _, r_far, *_ = env.step(np.zeros(5))
    assert r_near > r_far


def test_targets_are_reachable(env):
    kin = Kinematics()
    rng = np.random.default_rng(0)
    for seed in range(30):
        env.reset(seed=seed)
        t = env.target_pos
        assert t[2] >= env.cfg.target_min_z
        assert np.hypot(t[0], t[1]) >= env.cfg.target_min_radius
        assert abs(np.degrees(np.arctan2(t[1], t[0]))) <= env.cfg.target_max_bearing_deg
        q = kin.solve_ik(t, env.q_low, env.q_high, rng=rng)
        assert q is not None, f"target {t} not reachable"


def test_seeding_is_deterministic():
    def rollout(seed):
        e = SO101ReachEnv(EnvConfig(dr=DomainRandomization.sim2real()))
        obs, _ = e.reset(seed=seed)
        e.action_space.seed(seed)
        out = [obs]
        for _ in range(20):
            obs, r, *_ = e.step(e.action_space.sample())
            out.append(np.append(obs, r))
        return np.concatenate(out)

    np.testing.assert_array_equal(rollout(7), rollout(7))
    assert not np.array_equal(rollout(7), rollout(8))


def test_domain_randomization_off_by_default(env):
    nominal = env.model.actuator_gainprm[:, 0].copy()
    for s in range(3):
        env.reset(seed=s)
        np.testing.assert_array_equal(env.model.actuator_gainprm[:, 0], nominal)
        assert env.latency == 0


def test_domain_randomization_changes_params():
    e = SO101ReachEnv(EnvConfig(dr=DomainRandomization.sim2real()))
    nominal_kp = e._nominal["gainprm"][:, 0]
    nominal_damp = e._nominal["damping"]
    e.reset(seed=1)
    kp = e.model.actuator_gainprm[:, 0]
    assert not np.allclose(kp, nominal_kp)
    np.testing.assert_allclose(e.model.actuator_biasprm[:, 1], -kp)  # stiffness stays consistent
    assert not np.allclose(e.model.dof_damping, nominal_damp)


def test_action_latency():
    cfg = EnvConfig(dr=DomainRandomization(enabled=True, action_latency_steps=(2, 2)))
    e = SO101ReachEnv(cfg)
    e.reset(seed=0)
    q0 = e.q_target.copy()
    e.step(np.ones(5))
    e.step(np.zeros(5))
    np.testing.assert_array_equal(e.q_target, q0)  # first action not applied yet
    e.step(np.zeros(5))
    assert np.all(e.q_target >= q0)
    assert np.any(e.q_target > q0)


def test_observation_noise():
    cfg = EnvConfig(dr=DomainRandomization(enabled=True, obs_noise_std=0.01))
    e = SO101ReachEnv(cfg)
    obs, _ = e.reset(seed=0)
    assert not np.allclose(obs[OBS_LAYOUT["qpos"]], e.data.qpos[e.qpos_adr], atol=1e-4)


def test_render_rgb_array():
    e = SO101ReachEnv(render_mode="rgb_array")
    e.reset(seed=0)
    img = e.render()
    assert img.shape == (480, 640, 3) and img.std() > 0
    e.close()


def test_old_config_without_bearing_samples_all_around():
    from so101_rl.config import EnvConfig

    d = EnvConfig().to_dict()
    del d["target_max_bearing_deg"]
    del d["target_sampling"]
    assert EnvConfig.from_dict(d).target_max_bearing_deg is None
    assert EnvConfig.from_dict(d).target_sampling == "joint"
    assert EnvConfig.from_dict(EnvConfig().to_dict()).target_max_bearing_deg == 90.0


def test_joint_space_sampling_still_works():
    from so101_rl.config import EnvConfig

    env = SO101ReachEnv(EnvConfig(target_sampling="joint"))
    env.reset(seed=0)
    assert env.target_pos[2] >= env.cfg.target_min_z
