import numpy as np
import pytest

from so101_rl.config import EnvConfig, PushConfig
from so101_rl.push_env import BLOCK_HALF, PUSH_OBS_DIM, PUSH_OBS_LAYOUT, SO101PushEnv
from so101_rl.tasks import config_from_dict, config_to_dict, env_class_for


@pytest.fixture(scope="module")
def env():
    e = SO101PushEnv()
    yield e
    e.close()


def test_spaces_and_shapes(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (PUSH_OBS_DIM,) == env.observation_space.shape
    assert env.action_space.shape == (5,)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (PUSH_OBS_DIM,) and np.isfinite(obs).all() and np.isfinite(r)
    assert {"distance", "is_success", "ee_to_block", "ee_to_push_point"} <= info.keys()


def test_observation_contents(env):
    obs, _ = env.reset(seed=3)
    L = PUSH_OBS_LAYOUT
    np.testing.assert_allclose(obs[L["block_pos"]], env.block_pos(), atol=1e-6)
    np.testing.assert_allclose(obs[L["ee_pos"]], env.ee_pos(), atol=1e-6)
    np.testing.assert_allclose(obs[L["goal_xy"]], env.goal_xy, atol=1e-6)
    np.testing.assert_allclose(obs[L["qvel"]], 0.0)


def test_block_rests_on_table_and_arm_starts_clear(env):
    for seed in range(20):
        env.reset(seed=seed)
        assert not env._arm_touches_block()
        b0 = env.block_pos()
        assert abs(b0[2] - BLOCK_HALF) < 1e-3
        for _ in range(10):
            env.step(np.zeros(5))
        assert np.linalg.norm(env.block_pos() - b0) < 1e-3


def test_start_and_goal_ranges(env):
    c = env.cfg
    for seed in range(50):
        env.reset(seed=seed)
        b, g = env.block_pos()[:2], env.goal_xy
        assert c.block_radius[0] - 1e-9 <= np.hypot(*b) <= c.block_radius[1] + 1e-9
        assert c.goal_radius[0] <= np.hypot(*g) <= c.goal_radius[1]
        assert abs(np.degrees(np.arctan2(g[1], g[0]))) <= c.goal_max_bearing_deg
        assert c.goal_offset[0] - 1e-6 <= np.linalg.norm(g - b) <= c.goal_offset[1] + 1e-6


def test_reward_prefers_block_near_goal_and_ee_near_block(env):
    a = np.zeros(5)
    r = env.compute_reward
    assert r(0.05, 0.01, 0.01, a) > r(0.05, 0.10, 0.10, a)  # block nearer the goal
    assert r(0.02, 0.10, 0.10, a) > r(0.20, 0.10, 0.10, a)  # tip nearer the push point
    assert r(0.05, 0.09, 0.10, a) > r(0.05, 0.11, 0.10, a)  # progress beats regress
    assert r(0.05, 0.02, 0.02, a) - r(0.05, 0.0301, 0.0301, a) > env.cfg.success_bonus * 0.9


def test_push_point_is_behind_block(env):
    env.reset(seed=4)
    p, b, g = env.push_point(), env.block_pos(), env.goal_xy
    assert np.linalg.norm(p[:2] - g) > np.linalg.norm(b[:2] - g)
    np.testing.assert_allclose(np.linalg.norm(p[:2] - b[:2]), env.cfg.push_point_offset, atol=1e-9)


def test_deterministic_seeding():
    e1, e2 = SO101PushEnv(), SO101PushEnv()
    o1, _ = e1.reset(seed=7)
    o2, _ = e2.reset(seed=7)
    np.testing.assert_array_equal(o1, o2)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(-1, 1, 5)
        o1, r1, *_ = e1.step(a)
        o2, r2, *_ = e2.step(a)
    np.testing.assert_array_equal(o1, o2)
    assert r1 == r2


def test_truncates_at_episode_length():
    e = SO101PushEnv(PushConfig(episode_steps=5))
    e.reset(seed=0)
    for _ in range(5):
        *_, trunc, _ = e.step(np.zeros(5))
    assert trunc


def test_task_registry_round_trip():
    push = config_from_dict(config_to_dict(PushConfig()))
    assert isinstance(push, PushConfig) and env_class_for(push) is SO101PushEnv
    reach = config_from_dict(config_to_dict(EnvConfig()))
    assert isinstance(reach, EnvConfig)
    legacy = EnvConfig().to_dict()  # reach runs saved before "task" existed
    assert isinstance(config_from_dict(legacy), EnvConfig)
    # run_real.py loads configs with EnvConfig.from_dict, so it refuses push models.
    with pytest.raises(ValueError):
        EnvConfig.from_dict(config_to_dict(PushConfig()))


def test_render():
    e = SO101PushEnv(render_mode="rgb_array")
    e.reset(seed=0)
    img = e.render()
    assert img.shape == (480, 640, 3) and img.std() > 1
    e.close()


def test_goal_offset_can_be_narrowed():
    e = SO101PushEnv()
    e.set_goal_offset(0.04, 0.05)
    for seed in range(20):
        e.reset(seed=seed)
        d = np.linalg.norm(e.goal_xy - e.block_pos()[:2])
        assert 0.04 - 1e-6 <= d <= 0.05 + 1e-6


def test_curriculum_stage_ranges():
    from so101_rl.curriculum import stage_range

    start, end = (0.04, 0.07), (0.06, 0.15)
    assert stage_range(0, 5, start, end) == start
    np.testing.assert_allclose(stage_range(4, 5, start, end), end)
    np.testing.assert_allclose(stage_range(2, 5, start, end), (0.05, 0.11))
    np.testing.assert_allclose(stage_range(9, 5, start, end), end)


def test_v1_reward_is_the_first_design():
    v1 = SO101PushEnv(PushConfig.v1())
    v2 = SO101PushEnv()
    a = np.zeros(5)
    # v1 has no progress term: same distances, different previous distance -> same reward
    assert v1.compute_reward(0.05, 0.10, 0.12, a) == v1.compute_reward(0.05, 0.10, 0.10, a)
    assert v2.compute_reward(0.05, 0.10, 0.12, a) > v2.compute_reward(0.05, 0.10, 0.10, a)
    # v1 linear term is 3 per metre
    np.testing.assert_allclose(
        v1.compute_reward(0.05, 0.10, 0.10, a) - v1.compute_reward(0.05, 0.11, 0.11, a),
        3.0 * 0.01 + (np.tanh(0.11 / 0.05) - np.tanh(0.10 / 0.05)),
    )


def test_saved_configs_get_the_right_reward_version():
    old_v1 = PushConfig.v1().to_dict()
    for k in ("reward_version", "progress_weight", "push_point_offset"):
        del old_v1[k]  # the first pilot's config had none of these
    assert PushConfig.from_dict(old_v1).reward_version == "v1"
    old_v2 = PushConfig().to_dict()
    del old_v2["reward_version"]  # overnight runs saved before reward versions existed
    assert PushConfig.from_dict(old_v2).reward_version == "v2"
