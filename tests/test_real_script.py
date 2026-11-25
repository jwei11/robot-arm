"""run_real.py logic, without hardware: safety filter, safe targets, fake and dry-run arms."""

import numpy as np
import pytest

import run_real
from so101_rl.config import EnvConfig
from so101_rl.kinematics import ARM_JOINTS, Kinematics
from so101_rl.mapping import SimRealMapping
from so101_rl.safety import SafetyFilter, interpolate


def _filter(max_step=0.026):
    lim = np.array([[-1.0, 1.0]] * 5)
    return SafetyFilter(lim[:, 0], lim[:, 1], lim * 2, margin=0.1, max_step=max_step)


def test_safety_rate_limit_and_clamp():
    f = _filter()
    prev = np.zeros(5)
    out = f(np.array([0.5, -0.5, 0.001, 5.0, -5.0]), prev)
    assert np.all(np.abs(out - prev) <= f.max_step + 1e-12)
    assert out[2] == pytest.approx(0.001)
    q = prev
    for _ in range(1000):
        q = f(np.full(5, 10.0), q)
    np.testing.assert_allclose(q, f.high)  # converges to, never exceeds, the safe box


def test_safety_box_is_intersection():
    f = SafetyFilter([-1, -1, -1, -1, -1], [1, 1, 1, 1, 1], np.array([[-0.5, 2.0]] * 5), margin=0.1, max_step=0.1)
    np.testing.assert_allclose(f.low, -0.4)
    np.testing.assert_allclose(f.high, 1.0)
    with pytest.raises(ValueError):
        SafetyFilter([0] * 5, [1] * 5, np.array([[2.0, 3.0]] * 5), margin=0.0, max_step=0.1)


def test_interpolate_step_bound():
    path = interpolate(np.zeros(5), np.array([1.0, -0.3, 0, 0, 0]), 0.05)
    assert len(path) == 20
    np.testing.assert_allclose(path[-1], [1.0, -0.3, 0, 0, 0])
    steps = np.diff(np.vstack([np.zeros(5), *path]), axis=0)
    assert np.abs(steps).max() <= 0.05 + 1e-12


def test_safe_targets_reachable(calibration):
    cfg = EnvConfig()
    kin = Kinematics()
    mapping = SimRealMapping(calibration)
    f = SafetyFilter(
        kin.limits[:, 0] + cfg.limit_margin,
        kin.limits[:, 1] - cfg.limit_margin,
        mapping.calibrated_limits_rad(),
        margin=np.radians(run_real.CALIB_MARGIN_DEG),
        max_step=0.1,
    )
    for t in run_real.SAFE_TARGETS:
        assert t[2] >= 0.1, "safe targets stay well above the table"
        q = kin.solve_ik(np.array(t), f.low, f.high)
        assert q is not None, f"safe target {t} not reachable inside the safe box"


class ConstantPolicy:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        assert obs.shape == (24,)
        return self.a, None


def test_fake_arm_run_respects_limits(calibration):
    mapping = SimRealMapping(calibration)
    arm = run_real.FakeArm(mapping, start_q=np.array([0.0, -1.2, 1.2, 0.8, 0.0]))
    max_step = np.radians(1.5)
    r = run_real.Runner(arm, mapping, EnvConfig(control_hz=1000), max_step=max_step, dry_run=False, log=lambda s: None)
    r.move_to(np.asarray(EnvConfig().home_qpos), "home")
    r.run_policy(ConstantPolicy(np.ones(5)), np.array(run_real.SAFE_TARGETS[0]), steps=60)
    sent = np.array([mapping.real_to_sim(a) for a in arm.sent])
    assert len(sent) > 60
    # every command inside the safe box and within max_step of the previous one
    assert np.all(sent >= r.safety.low - 1e-9) and np.all(sent <= r.safety.high + 1e-9)
    assert np.abs(np.diff(sent, axis=0)).max() <= max_step + 1e-9
    assert all("gripper.pos" in a for a in arm.sent)


class NoSendArm:
    """Looks like a real arm but fails the test if anything is sent."""

    def __init__(self, mapping):
        self.mapping = mapping

    def read(self):
        return self.mapping.sim_to_real(np.zeros(5)) | {"gripper.pos": 10.0}

    def send(self, action):
        raise AssertionError("dry run must never send")


def test_dry_run_never_sends(calibration):
    mapping = SimRealMapping(calibration)
    logs = []
    r = run_real.Runner(NoSendArm(mapping), mapping, EnvConfig(control_hz=1000), np.radians(1.5), True, logs.append)
    r.move_to(np.asarray(EnvConfig().home_qpos), "home")
    r.run_policy(ConstantPolicy(np.zeros(5)), np.array(run_real.SAFE_TARGETS[1]), steps=5)
    assert any(s.startswith("DRY-RUN would send") for s in logs)


def test_cli_gates():
    assert run_real.main(["--model", "x.zip", "--max-step-deg", "10"]) == 2
    assert run_real.main(["--model", "x.zip", "--live", "--fake"]) == 2
    assert run_real.main([]) == 2
    assert run_real.parse_args(["--model", "x.zip"]).live is False  # dry run by default


def test_joint_order_matches_env():
    assert tuple(ARM_JOINTS) == ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
