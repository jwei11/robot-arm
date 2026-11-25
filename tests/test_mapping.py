import json
from pathlib import Path

import numpy as np
import pytest

from so101_rl.kinematics import ARM_JOINTS, Kinematics
from so101_rl.mapping import SimRealMapping, lerobot_normalize, lerobot_unnormalize

DATA = Path(__file__).parent / "data"


def _random_q(rng, n=200):
    lim = Kinematics().limits
    return rng.uniform(lim[:, 0], lim[:, 1], size=(n, 5))


@pytest.mark.parametrize("use_degrees", [True, False])
def test_sim_real_round_trip(calibration, use_degrees):
    m = SimRealMapping(calibration, use_degrees=use_degrees)
    for q in _random_q(np.random.default_rng(0)):
        np.testing.assert_allclose(m.real_to_sim(m.sim_to_real(q)), q, atol=1e-9)


@pytest.mark.parametrize("use_degrees", [True, False])
def test_real_sim_round_trip(calibration, use_degrees):
    m = SimRealMapping(calibration, use_degrees=use_degrees)
    rng = np.random.default_rng(1)
    for _ in range(100):
        vals = {f"{j}.pos": float(rng.uniform(-80, 80)) for j in ARM_JOINTS}
        back = m.sim_to_real(m.real_to_sim(vals))
        for k, v in vals.items():
            assert back[k] == pytest.approx(v, abs=1e-9)


def test_degrees_mode_is_identity_by_default(calibration):
    """LeRobot's own SO-101 kinematics feeds degrees straight into this model."""
    m = SimRealMapping(calibration)
    vals = {f"{j}.pos": d for j, d in zip(ARM_JOINTS, [10.0, -20.0, 30.0, -40.0, 50.0], strict=True)}
    np.testing.assert_allclose(m.real_to_sim(vals), np.radians([10, -20, 30, -40, 50]))


def test_signs_and_offsets(calibration):
    m = SimRealMapping(calibration, signs={"elbow_flex": -1}, offsets_deg={"wrist_flex": 5.0})
    vals = {f"{j}.pos": 0.0 for j in ARM_JOINTS} | {"elbow_flex.pos": 30.0, "wrist_flex.pos": 15.0}
    q = m.real_to_sim(vals)
    assert q[2] == pytest.approx(np.radians(-30))
    assert q[3] == pytest.approx(np.radians(10))
    with pytest.raises(ValueError):
        SimRealMapping(calibration, signs={"elbow_flex": 2})


def test_user_waypoints_map_inside_sim_limits(calibration):
    """Poses the real arm actually reached must land within the model's joint range (+3 deg)."""
    m = SimRealMapping(calibration)
    lim = Kinematics().limits
    for wp in json.loads((DATA / "waypoints.json").read_text()):
        q = m.real_to_sim(wp)
        assert np.all(q >= lim[:, 0] - np.radians(3)) and np.all(q <= lim[:, 1] + np.radians(3)), wp


def test_calibrated_limits(calibration):
    m = SimRealMapping(calibration)
    lim = np.degrees(m.calibrated_limits_rad())
    # shoulder_lift: ticks 850..3242 around mid 2046 -> about +/-105 deg
    assert lim[1] == pytest.approx([-105.1, 105.1], abs=0.1)
    assert np.all(lim[:, 0] < 0) and np.all(lim[:, 1] > 0)


# --------------------------------------------------- agreement with LeRobot 0.4.4 itself
lerobot = pytest.importorskip("lerobot", reason="LeRobot not installed")


def _bus(calibration, mode):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    nm = MotorNormMode.DEGREES if mode else MotorNormMode.RANGE_M100_100
    motors = {j: Motor(calibration[j].id, "sts3215", nm) for j in ARM_JOINTS}
    cal = {j: MotorCalibration(**vars(calibration[j])) for j in ARM_JOINTS}
    return FeetechMotorsBus(port="NOT_A_PORT", motors=motors, calibration=cal)


@pytest.mark.parametrize("use_degrees", [True, False])
def test_matches_lerobot_normalize(calibration, use_degrees):
    bus = _bus(calibration, use_degrees)
    rng = np.random.default_rng(2)
    for j in ARM_JOINTS:
        c = calibration[j]
        for raw in rng.integers(c.range_min, c.range_max + 1, size=50):
            ours = lerobot_normalize(int(raw), c, use_degrees)
            theirs = bus._normalize({c.id: int(raw)})[c.id]
            assert ours == pytest.approx(theirs, abs=1e-9)


@pytest.mark.parametrize("use_degrees", [True, False])
def test_matches_lerobot_unnormalize(calibration, use_degrees):
    bus = _bus(calibration, use_degrees)
    rng = np.random.default_rng(3)
    for j in ARM_JOINTS:
        c = calibration[j]
        for v in rng.uniform(-95, 95, size=50):
            ours = lerobot_unnormalize(float(v), c, use_degrees, truncate=True)
            theirs = bus._unnormalize({c.id: float(v)})[c.id]
            assert ours == theirs


def test_sim_command_survives_lerobot_quantization(calibration):
    """sim q -> LeRobot value -> LeRobot raw ticks (int) -> LeRobot value -> sim q: error < 1 tick."""
    bus = _bus(calibration, True)
    m = SimRealMapping(calibration)
    one_tick = 2 * np.pi / 4095
    for q in _random_q(np.random.default_rng(4), 50):
        action = m.sim_to_real(q)
        raw = bus._unnormalize({calibration[j].id: action[f"{j}.pos"] for j in ARM_JOINTS})
        read = bus._normalize(raw)
        obs = {f"{j}.pos": read[calibration[j].id] for j in ARM_JOINTS}
        assert np.max(np.abs(m.real_to_sim(obs) - q)) <= one_tick + 1e-12
