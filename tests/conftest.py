from pathlib import Path

import pytest

from so101_rl.mapping import load_calibration

DATA = Path(__file__).parent / "data"


@pytest.fixture
def calibration():
    """Copy of the real 'my_follower' calibration file (LeRobot 0.4.4 format)."""
    return load_calibration(DATA / "my_follower.json")
