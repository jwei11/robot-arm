"""Hardware facts about this particular arm, shared by the scripts that use the simulator model.

JOINT_SIGNS: joints that turn the opposite way from the official SO-101 MuJoCo model.
Verified 2026-11-24 with pose_check.py (torque off, arm posed by hand, compared with
the render): shoulder_pan, shoulder_lift, elbow_flex and wrist_flex all match the model,
so no corrections are needed. If you recalibrate or rebuild a joint, re-run
pose_check.py and add {"joint": -1} here for any joint that comes out mirrored.
"""

PORT = "COM20"
ROBOT_ID = "my_follower"
JOINT_SIGNS: dict[str, int] = {}
