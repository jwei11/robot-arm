"""Read-only pose check: torque is switched OFF and never switched on.

Connects to the motor bus only (skips lerobot's connect/configure, which re-enables
torque), reads the joints, and renders what the simulator thinks the pose is, so you
can compare the picture with the real arm. Nothing moves.

    .\\.venv\\Scripts\\python.exe pose_check.py            # writes pose_check.png
"""

import sys

import imageio
import mujoco
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from arm_config import JOINT_SIGNS, PORT, ROBOT_ID
from so101_rl.env import SO101ReachEnv
from so101_rl.kinematics import ARM_JOINTS, Kinematics
from so101_rl.mapping import SimRealMapping


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "pose_check.png"
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, use_degrees=True))
    bus = robot.bus
    bus.connect()                 # bus only: no configure(), so torque is never enabled
    try:
        bus.disable_torque()
        deg = bus.sync_read("Present_Position")      # normalized: degrees from range middle
    finally:
        bus.port_handler.closePort()
    obs = {f"{j}.pos": float(v) for j, v in deg.items()}
    q = SimRealMapping(robot.calibration, use_degrees=True, signs=JOINT_SIGNS).real_to_sim(obs)
    kin = Kinematics()
    tip = kin.fk(q)
    print("joint degrees:", {j: round(obs[f'{j}.pos'], 1) for j in ARM_JOINTS})
    print(f"simulated tip: x {tip[0]*100:.1f}  y {tip[1]*100:.1f}  z {tip[2]*100:.1f} cm")

    env = SO101ReachEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.data.qpos[env.qpos_adr] = q
    mujoco.mj_forward(env.model, env.data)
    imageio.imwrite(out, env.render())
    env.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
