import time
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

robot = SO101Follower(SO101FollowerConfig(port="COM20", id="my_follower"))
robot.connect()

obs = robot.get_observation()
start = {k: v for k, v in obs.items() if k.endswith(".pos")}
print(start)

target = dict(start)
target["wrist_flex.pos"] += 10   # small nudge
robot.send_action(target)
time.sleep(1.5)
robot.send_action(start)         # back to where it was
time.sleep(1.5)

robot.disconnect()
