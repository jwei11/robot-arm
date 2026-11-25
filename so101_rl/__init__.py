"""SO-101 reach task: MuJoCo Gymnasium env, PPO training, and sim-to-real mapping."""

from gymnasium.envs.registration import register

from so101_rl.config import DomainRandomization, EnvConfig
from so101_rl.env import SO101ReachEnv

register(id="SO101Reach-v0", entry_point="so101_rl.env:SO101ReachEnv")
register(id="SO101Push-v0", entry_point="so101_rl.push_env:SO101PushEnv")

__all__ = ["DomainRandomization", "EnvConfig", "SO101ReachEnv"]
