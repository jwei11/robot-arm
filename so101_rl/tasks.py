"""Task registry: which config and env class belong to each task name."""

from __future__ import annotations

from so101_rl.config import EnvConfig, PushConfig
from so101_rl.env import SO101ReachEnv
from so101_rl.push_env import SO101PushEnv

TASKS = {"reach": (EnvConfig, SO101ReachEnv), "push": (PushConfig, SO101PushEnv)}


def config_to_dict(cfg) -> dict:
    task = "push" if isinstance(cfg, PushConfig) else "reach"
    return {**cfg.to_dict(), "task": task}


def config_from_dict(d: dict):
    """Rebuild a saved env_config.json. Files without a "task" key are reach configs."""
    task = d.get("task", "reach")
    return TASKS[task][0].from_dict(d)


def env_class_for(cfg):
    return SO101PushEnv if isinstance(cfg, PushConfig) else SO101ReachEnv
