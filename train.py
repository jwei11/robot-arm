"""Train PPO on the SO-101 reach task.

Example (PowerShell):
  .\\.venv\\Scripts\\python.exe train.py --timesteps 2000000 --run-name ppo_reach
  .\\.venv\\Scripts\\tensorboard.exe --logdir runs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from so101_rl.config import DomainRandomization, EnvConfig, PushConfig
from so101_rl.tasks import config_to_dict, env_class_for


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["reach", "push"], default="reach")
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="ppo_reach")
    p.add_argument("--logdir", type=Path, default=Path("runs"))
    p.add_argument("--dr", action="store_true", help="enable the sim2real domain-randomization preset")
    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--net", type=int, nargs="+", default=[256, 256])
    p.add_argument("--log-std-init", type=float, default=-1.0, help="initial log std of the Gaussian policy")
    p.add_argument("--precision-weight", type=float, default=None, help="override EnvConfig.precision_weight")
    p.add_argument(
        "--max-bearing",
        type=float,
        default=None,
        help="override EnvConfig.target_max_bearing_deg (deg from straight ahead; -1 = all around)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=4)
    # bookkeeping
    p.add_argument("--eval-freq", type=int, default=50_000, help="in total env steps")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--checkpoint-freq", type=int, default=250_000, help="in total env steps")
    p.add_argument("--dummy-vec", action="store_true", help="single-process vec env (debugging)")
    return p.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    run_dir = args.logdir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "push":
        if args.dr or args.precision_weight is not None or args.max_bearing is not None:
            raise SystemExit("--dr, --precision-weight and --max-bearing apply to the reach task only")
        cfg = PushConfig()
    else:
        cfg = EnvConfig(dr=DomainRandomization.sim2real() if args.dr else DomainRandomization())
    if args.precision_weight is not None:
        cfg.precision_weight = args.precision_weight
    if args.max_bearing is not None:
        cfg.target_max_bearing_deg = None if args.max_bearing < 0 else args.max_bearing
    (run_dir / "env_config.json").write_text(json.dumps(config_to_dict(cfg), indent=2))
    (run_dir / "train_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    vec_cls = DummyVecEnv if args.dummy_vec or args.n_envs == 1 else SubprocVecEnv
    env_cls = env_class_for(cfg)
    env = make_vec_env(env_cls, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls, env_kwargs={"config": cfg})
    eval_env = make_vec_env(env_cls, n_envs=1, seed=args.seed + 10_000, env_kwargs={"config": cfg})

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": {"pi": args.net, "vf": args.net}, "log_std_init": args.log_std_init},
        tensorboard_log=str(args.logdir),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    per_env = lambda n: max(1, n // args.n_envs)  # noqa: E731 - callbacks count per-env calls
    callbacks = CallbackList(
        [
            CheckpointCallback(per_env(args.checkpoint_freq), str(run_dir / "checkpoints"), name_prefix="ppo"),
            EvalCallback(
                eval_env,
                n_eval_episodes=args.eval_episodes,
                eval_freq=per_env(args.eval_freq),
                best_model_save_path=str(run_dir),
                log_path=str(run_dir / "eval"),
                deterministic=True,
            ),
        ]
    )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=args.run_name, progress_bar=False)
    model.save(run_dir / "final_model")
    dt = time.time() - t0
    print(f"Done: {args.timesteps} steps in {dt / 60:.1f} min ({args.timesteps / dt:.0f} steps/s). Saved to {run_dir}")
    env.close()
    eval_env.close()
    return run_dir


if __name__ == "__main__":
    main()
