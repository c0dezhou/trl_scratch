from __future__ import annotations

import argparse
import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _module_to_path(module: str) -> Path:
    rel = Path(*module.split(".")).with_suffix(".py")
    return rel


def _write_config(base_module: str, out_module: str, dataset_path: str) -> Path:
    out_path = _module_to_path(out_module)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""from __future__ import annotations

from {base_module} import Step5WorldModelConfig as BaseConfig


class Step5WorldModelConfig(BaseConfig):
    # auto-generated: override dataset path for this run
    dataset_path: str = {dataset_path!r}
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


def _get_run_name(config_module: str) -> str:
    mod = importlib.import_module(config_module)
    cfg = mod.Step5WorldModelConfig()
    return str(cfg.run_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["random", "mixed", "policy"], default="mixed")
    p.add_argument("--out", type=str, default="data/cartpole_pomdp_from_step3_mixed.npz")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_actions", action="store_true")
    p.add_argument("--random_action_prob", type=float, default=0.2)
    p.add_argument("--policy_config", type=str, default=None)
    p.add_argument("--policy_ckpt", type=str, default=None)

    p.add_argument("--base_config", type=str, default="configs.step5_world_model_cartpole_pomdp")
    p.add_argument("--config_out", type=str, default="configs.step5_world_model_cartpole_pomdp_auto")
    p.add_argument("--write_config", action="store_true")

    p.add_argument("--collect", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", dest="eval_offline", action="store_true")
    p.add_argument("--mpc", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--mpc_episodes", type=int, default=10)
    p.add_argument("--mpc_horizon", type=int, default=25)
    p.add_argument("--mpc_num_samples", type=int, default=1024)
    p.add_argument("--mpc_state_cost_coef", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in ("mixed", "policy"):
        if not args.policy_config or not args.policy_ckpt:
            raise ValueError("--policy_config and --policy_ckpt are required for mode=policy/mixed")

    # collect
    if args.collect:
        cmd = [
            sys.executable,
            "-m",
            "scripts.step4_dt.collect_cartpole_dataset",
            "--out",
            args.out,
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
        ]
        if args.mode == "random":
            cmd.append("--random_only")
        else:
            cmd += ["--policy_config", args.policy_config, "--policy_ckpt", args.policy_ckpt]
            if args.sample_actions:
                cmd.append("--sample_actions")
            if args.mode == "mixed":
                cmd += ["--random_action_prob", str(args.random_action_prob)]
        _run(cmd, dry_run=args.dry_run)

    # optionally write config override
    config_module = args.base_config
    if args.write_config:
        _write_config(args.base_config, args.config_out, args.out)
        config_module = args.config_out

    run_name = _get_run_name(config_module)
    default_ckpt = os.path.join("checkpoints", f"{run_name}_best.pt")

    if args.train:
        _run(
            [sys.executable, "-m", "scripts.step5_world_model.train_world_model", "--config", config_module],
            dry_run=args.dry_run,
        )

    if args.eval_offline:
        _run(
            [
                sys.executable,
                "-m",
                "scripts.step5_world_model.eval_world_model",
                "--ckpt",
                default_ckpt,
            ],
            dry_run=args.dry_run,
        )

    if args.mpc:
        cmd = [
            sys.executable,
            "-m",
            "scripts.step5_world_model.plan_mpc_cartpole",
            "--ckpt",
            default_ckpt,
            "--episodes",
            str(args.mpc_episodes),
            "--horizon",
            str(args.mpc_horizon),
            "--num_samples",
            str(args.mpc_num_samples),
        ]
        if args.mpc_state_cost_coef is not None:
            cmd += ["--state_cost_coef", str(args.mpc_state_cost_coef)]
        _run(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
