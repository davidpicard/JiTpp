#!/usr/bin/env python3
"""Evaluate all step checkpoints in an output directory and log FID/IS to wandb.

Recovers the existing wandb run from the output directory so metrics
appear on the same run as the training curves.  If multiple training
runs exist (e.g. after several resumes), --run-id lets you pick
explicitly; otherwise the most recent run (latest-run symlink) is used.

When multiple GPUs are available, checkpoints are distributed across them
(one GPU per checkpoint, round-robin).  Each GPU runs the full single-GPU
evaluation pipeline independently — no cross-GPU communication.
Results are collected by the main process and logged to wandb in step order.

Usage:
    python eval_all.py configs/jit_b_pom_4gpu.yaml
    python eval_all.py configs/jit_b_pom_4gpu.yaml --no-ema
    python eval_all.py configs/jit_b_pom_4gpu.yaml --run-id abcdef12
    python eval_all.py configs/jit_b_pom_4gpu.yaml --override sampling.cfg=2.5
"""
import argparse
import os
import re
import sys

import torch
import torch.multiprocessing as mp
import wandb
from omegaconf import OmegaConf

from evaluate import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_run_id(output_dir: str) -> str | None:
    """Read the wandb run ID from the latest-run symlink in output_dir/wandb/.

    WandbLogger creates <output_dir>/wandb/latest-run -> run-DATE-<run_id>.
    The run_id is the alphanumeric suffix after the last '-'.
    """
    link = os.path.join(output_dir, "wandb", "latest-run")
    if os.path.islink(link):
        target = os.readlink(link)          # e.g. "run-20260327_120000-abcdef12"
        return target.rsplit("-", 1)[-1]
    return None


def step_from_ckpt(path: str) -> int | None:
    """Return the training step encoded in a filename like step-step=00050000.ckpt."""
    m = re.search(r"step-step=(\d+)\.ckpt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def find_checkpoints(output_dir: str) -> list[tuple[int, str]]:
    """Return (step, path) pairs for all step-*.ckpt files, sorted by step."""
    ckpts = []
    for fname in os.listdir(output_dir):
        if not fname.endswith(".ckpt") or fname == "last.ckpt":
            continue
        full = os.path.join(output_dir, fname)
        if os.path.islink(full):
            continue
        step = step_from_ckpt(fname)
        if step is not None:
            ckpts.append((step, full))
    ckpts.sort()
    return ckpts


# ---------------------------------------------------------------------------
# Multi-GPU checkpoint worker
# ---------------------------------------------------------------------------

def _eval_worker(
    rank: int,
    world_size: int,
    cfg,
    ckpt_groups: list,      # ckpt_groups[rank] = [(step, path), ...]
    use_ema: bool,
    result_dict,            # Manager().dict(), keyed by step
) -> None:
    """Spawned worker: restrict to cuda:rank and evaluate assigned checkpoints."""
    # Restrict CUDA visibility to this rank's GPU before any CUDA call.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    import logging
    logging.disable(logging.WARNING)

    for step, ckpt_path in ckpt_groups[rank]:
        print(f"\n[GPU {rank}] {'=' * 50}")
        print(f"[GPU {rank}] Step {step}  —  {ckpt_path}")
        metrics = evaluate(cfg, ckpt_path, use_ema=use_ema)
        if metrics:
            result_dict[step] = metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser("JiT batch FID evaluation")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--no-ema", action="store_true",
                        help="Use raw model weights instead of EMA")
    parser.add_argument("--run-id", default=None,
                        help="Wandb run ID to resume (auto-detected if omitted)")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Dot-list config overrides, e.g. sampling.cfg=2.5")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    output_dir = cfg.logging.output_dir

    # --- discover checkpoints -----------------------------------------------
    ckpts = find_checkpoints(output_dir)
    if not ckpts:
        sys.exit(f"No step-*.ckpt checkpoints found in {output_dir}")

    print(f"Found {len(ckpts)} checkpoint(s):")
    for step, path in ckpts:
        print(f"  step {step:>8d}  {path}")
    print()

    # --- connect to wandb run -----------------------------------------------
    run_id = args.run_id or find_run_id(output_dir)
    mode = "offline" if cfg.logging.wandb_offline else "online"

    if run_id:
        print(f"Resuming wandb run {run_id!r} (mode={mode})")
        run = wandb.init(
            project=cfg.logging.wandb_project,
            id=run_id,
            resume="must",
            dir=output_dir,
            mode=mode,
        )
    else:
        print("No existing wandb run found — creating a new one.")
        run = wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_experiment,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=output_dir,
            mode=mode,
        )

    metric_prefix = "raw" if args.no_ema else "ema"
    use_ema = not args.no_ema

    # --- evaluate: multi-GPU or single-GPU ----------------------------------
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs — distributing {len(ckpts)} checkpoints round-robin.\n")

        # Round-robin assignment: GPU r handles ckpts[r], ckpts[r+N], ckpts[r+2N], ...
        ckpt_groups = [ckpts[r::num_gpus] for r in range(num_gpus)]
        for r, group in enumerate(ckpt_groups):
            steps = [s for s, _ in group]
            print(f"  GPU {r}: steps {steps}")
        print()

        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        result_dict = manager.dict()

        mp.spawn(
            _eval_worker,
            args=(num_gpus, cfg, ckpt_groups, use_ema, result_dict),
            nprocs=num_gpus,
            join=True,
        )

        # Log results in step order
        for step, ckpt_path in ckpts:
            metrics = result_dict.get(step)
            if metrics:
                log_dict = {f"eval/{metric_prefix}/{k}": v for k, v in metrics.items()}
                run.log(log_dict, step=step)
                print(f"Logged {log_dict} at step {step}")
            else:
                print(f"Step {step}: no metrics (FID stats unavailable).")

    else:
        # Single GPU or CPU: evaluate sequentially
        for step, ckpt_path in ckpts:
            print(f"\n{'=' * 60}")
            print(f"Step {step}  —  {ckpt_path}")
            metrics = evaluate(cfg, ckpt_path, use_ema=use_ema)
            if metrics:
                log_dict = {f"eval/{metric_prefix}/{k}": v for k, v in metrics.items()}
                run.log(log_dict, step=step)
                print(f"Logged {log_dict} at step {step}")
            else:
                print("No metrics (FID stats unavailable for this resolution).")

    run_id_final = run.id
    run.finish()
    print("\nDone.")

    # In offline mode wandb writes a *new* offline-run-TIMESTAMP-{id} directory
    # (separate from the training run's directory).  Print the sync command so
    # the user knows exactly which path to upload.
    if mode == "offline":
        wandb_dir = os.path.join(output_dir, "wandb")
        matches = sorted(
            e for e in os.listdir(wandb_dir)
            if e.startswith("offline-run-") and e.endswith(f"-{run_id_final}")
        )
        if matches:
            sync_path = os.path.join(wandb_dir, matches[-1])
            print(f"\nSync eval metrics with:\n  wandb sync {sync_path}")


if __name__ == "__main__":
    main()
