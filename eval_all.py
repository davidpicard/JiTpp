#!/usr/bin/env python3
"""Evaluate all step checkpoints in an output directory and log FID/IS to wandb.

Creates a fresh wandb run named <experiment>-eval-{ema|raw} so that sync
works reliably in offline mode.  Group it with the training run manually
in the wandb UI.

When multiple GPUs are available, checkpoints are distributed across them
(one GPU per checkpoint, round-robin).  Each GPU runs the full single-GPU
evaluation pipeline independently — no cross-GPU communication.
Results are logged to wandb as each checkpoint finishes (out-of-order fine).

Usage:
    python eval_all.py configs/jit_b_pom_4gpu.yaml
    python eval_all.py configs/jit_b_pom_4gpu.yaml --no-ema
    python eval_all.py configs/jit_b_pom_4gpu.yaml --override sampling.cfg=2.5
"""
import argparse
import os
import re
import sys
import threading

import torch
import torch.multiprocessing as mp
import wandb
from omegaconf import OmegaConf

from evaluate import evaluate


# ---------------------------------------------------------------------------
# Multi-GPU live display
# ---------------------------------------------------------------------------

class _QueueWriter:
    """Replaces sys.stdout in a worker process.

    Buffers text and emits one ("status", rank, line) message per logical
    line (split on \\n or \\r) to the shared display queue.  \\r lines
    (in-place progress updates from evaluate.py) are emitted without
    advancing the GPU's row, so the display thread keeps overwriting the
    same terminal line.
    """

    def __init__(self, rank: int, queue) -> None:
        self.rank = rank
        self.queue = queue
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while True:
            nl = self._buf.find("\n")
            cr = self._buf.find("\r")
            if nl == -1 and cr == -1:
                break
            # Whichever separator comes first wins
            if cr != -1 and (nl == -1 or cr < nl):
                line, self._buf = self._buf[:cr].strip(), self._buf[cr + 1:]
            else:
                line, self._buf = self._buf[:nl].strip(), self._buf[nl + 1:]
            if line:
                self.queue.put(("status", self.rank, line))
        return len(text)

    def flush(self) -> None:
        line = self._buf.strip()
        if line:
            self.queue.put(("status", self.rank, line))
            self._buf = ""

    def isatty(self) -> bool:
        return False


def _run_display(display_queue, num_gpus: int) -> None:
    """Display thread: one status row per GPU, permanent log lines above them.

    Messages are tuples:
      ("status", rank, text)  — overwrite GPU rank's row in-place
      ("log",    None, text)  — insert a permanent line above the status area
      None                    — stop sentinel

    Falls back to plain print() when stdout is not a TTY (e.g. redirected
    to a file), so escape codes never pollute log files.
    """
    rows = [f"GPU {i}: starting…" for i in range(num_gpus)]
    tty = sys.stdout.isatty()

    if tty:
        sys.stdout.write("\n".join(rows) + "\n")
        sys.stdout.flush()

    while True:
        msg = display_queue.get()
        if msg is None:
            break

        kind, rank, text = msg

        if not tty:
            if kind == "log":
                print(text, flush=True)
            continue

        if kind == "log":
            # Move up to the top of the status area, insert a blank line so
            # the status rows shift down, write the log line, then repaint.
            sys.stdout.write(
                f"\033[{num_gpus}A"   # cursor to top of status area
                f"\033[1L"            # insert blank line (CSI L — xterm/VTE)
                f"\033[2K\r{text}\n"  # write log, cursor now on first status row
            )
            for row in rows:
                sys.stdout.write(f"\033[2K\r{row}\n")
        else:
            rows[rank] = f"GPU {rank}: {text}"
            sys.stdout.write(f"\033[{num_gpus}A")
            for row in rows:
                sys.stdout.write(f"\033[2K\r{row}\n")

        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    cfg,
    ckpt_groups: list,      # ckpt_groups[rank] = [(step, path), ...]
    use_ema: bool,
    result_queue,           # mp.Queue — worker puts (step, metrics) as each ckpt finishes
    display_queue=None,     # mp.Queue — routes stdout to the live display thread
) -> None:
    """Spawned worker: restrict to cuda:rank and evaluate assigned checkpoints."""
    # Restrict CUDA visibility to this rank's GPU before any CUDA call.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    import logging
    logging.disable(logging.WARNING)

    # Route all prints (including progress lines from evaluate.py) to the
    # live display.  Falls back to normal stdout when no queue is provided.
    if display_queue is not None:
        sys.stdout = _QueueWriter(rank, display_queue)

    for step, ckpt_path in ckpt_groups[rank]:
        print(f"step {step} — {os.path.basename(ckpt_path)}")
        metrics = evaluate(cfg, ckpt_path, use_ema=use_ema)
        result_queue.put((step, metrics))

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser("JiT batch FID evaluation")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--no-ema", action="store_true",
                        help="Use raw model weights instead of EMA")
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

    # --- create a fresh wandb run for eval metrics --------------------------
    # Always create a new run rather than resuming the training run.
    # Resuming a closed offline run and appending records to it causes
    # protobuf parse errors during `wandb sync` on some wandb versions.
    # Name the run <experiment>-eval so it is easy to group with the
    # training run in the wandb UI.
    mode = "offline" if cfg.logging.wandb_offline else "online"
    metric_prefix = "raw" if args.no_ema else "ema"
    run = wandb.init(
        project=cfg.logging.wandb_project,
        name=f"{cfg.logging.wandb_experiment}-eval-{metric_prefix}",
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=output_dir,
        mode=mode,
    )
    print(f"Started wandb run {run.id!r} (mode={mode})")

    use_ema = not args.no_ema

    def _log_result(step: int, metrics: dict) -> None:
        if not metrics:
            return
        log_dict = {f"eval/{metric_prefix}/{k}": v for k, v in metrics.items()}
        run.log({"checkpoint_step": step, **log_dict})

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
        result_queue = ctx.Queue()
        display_queue = ctx.Queue()

        workers = [
            ctx.Process(
                target=_eval_worker,
                args=(rank, cfg, ckpt_groups, use_ema, result_queue, display_queue),
            )
            for rank in range(num_gpus)
        ]
        for w in workers:
            w.start()

        # Display thread: one live status row per GPU in the terminal.
        disp_thread = threading.Thread(
            target=_run_display, args=(display_queue, num_gpus), daemon=True
        )
        disp_thread.start()

        # Log each result as soon as any worker produces it.
        for _ in range(len(ckpts)):
            step, metrics = result_queue.get()
            if metrics:
                _log_result(step, metrics)
                log_dict = {f"eval/{metric_prefix}/{k}": v for k, v in metrics.items()}
                display_queue.put(("log", None, f"Logged {log_dict} at step {step}"))
            else:
                display_queue.put(("log", None, f"Step {step}: no metrics (FID stats unavailable)."))

        for w in workers:
            w.join()

        display_queue.put(None)   # stop display thread
        disp_thread.join()

    else:
        # Single GPU or CPU: evaluate sequentially
        for step, ckpt_path in ckpts:
            print(f"\n{'=' * 60}")
            print(f"Step {step}  —  {ckpt_path}")
            metrics = evaluate(cfg, ckpt_path, use_ema=use_ema)
            if metrics:
                _log_result(step, metrics)
                log_dict = {f"eval/{metric_prefix}/{k}": v for k, v in metrics.items()}
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
