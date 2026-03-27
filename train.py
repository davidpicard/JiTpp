#!/usr/bin/env python3
"""JiT training entry point using PyTorch Lightning.

Usage:
    # Full training run (JiT-B, 4 GPUs):
    torchrun --standalone --nproc_per_node=4 train.py configs/jit_b_4gpu.yaml

    # Debug run:
    python train.py configs/debug.yaml

    # Override individual config values at the command line:
    python train.py configs/jit_b_4gpu.yaml --override training.batch_size=64 data.path=/my/data

    # Explicit resume (auto-resume from last.ckpt in output_dir is on by default):
    python train.py configs/jit_b_4gpu.yaml --override logging.resume=./output/jit_b_4gpu/step-00050000.ckpt

SLURM / auto-requeue
--------------------
When running under SLURM, Lightning automatically requeueing the job when it
receives a preemption signal (SIGUSR1).  The last checkpoint written by
ModelCheckpoint is picked up on the next run via the auto-resume logic below,
so no manual intervention is needed after a job is killed or preempted.
"""
import argparse
import os

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.plugins.io import AsyncCheckpointIO

from callbacks import VisualizationCallback
from data_module import ImageNetDataModule
from lit_jit import JiTLightningModule


def main() -> None:
    parser = argparse.ArgumentParser("JiT Training")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dot-list config overrides, e.g. training.batch_size=64",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # pl.seed_everything is known to cause issues with Lightning's DDP workers
    # (broken RNG state propagation). Leave seeding to the user / launcher.

    # Required for torch.compile used inside the model blocks
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    os.makedirs(cfg.logging.output_dir, exist_ok=True)

    datamodule = ImageNetDataModule(cfg)
    module = JiTLightningModule(cfg)

    n_params = sum(p.numel() for p in module.denoiser.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.2f}M")

    logger = WandbLogger(
        project=cfg.logging.wandb_project,
        name=cfg.logging.wandb_experiment,
        save_dir=cfg.logging.output_dir,
        offline=cfg.logging.wandb_offline,
        log_model=False,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        VisualizationCallback(cfg),
        # Step-based checkpoint + last.ckpt symlink (used for auto-resume).
        # save_every_n_steps=0 disables periodic saves but last.ckpt is still
        # written at the end of training / on SLURM preemption.
        ModelCheckpoint(
            dirpath=cfg.logging.output_dir,
            filename="step-{step:08d}",
            every_n_train_steps=cfg.logging.save_every_n_steps or None,
            save_last="link",
            save_top_k=-1 if cfg.logging.save_every_n_steps else 0,
        ),
    ]

    # ------------------------------------------------------------------
    # Resume logic
    #   1. Explicit path in cfg.logging.resume  → use it (existing behaviour)
    #   2. No explicit path                     → auto-detect last.ckpt in
    #                                             output_dir (set by a previous
    #                                             run or SLURM requeue)
    # ------------------------------------------------------------------
    resume_path: str | None = cfg.logging.get("resume") or None
    if resume_path:
        if os.path.isdir(resume_path):
            candidate = os.path.join(resume_path, "last.ckpt")
            resume_path = candidate if os.path.isfile(candidate) else None
        elif not os.path.isfile(resume_path):
            print(f"Warning: resume path '{resume_path}' not found, training from scratch.")
            resume_path = None
    else:
        candidate = os.path.join(cfg.logging.output_dir, "last.ckpt")
        if os.path.isfile(candidate):
            resume_path = candidate
            print(f"Auto-resuming from {resume_path}")

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        devices=cfg.hardware.gpus_per_node,
        num_nodes=cfg.hardware.num_nodes,
        accelerator="gpu",
        strategy="ddp",
        precision=cfg.hardware.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.logging.log_freq,
        default_root_dir=cfg.logging.output_dir,
        benchmark=True,
        enable_progress_bar=True,
        plugins=[
            *([ SLURMEnvironment(auto_requeue=True)] if "SLURM_JOB_ID" in os.environ else []),
            AsyncCheckpointIO(),
        ],
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
