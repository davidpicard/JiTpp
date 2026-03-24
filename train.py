#!/usr/bin/env python3
"""JiT training entry point using PyTorch Lightning.

Usage:
    # Full training run (JiT-B, 4 GPUs):
    python train.py configs/jit_b_4gpu.yaml

    # Debug run:
    python train.py configs/debug.yaml

    # Override individual config values at the command line:
    python train.py configs/jit_b_4gpu.yaml --override training.batch_size=64 data.path=/my/data

    # Resume from a checkpoint directory:
    python train.py configs/jit_b_4gpu.yaml --override logging.resume=./output/jit_b_4gpu/last.ckpt
"""
import argparse
import os

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

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
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        VisualizationCallback(cfg),
        ModelCheckpoint(
            dirpath=cfg.logging.output_dir,
            filename="checkpoint-{epoch:04d}",
            every_n_epochs=cfg.logging.save_freq,
            save_last=True,
            save_top_k=-1,   # keep all periodic checkpoints
        ),
    ]

    # Resolve resume path: accept either a .ckpt file or a directory
    resume_path: str | None = cfg.logging.get("resume") or None
    if resume_path:
        if os.path.isdir(resume_path):
            candidate = os.path.join(resume_path, "last.ckpt")
            resume_path = candidate if os.path.isfile(candidate) else None
        elif not os.path.isfile(resume_path):
            print(f"Warning: resume path '{resume_path}' not found, training from scratch.")
            resume_path = None

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
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
