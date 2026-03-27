#!/usr/bin/env python3
"""Standalone evaluation script: generate images from a JiT checkpoint and compute FID/IS.

Usage:
    python evaluate.py configs/jit_b_4gpu.yaml path/to/checkpoint.ckpt

    # Use raw model weights instead of EMA:
    python evaluate.py configs/jit_b_4gpu.yaml path/to/checkpoint.ckpt --no-ema

    # Override sampling settings:
    python evaluate.py configs/jit_b_4gpu.yaml path/to/checkpoint.ckpt \\
        --override sampling.cfg=2.5 sampling.num_steps=100

This script runs on a single GPU. For distributed generation, extend accordingly.
"""
import argparse
import copy
import os
import shutil

import cv2
import numpy as np
import torch
import torch_fidelity
from omegaconf import OmegaConf

from lit_jit import JiTLightningModule
from util.image import denormalize


def evaluate(cfg, ckpt_path: str, use_ema: bool = True, output_dir: str | None = None) -> dict:
    if output_dir is None:
        output_dir = cfg.logging.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {ckpt_path}")
    module = JiTLightningModule.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu")
    module = module.to(device)
    module.eval()

    # Move EMA params to device (they are stored on CPU in the checkpoint)
    if module.ema_params1 is not None:
        module.ema_params1 = [p.to(device) for p in module.ema_params1]
        module.ema_params2 = [p.to(device) for p in module.ema_params2]

    denoiser = module.denoiser

    # Swap to EMA weights for generation
    orig_state: dict | None = None
    if use_ema and module.ema_params1 is not None:
        print("Switching to EMA weights...")
        orig_state = copy.deepcopy(denoiser.state_dict())
        ema_state = copy.deepcopy(denoiser.state_dict())
        for i, (name, _) in enumerate(denoiser.named_parameters()):
            ema_state[name] = module.ema_params1[i]
        denoiser.load_state_dict(ema_state)
    else:
        print("Using raw model weights (no EMA).")

    num_images = cfg.sampling.num_images
    batch_size = cfg.sampling.gen_batch_size
    num_classes = cfg.data.num_classes
    img_size = cfg.model.img_size

    folder_name = "gen-{}-steps{}-cfg{}-interval{:.2f}-{:.2f}-n{}-res{}".format(
        denoiser.method, denoiser.steps, denoiser.cfg_scale,
        denoiser.cfg_interval[0], denoiser.cfg_interval[1],
        num_images, img_size,
    )
    save_dir = os.path.join(output_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating {num_images} images → {save_dir}")

    assert num_images % num_classes == 0, \
        f"num_images ({num_images}) must be divisible by num_classes ({num_classes})"
    labels_all = np.arange(num_classes).repeat(num_images // num_classes)

    img_idx = 0
    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)
            labels = torch.tensor(labels_all[start:end], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                images = denoiser.generate(labels)

            images = denormalize(images).clamp(0.0, 1.0)
            images = images.detach().cpu().numpy()

            for b in range(images.shape[0]):
                img = np.round(images[b].transpose(1, 2, 0) * 255).astype(np.uint8)
                img_bgr = img[:, :, ::-1]
                cv2.imwrite(os.path.join(save_dir, f"{img_idx:05d}.png"), img_bgr)
                img_idx += 1

            print(f"  {img_idx}/{num_images} images generated", end="\r")

    print()  # newline after progress

    # Restore original weights
    if orig_state is not None:
        denoiser.load_state_dict(orig_state)

    # Compute FID / IS against precomputed reference statistics
    fid_stats_map = {
        256: "fid_stats/jit_in256_stats.npz",
        512: "fid_stats/jit_in512_stats.npz",
    }
    fid_stats_file = fid_stats_map.get(img_size)
    metrics: dict = {}

    if fid_stats_file and os.path.exists(fid_stats_file):
        print("Computing FID and IS...")
        cuda = torch.cuda.is_available()

        # IS only needs input1
        isc_result = torch_fidelity.calculate_metrics(
            input1=save_dir,
            cuda=cuda,
            isc=True, fid=False, kid=False, prc=False,
            verbose=False,
        )
        metrics["is"] = isc_result[torch_fidelity.KEY_METRIC_ISC_MEAN]

        # FID: extract inception stats for generated images, then compare
        # against precomputed reference stats (torch_fidelity 0.4 dropped
        # fid_statistics_file as a top-level param, so we use the lower-level API).
        from torch_fidelity.metric_fid import (
            create_feature_extractor,
            fid_input_id_to_statistics_cached,
            fid_statistics_to_metric,
            resolve_feature_extractor,
            resolve_feature_layer_for_metric,
        )
        fid_kwargs = dict(input1=save_dir, cuda=cuda, fid=True, verbose=False)
        extractor_name = resolve_feature_extractor(**fid_kwargs)
        feat_layer    = resolve_feature_layer_for_metric("fid", **fid_kwargs)
        feat_extractor = create_feature_extractor(extractor_name, [feat_layer], **fid_kwargs)
        stats_gen = fid_input_id_to_statistics_cached(1, feat_extractor, feat_layer, **fid_kwargs)

        ref = np.load(fid_stats_file)
        stats_ref = {
            "mu":    ref["mu"].astype(stats_gen["mu"].dtype),
            "sigma": ref["sigma"].astype(stats_gen["sigma"].dtype),
        }

        metrics["fid"] = fid_statistics_to_metric(stats_gen, stats_ref, False)[torch_fidelity.KEY_METRIC_FID]

        print(f"FID: {metrics['fid']:.4f}   IS: {metrics['is']:.4f}")
        shutil.rmtree(save_dir)
    else:
        print(f"No FID stats available for resolution {img_size}px — images saved at {save_dir}.")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser("JiT Evaluation")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("checkpoint", type=str, help="Path to .ckpt checkpoint file")
    parser.add_argument("--no-ema", action="store_true",
                        help="Use raw model weights instead of EMA weights")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for generated images")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Dot-list config overrides")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    evaluate(cfg, args.checkpoint, use_ema=not args.no_ema, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
