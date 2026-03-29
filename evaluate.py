#!/usr/bin/env python3
"""Standalone evaluation script: generate images from a JiT checkpoint and compute FID/IS.

Usage:
    python evaluate.py configs/jit_b_4gpu.yaml path/to/checkpoint.ckpt

    # Use raw model weights instead of EMA:
    python evaluate.py configs/jit_b_4gpu.yaml path/to/checkpoint.ckpt --no-ema

    # Override sampling settings:
    python evaluate.py configs/jit_b_4gpu.yaml path/to/checkpoint.ckpt \\
        --override sampling.cfg=2.5 sampling.num_steps=100

Always runs on a single GPU (or CPU).  For multi-checkpoint parallelism use eval_all.py.
"""
import argparse
import copy
import os

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.linalg import sqrtm

from lit_jit import JiTLightningModule
from util.image import denormalize


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_fid(mu_gen, sigma_gen, mu_ref, sigma_ref) -> float:
    """Frechet Inception Distance between two Gaussians."""
    diff = mu_gen - mu_ref
    covmean, _ = sqrtm(sigma_gen @ sigma_ref, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2.0 * covmean))


def _compute_is(logits: np.ndarray, splits: int = 10) -> float:
    """Inception Score from raw logits (shape N x num_classes)."""
    p_yx = torch.from_numpy(logits).float().softmax(dim=-1).numpy()
    scores = []
    n = p_yx.shape[0]
    for k in range(splits):
        part = p_yx[k * (n // splits): (k + 1) * (n // splits)]
        p_y = part.mean(axis=0, keepdims=True)
        kl = (part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))).sum(axis=1)
        scores.append(np.exp(kl.mean()))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_denoiser(device: torch.device, cfg, ckpt_path: str, use_ema: bool):
    """Load checkpoint, move to device, optionally swap to EMA weights."""
    module = JiTLightningModule.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu")
    module = module.to(device).eval()

    if module.ema_params1 is not None:
        module.ema_params1 = [p.to(device) for p in module.ema_params1]
        module.ema_params2 = [p.to(device) for p in module.ema_params2]

    denoiser = module.denoiser

    if use_ema and module.ema_params1 is not None:
        print("Switching to EMA weights...")
        ema_state = copy.deepcopy(denoiser.state_dict())
        for i, (name, _) in enumerate(denoiser.named_parameters()):
            ema_state[name] = module.ema_params1[i]
        denoiser.load_state_dict(ema_state)
    else:
        print("Using raw model weights (no EMA).")

    # Compile net.forward — the JiT model called inside both training forward
    # and generation.  Compiling denoiser.forward (the loss wrapper) would have
    # no effect on generation, which calls self.net(...) directly.
    denoiser.net.forward = torch.compile(denoiser.net.forward, dynamic=True)
    return denoiser


# ---------------------------------------------------------------------------
# Generation + in-memory feature extraction (single GPU)
# ---------------------------------------------------------------------------

def _generate_and_extract(
    denoiser, labels_all: np.ndarray, batch_size: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate images and extract InceptionV3 features in memory.

    Uses a secondary CUDA stream to overlap InceptionV3 with the next
    generation step.  Returns (pool3_np [N,2048], logits_np [N,1008]).
    """
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    cuda = device.type == "cuda"
    inception = FeatureExtractorInceptionV3(
        name="inception-v3-compat",
        features_list=["2048", "logits_unbiased"],
    ).to(device).eval()

    inception_stream = torch.cuda.Stream(device=device) if cuda else None

    pool3_list: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    pending_imgs: torch.Tensor | None = None
    pending_event: torch.cuda.Event | None = None

    def _extract(imgs_uint8: torch.Tensor) -> None:
        nonlocal pending_event
        if inception_stream is not None:
            with torch.cuda.stream(inception_stream):
                inception_stream.wait_event(pending_event)
                imgs_uint8.record_stream(inception_stream)
                with torch.no_grad():
                    feats = inception(imgs_uint8)
                pool3_list.append(feats[0].float().cpu())
                logits_list.append(feats[1].float().cpu())
            # Do NOT synchronize here — let generation of the next batch
            # overlap with InceptionV3 on the secondary stream.
        else:
            with torch.no_grad():
                feats = inception(imgs_uint8)
            pool3_list.append(feats[0].float())
            logits_list.append(feats[1].float())

    num_images = len(labels_all)
    img_idx = 0

    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            if pending_imgs is not None:
                _extract(pending_imgs)
                pending_imgs = None

            end = min(start + batch_size, num_images)
            labels = torch.tensor(labels_all[start:end], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                images = denoiser.generate(labels)

            images_f = denormalize(images).clamp(0.0, 1.0)
            imgs_uint8 = (images_f * 255).to(torch.uint8).detach()

            if cuda:
                pending_event = torch.cuda.Event()
                pending_event.record()

            pending_imgs = imgs_uint8
            img_idx += end - start
            print(f"  {img_idx}/{num_images} images", end="\r", flush=True)

    if pending_imgs is not None:
        _extract(pending_imgs)
    if inception_stream is not None:
        inception_stream.synchronize()
    print()

    pool3_np = torch.cat(pool3_list, dim=0).numpy().astype(np.float64)
    logits_np = torch.cat(logits_list, dim=0).numpy()
    return pool3_np, logits_np


# ---------------------------------------------------------------------------
# Public evaluate() entry point
# ---------------------------------------------------------------------------

def evaluate(cfg, ckpt_path: str, use_ema: bool = True, output_dir: str | None = None) -> dict:
    torch.set_float32_matmul_precision("high")

    if output_dir is None:
        output_dir = cfg.logging.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_images = cfg.sampling.num_images
    batch_size = cfg.sampling.gen_batch_size
    num_classes = cfg.data.num_classes
    img_size = cfg.model.img_size

    assert num_images % num_classes == 0, \
        f"num_images ({num_images}) must be divisible by num_classes ({num_classes})"
    labels_all = np.arange(num_classes).repeat(num_images // num_classes)

    fid_stats_map = {
        256: "fid_stats/jit_in256_stats.npz",
        512: "fid_stats/jit_in512_stats.npz",
    }
    fid_stats_file = fid_stats_map.get(img_size)
    has_fid_stats = bool(fid_stats_file and os.path.exists(fid_stats_file))

    print(f"Loading checkpoint: {ckpt_path}")
    denoiser = _load_denoiser(device, cfg, ckpt_path, use_ema)

    metrics: dict = {}

    if has_fid_stats:
        print(f"Generating {num_images} images (in-memory)...")
        all_pool3, all_logits = _generate_and_extract(denoiser, labels_all, batch_size, device)

        mu_gen = all_pool3.mean(axis=0)
        sigma_gen = np.cov(all_pool3, rowvar=False)
        ref = np.load(fid_stats_file)
        fid_score = _compute_fid(
            mu_gen, sigma_gen,
            ref["mu"].astype(np.float64), ref["sigma"].astype(np.float64),
        )
        is_score = _compute_is(all_logits)
        print(f"FID: {fid_score:.4f}   IS: {is_score:.4f}")
        metrics = {"fid": fid_score, "is": is_score}

    else:
        s = cfg.sampling
        folder_name = "gen-{}-steps{}-cfg{}-interval{:.2f}-{:.2f}-n{}-res{}".format(
            s.method, s.num_steps, s.cfg, s.interval_min, s.interval_max,
            num_images, img_size,
        )
        save_dir = os.path.join(output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Generating {num_images} images → {save_dir}")

        img_idx = 0
        with torch.no_grad():
            for start in range(0, num_images, batch_size):
                end = min(start + batch_size, num_images)
                labels = torch.tensor(labels_all[start:end], dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    images = denoiser.generate(labels)
                images = denormalize(images).clamp(0.0, 1.0).detach().cpu().numpy()
                for b in range(images.shape[0]):
                    img = np.round(images[b].transpose(1, 2, 0) * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, f"{img_idx:05d}.png"), img[:, :, ::-1])
                    img_idx += 1
                print(f"  {img_idx}/{num_images} images saved", end="\r")
        print()
        print(f"No FID stats for {img_size}px — images saved at {save_dir}.")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
