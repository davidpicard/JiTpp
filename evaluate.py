#!/usr/bin/env python3
"""Standalone evaluation script: generate images from a JiT checkpoint and compute FID/IS.

Usage:
    python evaluate.py configs/jit_b_4gpu.yaml path/to/checkpoint.ckpt

    # Use raw model weights instead of EMA:
    python evaluate.py configs/jit_b_4gpu.yaml path/to/checkpoint.ckpt --no-ema

    # Override sampling settings:
    python evaluate.py configs/jit_b_4gpu.yaml path/to/checkpoint.ckpt \\
        --override sampling.cfg=2.5 sampling.num_steps=100

All available GPUs are used automatically.  Each GPU generates an equal share of
images; features are extracted in-memory and aggregated on the coordinator process.
"""
import argparse
import copy
import os

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
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
        ema_state = copy.deepcopy(denoiser.state_dict())
        for i, (name, _) in enumerate(denoiser.named_parameters()):
            ema_state[name] = module.ema_params1[i]
        denoiser.load_state_dict(ema_state)

    return denoiser


# ---------------------------------------------------------------------------
# Per-rank generation + in-memory feature extraction
# ---------------------------------------------------------------------------

def _generate_and_extract(
    denoiser, labels_chunk: np.ndarray, batch_size: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate images for labels_chunk and extract InceptionV3 features.

    Uses a CUDA stream to overlap InceptionV3 with the next generation step.
    Returns (pool3_np [N,2048], logits_np [N,1008]).
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
            inception_stream.synchronize()
        else:
            with torch.no_grad():
                feats = inception(imgs_uint8)
            pool3_list.append(feats[0].float())
            logits_list.append(feats[1].float())

    num_chunk = len(labels_chunk)
    img_idx = 0
    rank_prefix = f"[cuda:{device.index}] " if cuda else ""

    with torch.no_grad():
        for start in range(0, num_chunk, batch_size):
            if pending_imgs is not None:
                _extract(pending_imgs)
                pending_imgs = None

            end = min(start + batch_size, num_chunk)
            labels = torch.tensor(labels_chunk[start:end], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                images = denoiser.generate(labels)

            images_f = denormalize(images).clamp(0.0, 1.0)
            imgs_uint8 = (images_f * 255).to(torch.uint8).detach()

            if cuda:
                pending_event = torch.cuda.Event()
                pending_event.record()

            pending_imgs = imgs_uint8
            img_idx += end - start
            print(f"  {rank_prefix}{img_idx}/{num_chunk} images", end="\r", flush=True)

    if pending_imgs is not None:
        _extract(pending_imgs)
    print()

    pool3_np = torch.cat(pool3_list, dim=0).numpy().astype(np.float64)
    logits_np = torch.cat(logits_list, dim=0).numpy()
    return pool3_np, logits_np


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------

def _worker_in_memory(
    rank: int,
    world_size: int,
    cfg,
    ckpt_path: str,
    use_ema: bool,
    labels_all: np.ndarray,
    batch_size: int,
    result_dict,           # Manager().dict()
) -> None:
    """Spawned worker: load model on cuda:rank, generate label chunk, extract features."""
    import logging
    logging.disable(logging.WARNING)

    # Fix 3: diversify RNG so each worker generates different noise patterns.
    torch.manual_seed(torch.initial_seed() + rank)

    device = torch.device(f"cuda:{rank}")
    denoiser = _load_denoiser(device, cfg, ckpt_path, use_ema)

    # Fix 2: interleaved (round-robin) split — every GPU sees a representative
    # cross-section of all classes, avoiding a degenerate per-GPU distribution.
    labels_chunk = labels_all[rank::world_size]

    pool3_np, logits_np = _generate_and_extract(denoiser, labels_chunk, batch_size, device)
    result_dict[rank] = (pool3_np, logits_np)


def _worker_to_disk(
    rank: int,
    world_size: int,
    cfg,
    ckpt_path: str,
    use_ema: bool,
    labels_all: np.ndarray,
    batch_size: int,
    save_dir: str,
) -> None:
    """Spawned worker: generate label chunk and save images with correct global indices."""
    import logging
    logging.disable(logging.WARNING)

    torch.manual_seed(torch.initial_seed() + rank)

    device = torch.device(f"cuda:{rank}")
    denoiser = _load_denoiser(device, cfg, ckpt_path, use_ema)

    # Interleaved split: rank r handles labels_all[r], labels_all[r+N], ...
    # Global file index for the i-th image of this rank = rank + i*world_size.
    labels_chunk = labels_all[rank::world_size]

    with torch.no_grad():
        for i_start in range(0, len(labels_chunk), batch_size):
            i_end = min(i_start + batch_size, len(labels_chunk))
            labels = torch.tensor(labels_chunk[i_start:i_end], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                images = denoiser.generate(labels)
            images = denormalize(images).clamp(0.0, 1.0).detach().cpu().numpy()
            for b in range(images.shape[0]):
                global_idx = rank + (i_start + b) * world_size
                img = np.round(images[b].transpose(1, 2, 0) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, f"{global_idx:05d}.png"), img[:, :, ::-1])


# ---------------------------------------------------------------------------
# Public evaluate() entry point
# ---------------------------------------------------------------------------

def evaluate(cfg, ckpt_path: str, use_ema: bool = True, output_dir: str | None = None) -> dict:
    if output_dir is None:
        output_dir = cfg.logging.output_dir

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"GPUs available: {num_gpus}")

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
    ema_str = "EMA" if use_ema else "raw"
    print(f"Weights: {ema_str}   images: {num_images}   batch: {batch_size}")

    metrics: dict = {}

    if has_fid_stats:
        if num_gpus > 1:
            # Fix 1: load InceptionV3 in the main process before spawning workers so
            # its weights are fully cached on disk.  Multiple workers simultaneously
            # downloading to the same Lustre/NFS path causes cache corruption and
            # produces garbage features → absurd FID.
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
            print("Warming InceptionV3 weight cache...")
            FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=["2048"])
            print(f"Using {num_gpus} GPUs (in-memory pipeline)...")
            ctx = mp.get_context("spawn")
            manager = ctx.Manager()
            result_dict = manager.dict()
            mp.spawn(
                _worker_in_memory,
                args=(num_gpus, cfg, ckpt_path, use_ema, labels_all, batch_size, result_dict),
                nprocs=num_gpus,
                join=True,
            )
            # Aggregate features from all workers in rank order
            pool3_parts = [result_dict[r][0] for r in range(num_gpus)]
            logits_parts = [result_dict[r][1] for r in range(num_gpus)]
            all_pool3 = np.concatenate(pool3_parts, axis=0)
            all_logits = np.concatenate(logits_parts, axis=0)
        else:
            print("Using 1 GPU (in-memory pipeline)..." if num_gpus == 1 else "Using CPU (in-memory)...")
            device = torch.device("cuda" if num_gpus == 1 else "cpu")
            denoiser = _load_denoiser(device, cfg, ckpt_path, use_ema)
            print(f"Generating {num_images} images...")
            all_pool3, all_logits = _generate_and_extract(denoiser, labels_all, batch_size, device)

        # Compute metrics
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
        # Fallback: save images to disk
        folder_name = "gen-{}-steps{}-cfg{}-interval{:.2f}-{:.2f}-n{}-res{}".format(
            *_denoiser_id(cfg, ckpt_path, use_ema), num_images, img_size,
        )
        save_dir = os.path.join(output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Generating {num_images} images → {save_dir}")

        if num_gpus > 1:
            mp.spawn(
                _worker_to_disk,
                args=(num_gpus, cfg, ckpt_path, use_ema, labels_all, batch_size, save_dir),
                nprocs=num_gpus,
                join=True,
            )
        else:
            device = torch.device("cuda" if num_gpus == 1 else "cpu")
            denoiser = _load_denoiser(device, cfg, ckpt_path, use_ema)
            _save_to_disk(denoiser, labels_all, batch_size, device, save_dir, offset=0)

        print(f"No FID stats for {img_size}px — images saved at {save_dir}.")

    return metrics


def _denoiser_id(cfg, ckpt_path: str, use_ema: bool) -> tuple:
    """Return (method, steps, cfg_scale, interval_min, interval_max) from cfg for folder naming."""
    s = cfg.sampling
    return s.method, s.num_steps, s.cfg, s.interval_min, s.interval_max


def _save_to_disk(denoiser, labels_chunk, batch_size, device, save_dir, offset=0):
    img_idx = offset
    with torch.no_grad():
        for start in range(0, len(labels_chunk), batch_size):
            end = min(start + batch_size, len(labels_chunk))
            labels = torch.tensor(labels_chunk[start:end], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                images = denoiser.generate(labels)
            images = denormalize(images).clamp(0.0, 1.0).detach().cpu().numpy()
            for b in range(images.shape[0]):
                img = np.round(images[b].transpose(1, 2, 0) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, f"{img_idx:05d}.png"), img[:, :, ::-1])
                img_idx += 1
            print(f"  {img_idx - offset}/{len(labels_chunk)} images saved", end="\r")
    print()


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
