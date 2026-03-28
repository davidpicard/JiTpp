import copy
import math

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from denoiser import Denoiser
from util.image import normalize
from util.misc import add_weight_decay


class JiTLightningModule(pl.LightningModule):
    """PyTorch Lightning module wrapping the JiT denoising diffusion model."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.denoiser = Denoiser(
            model_arch=cfg.model.arch,
            img_size=cfg.model.img_size,
            num_classes=cfg.data.num_classes,
            attn_dropout=cfg.model.attn_dropout,
            proj_dropout=cfg.model.proj_dropout,
            mixer=cfg.model.get("mixer", "attention"),
            pom_degree=cfg.model.get("pom_degree", 3),
            pom_expand=cfg.model.get("pom_expand", 1),
            pom_n_groups=cfg.model.get("pom_n_groups", 1),
            label_drop_prob=cfg.diffusion.label_drop_prob,
            P_mean=cfg.diffusion.P_mean,
            P_std=cfg.diffusion.P_std,
            t_eps=cfg.diffusion.t_eps,
            noise_scale=cfg.diffusion.noise_scale,
            sampling_method=cfg.sampling.method,
            num_sampling_steps=cfg.sampling.num_steps,
            cfg_scale=cfg.sampling.cfg,
            interval_min=cfg.sampling.interval_min,
            interval_max=cfg.sampling.interval_max,
        )

        # EMA param lists — initialized lazily on first training batch
        self.ema_params1: list[torch.Tensor] | None = None
        self.ema_params2: list[torch.Tensor] | None = None

    def configure_model(self) -> None:
        # Called by Lightning after DDP wrapping, before training starts.
        # Compile net.forward — the JiT model used in both training and generation.
        # Compiling the Denoiser wrapper (denoiser.forward) would only speed up
        # the training loss path; generation calls self.net(...) directly and would
        # get no benefit.  Compiling only the method (not the module) keeps
        # state_dict keys unchanged (no _orig_mod prefix).
        # dynamic=True avoids recompilation when shapes change (e.g. batch size
        # during visualization, or seq-len change when in-context tokens are added).
        self.denoiser.net.forward = torch.compile(self.denoiser.net.forward, dynamic=True)

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    def _init_ema(self) -> None:
        self.ema_params1 = [p.detach().clone() for p in self.denoiser.parameters()]
        self.ema_params2 = [p.detach().clone() for p in self.denoiser.parameters()]

    def _ensure_ema_ready(self) -> None:
        """Initialize EMA on first call; move to the current device on device changes."""
        if self.ema_params1 is None:
            self._init_ema()
            return
        if self.ema_params1[0].device != self.device:
            self.ema_params1 = [p.to(self.device) for p in self.ema_params1]
            self.ema_params2 = [p.to(self.device) for p in self.ema_params2]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, labels = batch
        # Normalize uint8 images to [-1, 1]
        x = normalize(x)
        loss = self.denoiser(x, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=False,
                 sync_dist=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self._ensure_ema_ready()
        d1 = self.cfg.diffusion.ema_decay1
        d2 = self.cfg.diffusion.ema_decay2
        for targ, src in zip(self.ema_params1, self.denoiser.parameters()):
            targ.detach().mul_(d1).add_(src.detach(), alpha=1.0 - d1)
        for targ, src in zip(self.ema_params2, self.denoiser.parameters()):
            targ.detach().mul_(d2).add_(src.detach(), alpha=1.0 - d2)

    # ------------------------------------------------------------------
    # Checkpoint: persist EMA alongside the regular state dict
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_params1 is not None:
            checkpoint["ema_params1"] = [p.cpu().clone() for p in self.ema_params1]
            checkpoint["ema_params2"] = [p.cpu().clone() for p in self.ema_params2]

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state = checkpoint.get("state_dict", {})

        # Migrate old ComPoM checkpoints: po_proj+se_proj → qc_proj+se_bias.
        # Old format: po_proj.weight (2-D Linear), se_proj.weight, se_proj.bias.
        # New format: qc_proj.weight = cat([po_proj.weight, se_proj.weight], dim=0),
        #             se_bias = se_proj.bias.
        # The n_groups>1 path uses Conv1d (3-D weight) and is unchanged.
        migrated = False
        new_state: dict = {}
        skip: set = set()
        for k, v in state.items():
            if k in skip:
                continue
            if k.endswith(".po_proj.weight") and v.dim() == 2:
                base = k[: -len(".po_proj.weight")]
                se_w_k = base + ".se_proj.weight"
                se_b_k = base + ".se_proj.bias"
                new_state[base + ".qc_proj.weight"] = torch.cat([v, state[se_w_k]], dim=0)
                new_state[base + ".se_bias"] = state[se_b_k]
                skip.update({se_w_k, se_b_k})
                migrated = True
            else:
                new_state[k] = v
        if migrated:
            checkpoint["state_dict"] = new_state
            # EMA param list order changed; discard stale EMA so it is
            # re-initialised from the loaded weights on the first batch.
            checkpoint.pop("ema_params1", None)
            checkpoint.pop("ema_params2", None)
            self.ema_params1 = None
            self.ema_params2 = None
            print("Migrated ComPoM checkpoint: po_proj+se_proj → qc_proj+se_bias (EMA reset)")
            return

        if "ema_params1" in checkpoint:
            # Keep on CPU; _ensure_ema_ready() will move them on the first batch
            self.ema_params1 = checkpoint.pop("ema_params1")
            self.ema_params2 = checkpoint.pop("ema_params2")

    # ------------------------------------------------------------------
    # Optimizer + LR schedule
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        param_groups = add_weight_decay(self.denoiser, self.cfg.training.weight_decay)

        # Scale base LR by effective global batch size
        eff_batch = self.cfg.training.batch_size * self.trainer.world_size
        lr = self.cfg.training.blr * eff_batch / 256.0

        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

        # Per-step LR schedule matching the original warmup + constant/cosine logic
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(
            self.cfg.training.warmup_epochs * total_steps / self.cfg.training.epochs
        )
        min_lr = self.cfg.training.min_lr
        schedule = self.cfg.training.lr_schedule

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            if schedule == "constant":
                return 1.0
            # cosine decay
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr / lr + (1.0 - min_lr / lr) * decay

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
