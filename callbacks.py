import copy

import numpy as np
import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid

from util.image import denormalize

# Fixed set of ImageNet classes to visualize (covers a range of categories)
VIS_CLASSES = [1, 9, 18, 249, 928, 949, 888, 409, 980]

CLASS_NAMES = {
    1:   "goldfish",
    9:   "hen",
    18:  "magpie",
    249: "malamute",
    928: "ice cream",
    949: "strawberry",
    888: "vase",
    409: "analog clock",
    980: "volcano",
}

# Captions are shared across all weight/CFG combinations
_CAPTIONS = [f"{CLASS_NAMES[c]} (cls {c})" for c in VIS_CLASSES]


class VisualizationCallback(pl.Callback):
    """Log 2×2 sample grids for a fixed set of ImageNet classes to wandb.

    At every ``vis_every_n_steps`` training steps, four samples per class are
    generated in four panels:
      - **raw/no_cfg**  : raw model weights, guidance scale = 1.0
      - **raw/cfg**     : raw model weights, guidance scale = vis_cfg
      - **ema/no_cfg**  : EMA weights (when available), guidance scale = 1.0
      - **ema/cfg**     : EMA weights (when available), guidance scale = vis_cfg

    Only runs on global rank 0 to avoid stalling data-parallel training.
    """

    def __init__(self, cfg):
        super().__init__()
        self.every_n_steps = cfg.logging.vis_every_n_steps
        self.vis_cfg = cfg.logging.vis_cfg
        self.vis_sampling_steps = cfg.logging.vis_sampling_steps

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return
        self._generate_and_log(trainer, pl_module)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_and_log(self, trainer, pl_module) -> None:
        denoiser = pl_module.denoiser
        device = pl_module.device

        was_training = denoiser.training
        denoiser.eval()

        orig_cfg_scale = denoiser.cfg_scale
        orig_steps = denoiser.steps
        denoiser.steps = self.vis_sampling_steps

        try:
            # --- raw model weights ----------------------------------------
            raw_panels = self._run_generation(denoiser, device)

            # --- EMA weights (if initialised) -----------------------------
            ema_panels = None
            if pl_module.ema_params1 is not None:
                orig_state = copy.deepcopy(denoiser.state_dict())
                try:
                    ema_state = copy.deepcopy(denoiser.state_dict())
                    for i, (name, _) in enumerate(denoiser.named_parameters()):
                        ema_state[name] = pl_module.ema_params1[i]
                    denoiser.load_state_dict(ema_state)
                    ema_panels = self._run_generation(denoiser, device)
                finally:
                    denoiser.load_state_dict(orig_state)
        finally:
            denoiser.cfg_scale = orig_cfg_scale
            denoiser.steps = orig_steps
            if was_training:
                denoiser.train()

        # --- log to wandb -------------------------------------------------
        step = trainer.global_step
        for mode in ("no_cfg", "cfg"):
            trainer.logger.log_image(
                key=f"samples/raw_{mode}",
                images=raw_panels[mode],
                caption=_CAPTIONS,
                step=step,
            )
        if ema_panels is not None:
            for mode in ("no_cfg", "cfg"):
                trainer.logger.log_image(
                    key=f"samples/ema_{mode}",
                    images=ema_panels[mode],
                    caption=_CAPTIONS,
                    step=step,
                )

    def _run_generation(self, denoiser, device) -> dict[str, list]:
        """Generate 2×2 grids for both CFG modes across all visualization classes."""
        panels: dict[str, list] = {}
        for cfg_scale, mode in [(1.0, "no_cfg"), (self.vis_cfg, "cfg")]:
            denoiser.cfg_scale = cfg_scale
            grids = []
            for cls in VIS_CLASSES:
                labels = torch.full((4,), cls, dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    imgs = denoiser.generate(labels)
                imgs = denormalize(imgs).clamp(0.0, 1.0).float().cpu()
                grid = make_grid(imgs, nrow=2, padding=2)
                grids.append((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            panels[mode] = grids
        return panels
