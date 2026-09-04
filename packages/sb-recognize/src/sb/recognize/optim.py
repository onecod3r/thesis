"""Optimization pieces the 1st-place recipe needs and torch does not ship.

The reference trains with ``RectifiedAdam(schedule, weight_decay=lr*0.1)``
wrapped in ``Lookahead(sync_period=5)``, a cosine one-cycle LR, and Adversarial
Weight Perturbation from epoch 15. Torch has RAdam; the other three live here.

Mapping notes, because they are easy to get subtly wrong:

- **Weight decay.** tfa applies ``p -= decay * p`` with ``decay = lr * 0.1``;
  torch's decoupled implementations apply ``p -= lr * wd * p``. So
  ``weight_decay=0.1`` in torch reproduces the reference exactly — the schedule
  on the decay term is already implied by the schedule on the LR.
- **RAdam rectification.** tfa exposes ``sma_threshold=4``; torch hard-codes 5.
  This only changes how many warm-up steps run in SGD-with-momentum mode before
  the adaptive term switches on, and cannot be configured away.
- **AWP is scale-invariant**, which is what lets it coexist with a GradScaler
  without a second ``unscale_`` — see :class:`AWP`.
"""

import math
from contextlib import contextmanager

import torch


def cosine_one_cycle(optimizer, total_steps: int, warmup_steps: int = 0,
                     lr_min_ratio: float = 0.002):
    """Linear warmup then cosine decay to ``lr_min_ratio * lr``, stepped per
    BATCH (the reference schedules per step, not per epoch).

    The reference config sets ``warmup=0``, so the default here is a pure cosine
    — but warmup is kept available because this port runs at a different batch
    size, and a shorter batch means a noisier first few hundred steps.
    """
    total_steps = max(1, int(total_steps))

    def ratio(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return lr_min_ratio + 0.5 * (1 - lr_min_ratio) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, ratio)


class Lookahead:
    """Lookahead (Zhang et al. 2019), ``sync_period=5`` in the reference.

    Keeps a set of "slow" weights that are pulled ``alpha`` of the way toward the
    fast weights every ``k`` steps, then copied back over them.

    This is deliberately **not** an optimizer wrapper. Wrapping means
    ``GradScaler`` has to accept a non-``Optimizer`` (it duck-types fine at
    runtime, but it is typed against ``torch.optim.Optimizer`` and the repo type
    checks with ``ty``). Instead the base optimizer stays the one the scaler
    sees, and the slow-weight update is a separate call after it::

        scaler.step(base_opt); scaler.update(); lookahead.sync()

    which is exactly equivalent — Lookahead only ever acts *after* a completed
    base step — and keeps both the scaler and the type checker happy.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, k: int = 5,
                 alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow = [[p.detach().clone() for p in g["params"]]
                     for g in optimizer.param_groups]

    def sync(self) -> bool:
        """Call once after every completed optimizer step. Returns True on the
        steps where the slow weights were actually synced."""
        self.step_count += 1
        if self.step_count % self.k:
            return False
        with torch.no_grad():
            for group, slow_group in zip(self.optimizer.param_groups, self.slow):
                for fast, slow in zip(group["params"], slow_group):
                    slow.add_(fast.detach() - slow, alpha=self.alpha)
                    fast.detach().copy_(slow)
        return True

    # --- resume ----------------------------------------------------------
    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "slow": self.slow}

    def load_state_dict(self, sd: dict) -> None:
        self.step_count = sd["step_count"]
        self.slow = [[t.to(p.device) for t, p in zip(group_slow, group["params"])]
                     for group_slow, group in zip(sd["slow"],
                                                  self.optimizer.param_groups)]


class AWP:
    """Adversarial Weight Perturbation — the reference's main regularizer.

    Each step, after the normal backward, every weight is nudged in the
    direction that most increases the loss::

        w += delta * (g / ||g||) * ||w||

    a second forward/backward is taken at that perturbed point, and the weights
    are restored before the optimizer applies the *adversarial* gradient. The
    model is pushed toward a flatter minimum, which is exactly the right tool
    here given the plateau was diagnosed as overfitting (TODO §7.1).

    **Cost: it roughly doubles per-step time** (two forward/backward passes).

    **Interaction with AMP.** With ``eps=0`` the perturbation
    ``delta * g/||g|| * ||w||`` is invariant to any uniform rescaling of ``g``,
    so it can be computed straight from the *scaled* gradients — no second
    ``unscale_`` call, which ``GradScaler`` would reject. Non-finite gradients
    (a scaler overflow) skip the perturbation for that step.

    **Stability.** The reference author reports NaN losses at ``delta=0.2``
    roughly one run in five. Halve it (or lower the LR) if that happens; the
    notebook's config exposes both.
    """

    def __init__(self, model, delta: float = 0.2, eps: float = 0.0,
                 start_step: int = 0):
        self.model = model
        self.delta = delta
        self.eps = eps
        self.start_step = start_step
        self._backup: dict[str, torch.Tensor] = {}

    def active(self, step: int) -> bool:
        return self.delta > 0 and step >= self.start_step

    def perturb(self) -> bool:
        """Move every weight along its gradient. Returns False (and changes
        nothing) if the gradients are not finite."""
        self._backup = {}
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is None or p.dim() <= 1:
                    continue  # biases / norm scales: perturbing them is noise
                grad_norm = torch.norm(p.grad)
                weight_norm = torch.norm(p)
                if not torch.isfinite(grad_norm) or grad_norm == 0:
                    self.restore()  # undo any partial perturbation
                    return False
                self._backup[name] = p.detach().clone()
                p.add_(self.delta * p.grad / (grad_norm + self.eps)
                       * (weight_norm + self.eps))
        return True

    def restore(self) -> None:
        if not self._backup:
            return
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self._backup:
                    p.copy_(self._backup[name])
        self._backup = {}


@contextmanager
def frozen_bn_stats(model):
    """Run a forward pass without letting it move any BatchNorm running stats.

    AWP takes its second forward pass at weights perturbed by ``delta * ||w||``.
    That pass exists only to produce a gradient — but a plain ``model.train()``
    forward also *updates* every BatchNorm's ``running_mean``/``running_var``,
    so the perturbed activations get folded into the statistics the model is
    later evaluated with. Run 1787483814 is what that looks like: after AWP
    switched on, ``stem_bn.running_var`` walked from 4.8 to 8467 while the
    stem's weight norm went 16 -> 224, and val accuracy never recovered.

    Batch statistics are still used for the normalization itself (setting the
    modules to ``eval()`` instead would change the adversarial loss), only the
    running buffers are held still: PyTorch updates them as
    ``running = (1 - momentum) * running + momentum * batch``, so ``momentum=0``
    is exactly "compute, don't remember".
    """
    saved = [(m, m.momentum) for m in model.modules()
             if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m, _ in saved:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, momentum in saved:
            m.momentum = momentum
