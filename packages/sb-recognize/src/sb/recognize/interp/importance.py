"""Per-landmark importance/ranking for a trained DNN/LSTM/GRU (TODO §3).

Three metrics, deliberately kept simple and cheap enough to run for all 543
landmarks x 3 architectures without needing a subsample:

1. :func:`attention_weights` — the learned :class:`~sb.recognize.interp.
   models.LandmarkAttention` gate itself. Free (no forward pass), structural,
   and — because the same gate module sits in front of the DNN, LSTM and GRU
   — the one metric that is directly comparable *across* architectures without
   needing to reverse-engineer each one's internal weight shapes.
2. :func:`gradient_saliency` — mean |gradient x input| of the true-class score
   w.r.t. each landmark's channel block, over a validation sample. Behavioral
   (depends on what the trained trunk actually does with the gated features),
   and, for the RNNs, integrates contributions from every timestep the
   recurrence carries a landmark's information through, not just the last one.
3. :func:`permutation_importance` — replace one landmark's channel block
   (across the whole sequence) with another sample's, holding everything else
   fixed, and measure the video-accuracy drop. Most faithful to "how much does
   the model's prediction actually depend on this landmark", most expensive
   (one forward pass over the sample per landmark).

:func:`combine_ranking` folds the three into one ranking score by averaging
each metric's normalized rank (0 = least important, 1 = most) — rank-averaging
rather than averaging raw values because the three are on incomparable scales
(a sigmoid gate in (0,1), a saliency magnitude, an accuracy delta).
"""

import numpy as np
import pandas as pd
import torch

from sb.core.schema import GROUPS
from sb.recognize.interp.features import CHANNELS_PER_LANDMARK, PER_LANDMARK_DIM
from sb.recognize.interp.models import N_LANDMARKS
from sb.recognize.interp.train import _frame_mask

LANDMARK_REGION = np.empty(N_LANDMARKS, dtype=object)
for _g in GROUPS:
    LANDMARK_REGION[_g.slice] = _g.name


def attention_weights(model) -> np.ndarray:
    """(543,) sigmoid gate values, straight off the trained module."""
    return model.attn.gate().detach().cpu().numpy()


def gradient_saliency(model, arch: str, feats: torch.Tensor, lengths: torch.Tensor,
                       labels: torch.Tensor, device) -> np.ndarray:
    """Mean |grad x input| per landmark over one already-batched, already
    length-sorted (descending) sample — the same sample every metric and every
    architecture should use so the three rankings are comparable.
    """
    model.eval()
    feats = feats.to(device).clone().requires_grad_(True)
    labels = labels.to(device)
    # cuDNN's fused RNN kernel only supports backward() in training mode; since
    # eval mode is required here (to keep dropout off for a faithful saliency
    # signal), fall back to PyTorch's native RNN backward for this computation.
    with torch.backends.cudnn.flags(enabled=False):
        if arch == "dnn":
            B, T, _ = feats.shape
            mask = _frame_mask(lengths, T, device)
            logits = model(feats)
            sel = logits.gather(-1, labels.view(-1, 1, 1).expand(-1, T, 1)).squeeze(-1)
            score = (sel * mask).sum()
        else:
            logits = model(feats, lengths)
            sel = logits.gather(-1, labels.view(-1, 1)).squeeze(-1)
            score = sel.sum()
        (grad,) = torch.autograd.grad(score, feats)
    sal = (grad * feats).detach()
    per_landmark = sal[..., :PER_LANDMARK_DIM].reshape(
        *sal.shape[:-1], N_LANDMARKS, CHANNELS_PER_LANDMARK
    ).abs().sum(-1)  # (B, T, 543)
    if arch == "dnn":
        per_landmark = per_landmark * mask.unsqueeze(-1)
        denom = mask.sum().item()
    else:
        denom = per_landmark.shape[0] * per_landmark.shape[1]
    return (per_landmark.sum(dim=(0, 1)).cpu().numpy() / max(denom, 1))


@torch.no_grad()
def _video_preds(model, arch: str, feats: torch.Tensor, lengths: torch.Tensor, device) -> torch.Tensor:
    feats = feats.to(device)
    if arch == "dnn":
        B, T, _ = feats.shape
        mask = _frame_mask(lengths, T, device)
        logits = model(feats)
        probs = torch.softmax(logits, dim=-1) * mask.unsqueeze(-1)
        return (probs.sum(1) / mask.sum(1, keepdim=True)).argmax(-1).cpu()
    logits = model(feats, lengths)
    return logits.argmax(-1).cpu()


@torch.no_grad()
def permutation_importance(
    model, arch: str, feats: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor,
    device, seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Baseline video-accuracy + (543,) accuracy DROP per landmark on shuffling
    that landmark's channel block across the batch (whole-sequence swap, not
    per-frame — a sample's own padding can bleed into a shorter sample's
    valid frames when lengths differ, which is a known approximation, not a
    bug: it behaves like a mix of permutation and ablation for the tail of a
    shorter sequence).
    """
    rng = np.random.default_rng(seed)
    labels_cpu = labels.cpu()
    base_preds = _video_preds(model, arch, feats, lengths, device)
    baseline_acc = (base_preds == labels_cpu).float().mean().item()
    drops = np.zeros(N_LANDMARKS, dtype=np.float64)
    B = feats.shape[0]
    for li in range(N_LANDMARKS):
        perm = torch.from_numpy(rng.permutation(B))
        start, end = li * CHANNELS_PER_LANDMARK, (li + 1) * CHANNELS_PER_LANDMARK
        perturbed = feats.clone()
        perturbed[..., start:end] = feats[perm][..., start:end]
        preds = _video_preds(model, arch, perturbed, lengths, device)
        acc = (preds == labels_cpu).float().mean().item()
        drops[li] = baseline_acc - acc
    return baseline_acc, drops


def _normalized_rank(values: np.ndarray) -> np.ndarray:
    """(N,) values -> (N,) rank in [0, 1], 1 = most important (highest value)."""
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values))
    return ranks / max(len(values) - 1, 1)


def combine_ranking(metrics: dict[str, np.ndarray]) -> pd.DataFrame:
    """``{metric_name: (543,) raw values}`` -> one DataFrame, one row per
    landmark, with each metric's raw value, its normalized rank, region, and
    the averaged ``ranking_score`` (mean of the normalized ranks)."""
    df = pd.DataFrame({"landmark": np.arange(N_LANDMARKS), "region": LANDMARK_REGION})
    rank_cols = []
    for name, values in metrics.items():
        df[name] = values
        rank_col = f"{name}_rank"
        df[rank_col] = _normalized_rank(np.asarray(values))
        rank_cols.append(rank_col)
    df["ranking_score"] = df[rank_cols].mean(axis=1)
    return df.sort_values("ranking_score", ascending=False).reset_index(drop=True)


def region_summary(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ``ranking_score`` (+ each raw metric) per region — the region-level
    heatmap input (Face / Pose / Left hand / Right hand)."""
    value_cols = [c for c in ranking_df.columns if c not in ("landmark", "region")]
    return (
        ranking_df.groupby("region")[value_cols]
        .mean()
        .sort_values("ranking_score", ascending=False)
    )
