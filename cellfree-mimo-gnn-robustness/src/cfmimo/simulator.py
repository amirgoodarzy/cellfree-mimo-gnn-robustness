"""Downlink single-antenna CF-mMIMO with orthogonal pilots and MR precoding.

This is a simplified use-and-then-forget (UatF) ergodic lower-bound model.
Noise, transmit power, and large-scale gains are dimensionless normalized values.
No pilot contamination, spatial correlation, hardware effects, or fronthaul delay.
"""
from __future__ import annotations
import numpy as np
import torch


def make_networks(n: int, m: int = 16, k: int = 4, seed: int = 1,
                  area_m: float = 500.0, dref_m: float = 50.0,
                  pathloss_exp: float = 3.2) -> torch.Tensor:
    """Independent AP/UE layouts; returns normalized large-scale gains [N,M,K]."""
    rng = np.random.default_rng(seed)
    aps = rng.uniform(0, area_m, size=(n, m, 2))
    ues = rng.uniform(0, area_m, size=(n, k, 2))
    d = np.linalg.norm(aps[:, :, None, :] - ues[:, None, :, :], axis=-1)
    d = np.maximum(d, 10.0)
    beta = np.minimum((dref_m / d) ** pathloss_exp, 1.0)
    return torch.tensor(beta, dtype=torch.float32)


def association(beta: torch.Tensor, aps_per_ue: int = 8) -> torch.Tensor:
    """Each UE is connected to its L strongest APs (fixed for all policies)."""
    b, m, k = beta.shape
    l = min(aps_per_ue, m)
    idx = torch.topk(beta, k=l, dim=1).indices
    mask = torch.zeros_like(beta, dtype=torch.bool)
    return mask.scatter_(1, idx, True)


def channel_quality(beta: torch.Tensor, tau_p: int = 4,
                    pilot_power: float = 1.0, noise: float = 0.001) -> torch.Tensor:
    """MMSE estimate variance gamma=τp*pp*β²/(τp*pp*β+noise).

    Valid for independent Rayleigh channels and mutually orthogonal uplink pilots
    (tau_p >= K); pilot contamination is excluded by assumption.
    """
    return tau_p * pilot_power * beta.square() / (tau_p * pilot_power * beta + noise)


def power_from_scores(scores: torch.Tensor, mask: torch.Tensor,
                      p_ap: float = 1.0) -> torch.Tensor:
    """Nonnegative per-AP allocation with <=p_ap, including all-off APs."""
    valid = mask.to(scores.dtype)
    weighted = scores.clamp_min(0.0) * valid
    return p_ap * weighted / weighted.sum(-1, keepdim=True).clamp_min(1e-12)


def uniform_power(mask: torch.Tensor, p_ap: float = 1.0) -> torch.Tensor:
    return power_from_scores(torch.ones_like(mask, dtype=torch.float32), mask, p_ap)


def large_scale_power(beta_obs: torch.Tensor, mask: torch.Tensor,
                      p_ap: float = 1.0) -> torch.Tensor:
    return power_from_scores(beta_obs, mask, p_ap)


def spectral_efficiency(beta_true: torch.Tensor, power: torch.Tensor,
                        tau_p: int = 4, tau_c: int = 200,
                        pilot_power: float = 1.0, noise: float = 0.001) -> torch.Tensor:
    """Ergodic downlink UatF SE [batch,K], bits/s/Hz.

    With independent Rayleigh channels, orthogonal pilots and normalized MR:
    SINR_k=(Σ_m sqrt(p_mk gamma_mk))² /
           (Σ_m beta_mk Σ_j p_mj + noise).
    p_mk are nonnegative transmitted *power* fractions, not amplitudes.
    """
    if tau_p < beta_true.shape[-1] or tau_p >= tau_c:
        raise ValueError("Orthogonal-pilot example requires K <= tau_p < tau_c")
    gamma = channel_quality(beta_true, tau_p, pilot_power, noise)
    amplitude = torch.where(power > 0, (power.clamp_min(0) * gamma).clamp_min(1e-12).sqrt(), torch.zeros_like(power))
    desired = amplitude.sum(dim=1).square()
    interference = (beta_true * power.sum(dim=-1, keepdim=True)).sum(dim=1)
    sinr = desired / (interference + noise)
    return (1.0 - tau_p / tau_c) * torch.log2(1.0 + sinr)


def outage_mask(mask: torch.Tensor, fraction: float = 0.25,
                seed: int = 19) -> torch.Tensor:
    """Remove a reproducibly chosen fraction of APs per network."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    b, m, _ = mask.shape
    n_remove = round(m * fraction)
    rng = np.random.default_rng(seed)
    active = np.ones((b, m), dtype=bool)
    for i in range(b):
        if n_remove:
            active[i, rng.choice(m, size=n_remove, replace=False)] = False
    return mask & torch.as_tensor(active, device=mask.device)[:, :, None]


def random_csi(beta_true: torch.Tensor, epsilon: float = 0.3,
              seed: int = 23) -> torch.Tensor:
    """Bounded multiplicative corruption: log(beta_obs/beta_true) in [-eps,eps]."""
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-epsilon, epsilon, tuple(beta_true.shape))
    return (beta_true * torch.exp(torch.as_tensor(delta, dtype=beta_true.dtype))).clamp(1e-8, 1.0)


def worst_user_rate(beta_true: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    return spectral_efficiency(beta_true, power).min(dim=-1).values
