import torch
from cfmimo.simulator import (make_networks, association, channel_quality, uniform_power,
                              large_scale_power, spectral_efficiency, outage_mask)
from cfmimo.gnn import PowerGNN


def test_gamma_is_in_physical_bounds():
    beta = make_networks(8)
    gamma = channel_quality(beta)
    assert torch.all(gamma >= 0) and torch.all(gamma <= beta + 1e-6)


def test_uatf_rates_are_positive_and_finite():
    beta = make_networks(10)
    p = uniform_power(association(beta))
    se = spectral_efficiency(beta, p)
    assert se.shape == (10, 4)
    assert torch.isfinite(se).all() and (se >= 0).all()


def test_per_ap_constraints_and_outage():
    beta = make_networks(10)
    mask = association(beta)
    broken = outage_mask(mask, fraction=0.25)
    for p in [uniform_power(broken), large_scale_power(beta, broken),
              PowerGNN(hidden=16)(beta, broken)]:
        assert (p >= 0).all()
        assert (p.sum(dim=-1) <= 1.000001).all()
        assert torch.all(p[~broken] == 0)


def test_mask_all_off_is_safe():
    beta = make_networks(2)
    off = torch.zeros_like(beta, dtype=torch.bool)
    assert torch.all(PowerGNN(hidden=16)(beta, off) == 0)


def test_noise_reduction_improves_se_with_fixed_gamma_reference():
    beta = make_networks(6)
    p = uniform_power(association(beta))
    # We recompute channel estimation at each noise level. Higher noise hurts both.
    assert (spectral_efficiency(beta, p, noise=0.0001) >=
            spectral_efficiency(beta, p, noise=0.01)).all()


def test_gnn_permutation_equivariance():
    torch.manual_seed(8)
    beta = make_networks(3)
    mask = association(beta)
    model = PowerGNN(hidden=16).eval()
    p = model(beta, mask)
    ai = torch.randperm(beta.shape[1]); ui = torch.randperm(beta.shape[2])
    pb = model(beta[:, ai][:, :, ui], mask[:, ai][:, :, ui])
    assert torch.allclose(pb, p[:, ai][:, :, ui], atol=1e-6)


def test_iterative_reference_respects_constraints():
    from cfmimo.optimizer import iterative_power
    beta = make_networks(3)
    mask = association(beta)
    p = iterative_power(beta, mask, steps=3)
    assert torch.isfinite(p).all() and torch.all(p.sum(-1) <= 1.000001)
    assert torch.all(p[~mask] == 0)


def test_fgsm_log_gain_attack_is_bounded():
    from cfmimo.gnn import fgsm_csi
    beta = make_networks(3)
    mask = association(beta)
    model = PowerGNN(hidden=16)
    attacked = fgsm_csi(model, beta, mask, epsilon=0.3)
    deviation = (attacked.clamp_min(1e-8).log() - beta.log()).abs()
    assert deviation.max() <= 0.300001
