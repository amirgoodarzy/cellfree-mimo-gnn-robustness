"""Train and benchmark the exploratory AP-UE GNN. Uses CPU and deterministic seeds.

Example: python -m cfmimo.experiment --epochs 80 --out results
Run from repo root after `pip install -e .`.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .simulator import (make_networks, association, uniform_power, large_scale_power,
                        spectral_efficiency, outage_mask, random_csi)
from .gnn import PowerGNN, soft_min_user_rate, fgsm_csi
from .optimizer import iterative_power


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def rates(model, beta, mask, observed=None):
    obs = beta if observed is None else observed
    out = {}
    iterative = iterative_power(obs, mask)
    with torch.no_grad():
        for name, power in [('uniform', uniform_power(mask)),
                            ('gain_proportional', large_scale_power(obs, mask)),
                            ('bipartite_gnn', model(obs, mask)),
                            ('iterative_reference', iterative)]:
            assert power.min().item() >= 0
            assert power.sum(-1).max().item() <= 1.00001
            se = spectral_efficiency(beta, power)
            out[name] = (se.min(-1).values.cpu().numpy(), se.mean(-1).cpu().numpy())
    return out


def ci95_mean(x: np.ndarray, bootstrap_seed=777):
    rng = np.random.default_rng(bootstrap_seed)
    n = len(x)
    means = x[rng.integers(0, n, size=(2000, n))].mean(-1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--train', type=int, default=480)
    parser.add_argument('--test', type=int, default=192)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    m, k = 16, 4
    beta_train = make_networks(args.train, m=m, k=k, seed=args.seed+1)
    beta_val = make_networks(96, m=m, k=k, seed=args.seed+2)
    beta_test = make_networks(args.test, m=m, k=k, seed=args.seed+3)
    mask_train = association(beta_train, aps_per_ue=8)
    mask_val = association(beta_val, aps_per_ue=8)
    mask_test = association(beta_test, aps_per_ue=8)
    model = PowerGNN(hidden=32, rounds=2)
    optim = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    best_loss, best_epoch, best_state = float('inf'), -1, None
    history = []
    begin = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        shuf = torch.randperm(args.train)
        batch_losses = []
        for ids in shuf.split(48):
            p = model(beta_train[ids], mask_train[ids])
            se = spectral_efficiency(beta_train[ids], p)
            loss = -soft_min_user_rate(se, temperature=0.35).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            batch_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_p = model(beta_val, mask_val)
            val_loss = float(-soft_min_user_rate(spectral_efficiency(beta_val,val_p), temperature=0.35).mean().item())
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(batch_losses)), 'validation_loss': val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    train_seconds = time.perf_counter() - begin
    torch.save({'state_dict': best_state,
                'model': 'PowerGNN(hidden=32,rounds=2)',
                'training_seed': args.seed,
                'best_epoch': best_epoch}, root / 'model_state.pt')
    cases = {
        'clean': (mask_test, beta_test),
        '25pct_ap_outage': (outage_mask(mask_test, 0.25, seed=args.seed+9), beta_test),
        'bounded_random_csi_error': (mask_test, random_csi(beta_test, epsilon=0.3, seed=args.seed+10)),
    }
    # Attacker edits controller-side log large-scale fading, not physical propagation.
    cases['whitebox_fgsm_csi_error'] = (mask_test, fgsm_csi(model, beta_test, mask_test, epsilon=0.3).detach())
    records = []
    summaries = {}
    for case, (mask, observed) in cases.items():
        values = rates(model, beta_test, mask, observed)
        summaries[case] = {}
        for method, (min_rate, mean_rate) in values.items():
            summaries[case][method] = {
                'mean_worst_user_se': float(min_rate.mean()),
                'ci95_mean_worst_user_se': ci95_mean(min_rate),
                'median_worst_user_se': float(np.median(min_rate)),
                'mean_user_se': float(mean_rate.mean()),
            }
            for i in range(args.test):
                records.append({'scenario': i, 'condition': case, 'method': method,
                                'worst_user_se': float(min_rate[i]),
                                'mean_user_se': float(mean_rate[i])})
    with (root/'per_scenario.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['scenario','condition','method','worst_user_se','mean_user_se'])
        w.writeheader(); w.writerows(records)
    with (root/'training_history.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['epoch','train_loss','validation_loss']); w.writeheader(); w.writerows(history)
    info={'title':'CF-mMIMO miniature reproducible pilot study',
          'seed':args.seed, 'train_scenarios':args.train, 'validation_scenarios':96,
          'test_scenarios':args.test, 'AP_count':m, 'UE_count':k,
          'nearest_APs_per_UE':8, 'prelog':0.98,
          'power_per_AP':1.0, 'noise_power':0.001,
          'area_m':500, 'pathloss_exponent':3.2,
          'orthogonal_pilots':True, 'pilot_length':4, 'coherence_block':200,
          'CSI_log_error_epsilon':0.3, 'AP_failure_rate':0.25,
          'best_validation_epoch':best_epoch, 'train_seconds':train_seconds,
          'experiment_scope':'synthetic toy-scale networks only, no physical 3GPP calibration',
          'comparisons':summaries}
    (root/'metrics.json').write_text(json.dumps(info,indent=2))
    print(f'best_epoch={best_epoch} train_seconds={train_seconds:.1f}')
    for case, values in summaries.items():
        print('\n'+case)
        for method, v in values.items():
            print(f"  {method:19s} worst-SE mean={v['mean_worst_user_se']:.4f} CI95={v['ci95_mean_worst_user_se']} mean-SE={v['mean_user_se']:.4f}")

if __name__ == '__main__':
    main()
