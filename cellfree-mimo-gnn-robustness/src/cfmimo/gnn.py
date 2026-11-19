"""Bipartite AP–UE message-passing GNN with hard per-AP power feasibility."""
from __future__ import annotations
import torch
from torch import nn
from .simulator import channel_quality, spectral_efficiency


def mlp(inp: int, out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(),
                         nn.Linear(hidden, out), nn.ReLU())


class PowerGNN(nn.Module):
    """Permutation equivariant to relabeling APs and UEs.

    Each layer computes edge messages, aggregates via degree-normalized
    pooling at AP and UE nodes, and updates both node types. Each AP uses
    masked softmax over its served UEs; an isolated AP transmits zero.
    """
    def __init__(self, hidden: int = 32, rounds: int = 2):
        super().__init__()
        self.hidden = hidden
        self.rounds = rounds
        self.edge_init = mlp(2, hidden, hidden)
        self.ap_init = mlp(2, hidden, hidden)
        self.ue_init = mlp(2, hidden, hidden)
        self.edge_update = nn.ModuleList([mlp(3*hidden, hidden, hidden) for _ in range(rounds)])
        self.ap_update = nn.ModuleList([mlp(2*hidden, hidden, hidden) for _ in range(rounds)])
        self.ue_update = nn.ModuleList([mlp(2*hidden, hidden, hidden) for _ in range(rounds)])
        self.output = nn.Sequential(nn.Linear(3*hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, beta_obs: torch.Tensor, mask: torch.Tensor, p_ap: float = 1.) -> torch.Tensor:
        b, m, k = beta_obs.shape
        # Shared physical log-gain scaling; no per-topology normalization.
        beta = beta_obs.clamp(1e-8, 1.0)
        gamma = channel_quality(beta, tau_p=k)
        features = torch.stack([torch.log10(beta), (gamma / beta).clamp(0, 1)], dim=-1)
        valid = mask.to(features.dtype)
        ap_deg = valid.sum(dim=2, keepdim=True).clamp(min=1)
        ue_deg = valid.sum(dim=1).unsqueeze(-1).clamp(min=1)
        observed = features * valid.unsqueeze(-1)
        ap = self.ap_init(observed.sum(dim=2) / ap_deg)
        ue = self.ue_init(observed.sum(dim=1) / ue_deg)
        edge = self.edge_init(features) * valid.unsqueeze(-1)
        for t in range(self.rounds):
            ap_b = ap.unsqueeze(2).expand(-1, -1, k, -1)
            ue_b = ue.unsqueeze(1).expand(-1, m, -1, -1)
            msg = self.edge_update[t](torch.cat((edge, ap_b, ue_b), dim=-1)) * valid.unsqueeze(-1)
            ap_agg = msg.sum(dim=2) / ap_deg
            ue_agg = msg.sum(dim=1) / ue_deg
            ap = self.ap_update[t](torch.cat((ap, ap_agg), dim=-1))
            ue = self.ue_update[t](torch.cat((ue, ue_agg), dim=-1))
            edge = msg
        ap_b = ap.unsqueeze(2).expand(-1, -1, k, -1)
        ue_b = ue.unsqueeze(1).expand(-1, m, -1, -1)
        logits = self.output(torch.cat((edge, ap_b, ue_b), dim=-1)).squeeze(-1)
        # Masked softmax (including APs with no active edges); no NaN.
        logits = logits.masked_fill(~mask, -1e6)
        weights = torch.softmax(logits, dim=-1) * valid
        return p_ap * weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def soft_min_user_rate(se: torch.Tensor, temperature: float = 0.35) -> torch.Tensor:
    """Differentiable conservative approximation of worst-user SE."""
    return -temperature * torch.logsumexp(-se / temperature, dim=-1)


def fgsm_csi(model: PowerGNN, beta_true: torch.Tensor, mask: torch.Tensor,
             epsilon: float = 0.3) -> torch.Tensor:
    """White-box feature-space log-beta FGSM, targeting true worst-user SE.

    This is NOT an over-the-air attack and assumes attacker can change the
    controller's reported large-scale fading within bounded log error.
    """
    with torch.enable_grad():
        log_obs = beta_true.detach().clamp(1e-8,1.).log().requires_grad_(True)
        p = model(log_obs.exp(), mask)
        se = spectral_efficiency(beta_true, p)
        objective = soft_min_user_rate(se).mean()
        gradient = torch.autograd.grad(objective, log_obs)[0]
        # Gradient descent on objective, hence the negative sign.
        log_attack = log_obs.detach() - epsilon * gradient.sign()
    return log_attack.exp().clamp(1e-8, 1.0)
