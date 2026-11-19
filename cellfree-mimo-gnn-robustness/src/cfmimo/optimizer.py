"""Per-scenario offline iterative reference, not a global optimum certificate."""
import torch
from .simulator import spectral_efficiency
from .gnn import soft_min_user_rate


def iterative_power(beta_obs: torch.Tensor, mask: torch.Tensor,
                    steps: int = 125, lr: float = 0.15) -> torch.Tensor:
    """Adam on AP-wise softmax logits, maximizing soft-min *observed* UatF SE.

    Iterative heuristic, not an SOCP solver and not a certified upper bound.
    Uses the same input CSI as policies, thus fair under noisy observations.
    """
    with torch.enable_grad():
        logits = torch.nn.Parameter(beta_obs.detach().clamp_min(1e-8).log().clone())
        optimizer = torch.optim.Adam([logits], lr=lr)
        valid = mask.to(beta_obs.dtype)
        for _ in range(steps):
            p = torch.softmax(logits.masked_fill(~mask, -1e6), dim=-1)*valid
            p = p / p.sum(-1,keepdim=True).clamp_min(1e-12)
            loss = -soft_min_user_rate(spectral_efficiency(beta_obs, p), temperature=0.35).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        p = torch.softmax(logits.masked_fill(~mask, -1e6), dim=-1)*valid
        return (p / p.sum(-1,keepdim=True).clamp_min(1e-12)).detach()
