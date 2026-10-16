import torch


def solve_ode(velocity, x: torch.Tensor, steps: int, method: str, pbar) -> torch.Tensor:
    # fixed-step flow ODE from t=0 (noise) to t=1, matching torchdiffeq's midpoint/euler
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        if method == "midpoint":
            x_mid = x.add(velocity(x, t), alpha=0.5 * dt)
            x = x.add(velocity(x_mid, t + 0.5 * dt), alpha=dt)
        else:
            x = x.add(velocity(x, t), alpha=dt)
        pbar.update(1)
    return x
