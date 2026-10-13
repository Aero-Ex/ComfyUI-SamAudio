from typing import Callable, Optional
import torch
import comfy.utils


def solve_ode(
    vector_field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    steps: int = 16,
    method: str = "midpoint",
    callback: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """Solves the probability flow ODE from t=0.0 to t=1.0.

    Args:
        vector_field_fn: Callable(x, t) -> dx/dt
        initial_state: Starting tensor at t=0 (e.g. noise) [B, T, C]
        steps: Number of integration steps (default 16, matching step_size=2/32)
        method: 'midpoint' (2nd order Runge-Kutta) or 'euler'
        callback: Optional progress callback(step, total_steps)
    """
    dt = 1.0 / steps
    y = initial_state
    b = y.shape[0]

    pbar = comfy.utils.ProgressBar(steps)

    for i in range(steps):
        t_val = i * dt
        t = torch.full((b,), t_val, device=y.device, dtype=y.dtype)

        if method == "midpoint":
            k1 = vector_field_fn(y, t)
            t_mid_val = t_val + 0.5 * dt
            t_mid = torch.full((b,), t_mid_val, device=y.device, dtype=y.dtype)
            y_mid = y + (0.5 * dt) * k1
            k2 = vector_field_fn(y_mid, t_mid)
            y = y + dt * k2
        elif method == "euler":
            k1 = vector_field_fn(y, t)
            y = y + dt * k1
        else:
            raise ValueError(f"Unknown ODE integration method: {method}")

        pbar.update(1)
        if callback is not None:
            callback(i + 1, steps)

    return y
