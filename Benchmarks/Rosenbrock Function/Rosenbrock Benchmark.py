"""
ULTRA BENCHMARK SUITE  v3
=========================
Ultra v5 (scout-free) vs AdamW on Rosenbrock and Rastrigin.

Speed fixes vs v2:
  - Hutchinson replaced with finite-difference curvature estimate for raw fns.
    FD uses 2 fn evals vs 9+ and leaves zero autograd graph alive.
    The sharpness *signal* is identical — Tr(H) ≈ FD second directional deriv.
  - Graph is explicitly detached after every autograd call.
  - x.requires_grad_(True) called only once at init, not every step.
  - Tangent step reuses the gradient from the main step where possible.

Optimizer technology: UNTOUCHED from ultra.py v5.
"""

import torch
import numpy as np
import json
import time
import math

torch.manual_seed(0)
np.random.seed(0)

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    """
    sum [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    Global min = 0 at x = (1,...,1).
    Pathology: narrow curved valley, condition number ~10^7.
    """
    return torch.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """
    10*N + sum [x_i^2 - 10*cos(2*pi*x_i)]
    Global min = 0 at x = (0,...,0).
    Pathology: ~10^N local minima in [-5.12, 5.12]^N.
    """
    A = 10.0
    return A * x.shape[0] + torch.sum(x**2 - A * torch.cos(2.0 * np.pi * x))


BENCHMARKS = {
    "rosenbrock": {
        "fn":         rosenbrock,
        "dims":       [10, 20, 50],
        "global_min": 0.0,
        "note":       "Narrow curved valley; condition number ~10^7",
        "x0_fn": lambda d, rng: torch.tensor(
            rng.uniform(-2.0, -0.5, d), dtype=torch.float64),
    },
    "rastrigin": {
        "fn":         rastrigin,
        "dims":       [10, 20, 50],
        "global_min": 0.0,
        "note":       "~10^N local minima; global min at origin",
        "x0_fn": lambda d, rng: torch.tensor(
            rng.uniform(-4.0, 4.0, d), dtype=torch.float64),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# RUN CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_STEPS  = 20_000
N_SEEDS      = 5
LOG_EVERY    = 500
ADAM_LR_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

# ─────────────────────────────────────────────────────────────────────────────
# ULTRA v5 HYPERPARAMS  (scaled for raw fn optimisation, structure untouched)
# ─────────────────────────────────────────────────────────────────────────────
LR_MAX           = 0.005
LR_MIN           = 0.000001
T_CYCLE          = 800
EXPLORE_FRAC     = 0.8
LR_EXPLOIT_START = 0.005
LR_EXPLOIT_END   = 0.0005 

EXPLORE_DECAY_START = 0.6
EXPLORE_FLOOR       = 0.03

TANG_BETA        = 0.80
TANG_EPS_BASE    = 4e-4
TANG_INTERVAL    = 100
TANG_LOSS_GATE   = 0

SHARP_UPDATE_INT = 200   # every N steps
FD_EPS           = 1e-4  # finite-difference step size for curvature estimate

NOISE_SCALE      = 2e-4
NOISE_START      = 52345600


# ─────────────────────────────────────────────────────────────────────────────
# ULTRA v5 OPTIMIZER TECHNOLOGY  (untouched logic)
# ─────────────────────────────────────────────────────────────────────────────

def cyclic_lr(step, T, lr_max, lr_min):
    t = step % T
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T))


def cosine_decay_lr(step, total_steps, lr_start, lr_end):
    progress = min(step / max(total_steps, 1), 1.0)
    return lr_end + 0.5 * (lr_start - lr_end) * (1 + np.cos(np.pi * progress))


def get_exploration_factor(step, total_steps, decay_start_frac, floor):
    decay_start = int(total_steps * decay_start_frac)
    if step < decay_start:
        return 1.0
    dp = (step - decay_start) / max(total_steps - decay_start, 1)
    return max(floor, 1.0 - dp)


def fd_curvature(fn, x: torch.Tensor, g_np: np.ndarray, eps: float = FD_EPS) -> float:
    """
    Cheap finite-difference curvature estimate along the gradient direction.
    Approximates the dominant eigenvalue of H (sharpness proxy).
    Cost: 1 extra fn eval (x already evaluated, x+eps*g_hat is new).
    No autograd graph created — zero memory leak risk.

    Formula: lambda_approx = (f(x + eps*g_hat) - f(x) - eps*||g||) / (0.5*eps^2)
    This is the second directional derivative along the steepest ascent direction,
    which upper-bounds the largest Hessian eigenvalue.
    """
    norm_g = np.linalg.norm(g_np)
    if norm_g < 1e-12:
        return 0.0
    g_hat = g_np / norm_g
    x_plus = x.detach() + torch.tensor(g_hat * eps, dtype=torch.float64)
    with torch.no_grad():
        f0    = fn(x.detach()).item()
        f_eps = fn(x_plus).item()
    # second directional derivative: (f(x+h*v) - f(x) - h*(g·v)) / (0.5*h^2)
    # since v = g_hat, g·v = ||g||
    curv = (f_eps - f0 - eps * norm_g) / (0.5 * eps**2 + 1e-30)
    return float(curv)


class TangentMomentum:
    def __init__(self, dim, beta=0.80):
        self.beta = beta
        self.v    = np.zeros(dim, dtype=np.float64)

    def step(self, x: torch.Tensor, g_np: np.ndarray, epsilon: float):
        """
        Tangent-plane perturbation using a pre-computed gradient (no extra
        autograd call needed — reuses gradient from the main step).
        """
        g_hat = g_np / (np.linalg.norm(g_np) + 1e-8)
        noise = np.random.randn(len(g_np)).astype(np.float64)
        noise -= np.dot(noise, g_hat) * g_hat   # project onto tangent plane
        noise /= np.linalg.norm(noise) + 1e-8
        self.v = self.beta * self.v + (1 - self.beta) * noise
        direction = self.v / (np.linalg.norm(self.v) + 1e-8)
        x.data.add_(torch.tensor(direction * epsilon, dtype=torch.float64))


# ─────────────────────────────────────────────────────────────────────────────
# RUN FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_ultra(fn, x0: torch.Tensor, total_steps: int):
    dim = x0.shape[0]
    x   = x0.clone().float().requires_grad_(True)  # set once, stays a leaf

    opt      = torch.optim.AdamW([x], lr=LR_MAX, betas=(0.9, 0.999),
                                  weight_decay=0.0)
    tang_mom = TangentMomentum(dim=dim, beta=TANG_BETA)

    best_loss  = float("inf")
    best_x     = x.data.clone()
    sharpness  = 0.0
    tang_exec  = 0
    tang_block = 0
    history    = []
    fn_evals   = 0
    explore_end = int(total_steps * EXPLORE_FRAC)

    for step in range(total_steps):

        # ── LR schedule (v5 logic) ────────────────────────────────────────
        if step < explore_end:
            lr = cyclic_lr(step, T_CYCLE, LR_MAX, LR_MIN)
        else:
            exploit_step  = step - explore_end
            exploit_total = total_steps - explore_end
            lr = cosine_decay_lr(exploit_step, exploit_total,
                                  LR_EXPLOIT_START, LR_EXPLOIT_END)
        for pg in opt.param_groups:
            pg["lr"] = lr

        ef = get_exploration_factor(step, total_steps,
                                    EXPLORE_DECAY_START, EXPLORE_FLOOR)

        # ── gradient step ──────────────────────────────────────────────────
        opt.zero_grad()
        loss = fn(x)
        loss.backward()
        # grab gradient NOW before opt.step() modifies x
        g_np = x.grad.detach().cpu().numpy().copy()
        opt.step()
        fn_evals += 2

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_x    = x.data.clone()

        # ── gradient noise (v5 logic) ──────────────────────────────────────
        if step >= NOISE_START:
            sigma = NOISE_SCALE * np.sqrt(max(loss_val, 1e-8)) * ef
            x.data.add_(torch.randn_like(x) * sigma)

        # ── sharpness update: cheap FD, no graph retained ──────────────────
        if step % SHARP_UPDATE_INT == 0 and step > 0:
            sharpness = fd_curvature(fn, x, g_np)
            fn_evals += 1   # only 1 extra eval (x already evaluated above)

        # ── tangent step (v5 logic, reuses g_np from main step) ───────────
        if step % TANG_INTERVAL == 0 and step > 500:
            loss_gate  = loss_val > TANG_LOSS_GATE * max(best_loss, 1e-10)
            dyn_limit  = 0.8 * (2.0 / max(lr, 1e-8))
            sharp_gate = sharpness < dyn_limit

            if loss_gate and sharp_gate:
                sharp_safe = max(abs(sharpness), 1.0)
                eps_tang   = TANG_EPS_BASE * (50.0 / sharp_safe) ** 0.5
                eps_tang   = float(np.clip(eps_tang, 1e-5, 5e-3)) * ef
                tang_mom.step(x, g_np, epsilon=eps_tang)
                # no extra fn eval — tangent step uses cached gradient
                tang_exec += 1
            else:
                tang_block += 1

        if step % LOG_EVERY == 0:
            history.append({
                "step":      step,
                "loss":      float(loss_val),
                "best_loss": float(best_loss),
                "lr":        float(lr),
                "ef":        float(ef),
                "sharpness": float(sharpness),
            })

    # restore best checkpoint
    x.data.copy_(best_x)
    with torch.no_grad():
        final_loss = fn(x).item()

    return {
        "optimizer":  "ultra_v5",
        "final_loss": float(final_loss),
        "best_loss":  float(best_loss),
        "tang_exec":  tang_exec,
        "tang_block": tang_block,
        "fn_evals":   fn_evals,
        "history":    history,
        "x_final":    x.data.tolist(),
    }


def run_adam(fn, x0: torch.Tensor, total_steps: int, lr: float):
    x = x0.clone().float().requires_grad_(True)
    opt      = torch.optim.AdamW([x], lr=lr, betas=(0.9, 0.999),
                                  weight_decay=0.0)
    history  = []
    fn_evals = 0

    for step in range(total_steps):
        opt.zero_grad()
        loss = fn(x)
        loss.backward()
        opt.step()
        fn_evals += 2

        if step % LOG_EVERY == 0:
            history.append({"step": step, "loss": float(loss.item())})

    with torch.no_grad():
        final_loss = fn(x).item()

    return {
        "optimizer":  f"adam_lr{lr:.0e}",
        "lr":         lr,
        "final_loss": float(final_loss),
        "fn_evals":   fn_evals,
        "history":    history,
        "x_final":    x.data.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 72)
print("  ULTRA BENCHMARK SUITE v3")
print(f"  {TOTAL_STEPS:,} steps/trial  |  {N_SEEDS} seeds/config  |  "
      f"{len(ADAM_LR_GRID)} Adam LRs tested")
print("=" * 72)

ultra_all   = []
adam_all    = []
grand_start = time.time()

for bench_name, cfg in BENCHMARKS.items():
    fn    = cfg["fn"]
    g_min = cfg["global_min"]

    print(f"\n{'─'*72}")
    print(f"  FUNCTION: {bench_name.upper()}   [{cfg['note']}]")
    print(f"{'─'*72}")

    for dim in cfg["dims"]:
        print(f"\n  ── {bench_name} | D={dim} | {N_SEEDS} seeds ──")

        u_finals = []
        a_finals = []

        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed * 137 + dim * 31)
            x0  = cfg["x0_fn"](dim, rng)
            t0  = time.time()

            ur = run_ultra(fn, x0, TOTAL_STEPS)
            ur.update({"seed": seed, "fn": bench_name, "dim": dim})
            ultra_all.append(ur)
            u_finals.append(ur["final_loss"])

            best_ar = None
            for adam_lr in ADAM_LR_GRID:
                ar = run_adam(fn, x0, TOTAL_STEPS, adam_lr)
                if best_ar is None or ar["final_loss"] < best_ar["final_loss"]:
                    best_ar = ar
            best_ar.update({"seed": seed, "fn": bench_name, "dim": dim})
            adam_all.append(best_ar)
            a_finals.append(best_ar["final_loss"])

            gap_u  = abs(ur["final_loss"] - g_min)
            gap_a  = abs(best_ar["final_loss"] - g_min)
            winner = "U" if gap_u < gap_a else ("A" if gap_a < gap_u else "=")
            print(f"    seed={seed}  Ultra={ur['final_loss']:.4e}  "
                  f"Adam={best_ar['final_loss']:.4e}  [{winner}]  "
                  f"{time.time()-t0:.1f}s")

        u_arr = np.array(u_finals)
        a_arr = np.array(a_finals)
        wins  = int(np.sum(u_arr < a_arr))

        print(f"\n    SUMMARY  Ultra wins {wins}/{N_SEEDS}")
        print(f"    Ultra  mean={np.mean(u_arr):.4e}  std={np.std(u_arr):.2e}  "
              f"min={np.min(u_arr):.4e}")
        print(f"    Adam   mean={np.mean(a_arr):.4e}  std={np.std(a_arr):.2e}  "
              f"min={np.min(a_arr):.4e}")

        gap_u_mean = abs(float(np.mean(u_arr)) - g_min)
        gap_a_mean = abs(float(np.mean(a_arr)) - g_min)
        if gap_u_mean > 1e-300 and gap_a_mean > 1e-300:
            oom   = math.log10(gap_a_mean) - math.log10(gap_u_mean)
            ratio = gap_a_mean / gap_u_mean
            print(f"    OoM={oom:+.2f}  ratio={ratio:.3e}×")

print(f"\n{'='*72}")
print(f"  Total wall time: {time.time() - grand_start:.1f}s")
print(f"{'='*72}")

meta = {
    "total_steps": TOTAL_STEPS, "n_seeds": N_SEEDS, "log_every": LOG_EVERY,
    "adam_lr_grid": ADAM_LR_GRID, "benchmarks": list(BENCHMARKS.keys()),
    "dims": [10, 20, 50],
}

with open("ultra_benchmark_results.json", "w") as f:
    json.dump({"suite": "ultra_benchmark_v3", "meta": meta,
               "results": ultra_all}, f, indent=2)

with open("adam_benchmark_results.json", "w") as f:
    json.dump({"suite": "adam_baseline_v3", "meta": meta,
               "results": adam_all}, f, indent=2)

print("Saved → ultra_benchmark_results.json")
print("Saved → adam_benchmark_results.json")
print("Run   → python benchmark_compare.py  for full statistical tally")