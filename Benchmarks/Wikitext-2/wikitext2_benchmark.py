"""
TANGO — WikiText-2 NanoGPT Benchmark
======================================
Three-way comparison on WikiText-2 char-level NanoGPT:

  (A) Tango-Full   — cyclic LR explore phase + ef decay + cosine exploit
                     + trust-region tangent steps
  (B) Tango-NoEF   — cosine LR only throughout, ef pinned to 1.0
                     (isolates tangent momentum contribution alone)
  (C) AdamW-Best   — AdamW grid search over ADAM_LR_GRID, best result reported

Architecture:  direct port of the Rosenbrock ablation design to NanoGPT.
Key differences from Ultra_wikitext2_benchmark.ipynb:
  - EXPLORE_FRAC = 0.30 (not 0.0)  →  genuine cyclic explore phase
  - Trust-region accept/reject on every tangent step  (Eq. 12-15)
  - ef actually decays  (EF_DECAY_START=0.60, EF_FLOOR=0.03)
  - Noise removed entirely (no noise in Rosenbrock either; tangent handles exploration)
  - LR_EXPLOIT_END = 1e-5  (was 3e-5; tighter final convergence)
  - tang warmup guard lowered from 500 → 300

Porting notes from Rosenbrock → NanoGPT:
  - fd_curvature uses a mini-batch forward pass (not the full dataset)
  - TangentMomentum.step() applies the delta directly to model parameters
    (Rosenbrock worked on a 1-D tensor; here we work on the flat param vector)
  - Trust-region evaluation uses the same mini-batch xb,yb as the grad step
    to keep cost at 2 extra forward passes per tangent attempt
"""

# ── Cell 1: Imports & device ──────────────────────────────────────────────────
import copy, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print('⚠️  No GPU — running on CPU. Go to Runtime → Change runtime type → T4 GPU')


# ── Cell 2: Config ────────────────────────────────────────────────────────────
TOTAL_STEPS  = 5000
N_SEEDS      = 4
BATCH_SIZE   = 64
BLOCK_SIZE   = 256
EVAL_ITERS   = 50
LOG_EVERY    = 100
DATA_DIR     = './data'

# Model — identical to original WikiText-2 benchmark
N_LAYER  = 6
N_HEAD   = 6
N_EMBD   = 384
DROPOUT  = 0.2

# AdamW baselines
ADAM_LR_GRID = [3e-4, 1e-4]

# ── Tango hyperparameters ─────────────────────────────────────────────────────
# Learning rate schedule
LR_MAX            = 3e-4        # cyclic peak (also AdamW base LR)
LR_MIN            = 1e-5        # cyclic trough
T_CYCLE           = 500         # cycle length  →  ~3 full cycles in explore phase
EXPLORE_FRAC      = 0.30        # first 30% = 1500 steps of cyclic exploration
LR_EXPLOIT_START  = 3e-4        # cosine start for exploit
LR_EXPLOIT_END    = 1e-5        # cosine end   (tight final convergence)

# Tango-NoEF cosine schedule (full run, no cyclic phase)
LR_COSINE_START   = 3e-4
LR_COSINE_END     = 1e-5

# Exploration factor φ(t)
EF_DECAY_START    = 0.60        # φ stays 1.0 until 60% of total steps
EF_FLOOR          = 0.03        # φ decays to 3% minimum

# Curvature probe
SHARP_UPDATE_INT  = 200         # recompute curvature every N steps
FD_EPS            = 1e-4        # finite-difference step size

# Tangent momentum
TANG_BETA         = 0.80        # momentum coefficient
TANG_EPS_BASE     = 4e-4        # ε₀ base tangent step size
TANG_SIGMA        = 50.0        # normalisation constant σ
TANG_INTERVAL     = 100         # steps between tangent attempts
TANG_WARMUP       = 300         # steps before first tangent attempt

# Trust-region (Eq. 12-15 from paper)
TR_DELTA_INIT     = 1e-3        # initial trust radius Δ₀
TR_DELTA_MIN      = 1e-6        # hard floor
TR_DELTA_MAX      = 1.0         # hard ceiling
TR_ETA0           = 0.10        # rejection threshold ρ < η₀ → contract
TR_ETA1           = 0.75        # expansion threshold ρ ≥ η₁ → expand
TR_GAMMA_PLUS     = 2.0         # expand factor
TR_GAMMA_MINUS    = 0.50        # contract factor

print('Config loaded.')
print(f'  Explore phase : steps 0 → {int(TOTAL_STEPS * EXPLORE_FRAC)}  (cyclic LR, T={T_CYCLE})')
print(f'  Exploit phase : steps {int(TOTAL_STEPS * EXPLORE_FRAC)} → {TOTAL_STEPS}  (cosine decay)')
print(f'  φ decay       : starts at step {int(TOTAL_STEPS * EF_DECAY_START)}, floor={EF_FLOOR}')
print(f'  Trust region  : Δ₀={TR_DELTA_INIT}, η₀={TR_ETA0}, η₁={TR_ETA1}')


# ── Cell 3: Data ──────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'])
from datasets import load_dataset

print('Loading WikiText-2 ...')
wt2 = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)

train_text = '\n'.join(wt2['train']['text'])
val_text   = '\n'.join(wt2['validation']['text'])

chars  = sorted(set(train_text))
VOCAB  = len(chars)
stoi   = {c: i for i, c in enumerate(chars)}
itos   = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join(itos.get(i, '?') for i in l)

train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data   = torch.tensor(encode(val_text),   dtype=torch.long)

print(f'Dataset : WikiText-2 (char-level)')
print(f'Vocab   : {VOCAB} unique chars')
print(f'Train   : {len(train_data):,} chars')
print(f'Val     : {len(val_data):,} chars')

def get_batch(split, rng):
    source = train_data if split == 'train' else val_data
    ix = rng.integers(0, len(source) - BLOCK_SIZE, size=(BATCH_SIZE,))
    x  = torch.stack([source[i   : i+BLOCK_SIZE  ] for i in ix]).to(DEVICE)
    y  = torch.stack([source[i+1 : i+BLOCK_SIZE+1] for i in ix]).to(DEVICE)
    return x, y


# ── Cell 4: NanoGPT model ─────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head  = n_head
        self.n_embd  = n_embd
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj  = nn.Linear(n_embd, n_embd)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer('bias',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = self.attn_drop(F.softmax(att, dim=-1))
        return self.resid_drop(self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)))

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(),
            nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.mlp  = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[Block(N_EMBD, N_HEAD, BLOCK_SIZE, DROPOUT)
                                        for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.head    = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.LayerNorm): nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos    = torch.arange(T, device=idx.device).unsqueeze(0)
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x      = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss   = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            next_tok  = torch.multinomial(F.softmax(logits[:, -1, :] / temperature, dim=-1), 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

n_params = sum(p.numel() for p in NanoGPT().parameters())
print(f'NanoGPT ready — {n_params/1e6:.2f}M parameters')


# ── Cell 5: Tango schedule & probe helpers ────────────────────────────────────

def cyclic_lr(step, T, lr_max, lr_min):
    """Inverted cosine cycle — Eq. 3."""
    t = step % T
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t / T))

def cosine_decay_lr(step, total_steps, lr_start, lr_end):
    """Standard cosine decay — Eq. 5."""
    p = min(step / max(total_steps, 1), 1.0)
    return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * p))

def exploration_factor(step, total_steps, decay_start_frac, floor):
    """φ(t) — Eq. 4. Returns 1.0 until decay_start_frac, then decays to floor."""
    decay_start = int(total_steps * decay_start_frac)
    if step < decay_start:
        return 1.0
    dp = (step - decay_start) / max(total_steps - decay_start, 1)
    return max(floor, 1.0 - dp)


def get_flat_params(model):
    return np.concatenate(
        [p.detach().cpu().float().view(-1).numpy() for p in model.parameters()]
    ).astype(np.float32)

def set_flat_params(model, flat):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(torch.tensor(flat[offset:offset+n], dtype=p.dtype,
                                   device=p.device).view(p.shape))
        offset += n


def fd_curvature_nn(model, xb, yb, g_np, eps=FD_EPS):
    """
    Curvature estimate along unit gradient — Eq. 6.
    Cost: 1 extra forward pass, no autograd graph retained.
    """
    norm_g = np.linalg.norm(g_np)
    if norm_g < 1e-12:
        return 0.0
    norm_g = min(norm_g, 1.0)
    g_hat  = g_np / norm_g
    params0 = get_flat_params(model)
    with torch.no_grad():
        _, l0 = model(xb, yb); f0 = l0.item()
    set_flat_params(model, params0 + g_hat * eps)
    with torch.no_grad():
        _, l1 = model(xb, yb); f1 = l1.item()
    set_flat_params(model, params0)
    return float((f1 - f0 - eps * norm_g) / (0.5 * eps**2 + 1e-30))


class TangentMomentum:
    """Tangent-plane momentum buffer — Eq. 7-8."""
    def __init__(self, dim, beta=TANG_BETA):
        self.beta = beta
        self.v    = np.zeros(dim, np.float32)

    def get_direction(self, g_np):
        g_hat = g_np / (np.linalg.norm(g_np) + 1e-8)
        noise = np.random.randn(len(g_np)).astype(np.float32)
        noise -= np.dot(noise, g_hat) * g_hat   # project onto tangent plane
        noise /= np.linalg.norm(noise) + 1e-8
        self.v = self.beta * self.v + (1.0 - self.beta) * noise
        return self.v / (np.linalg.norm(self.v) + 1e-8)


@torch.no_grad()
def estimate_val_loss(model, rng):
    model.eval()
    losses = [model(xb, yb)[1].item()
              for xb, yb in (get_batch('val', rng) for _ in range(EVAL_ITERS))]
    model.train()
    return float(np.mean(losses))

print('Tango helpers ready.')


# ── Cell 6: run_tango & run_adam ──────────────────────────────────────────────

def run_tango(seed, total_steps,
              use_cyclic_lr=True,
              use_ef_decay=True,
              label='Tango-Full'):
    """
    Run one seed of Tango on NanoGPT/WikiText-2.

    use_cyclic_lr=True  → cyclic explore phase + cosine exploit  (Tango-Full)
    use_cyclic_lr=False → cosine decay throughout                 (Tango-NoEF)
    use_ef_decay=True   → φ(t) decays after EF_DECAY_START
    use_ef_decay=False  → φ(t) = 1.0 throughout
    """
    torch.manual_seed(seed); np.random.seed(seed)
    rng   = np.random.default_rng(seed)
    model = NanoGPT().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_MAX,
                               betas=(0.9, 0.999), weight_decay=0.0)

    n_params    = sum(p.numel() for p in model.parameters())
    tang_mom    = TangentMomentum(dim=n_params)
    explore_end = int(total_steps * EXPLORE_FRAC)

    best_val_loss  = float('inf')
    best_state     = copy.deepcopy(model.state_dict())
    sharpness      = 0.0
    tr_delta       = TR_DELTA_INIT

    tang_exec        = 0
    tang_block_sharp = 0
    tang_block_loss  = 0
    trust_accept     = 0
    trust_reject     = 0
    history          = []

    print(f'  [{label} seed={seed}] {n_params/1e6:.2f}M params | {total_steps} steps')
    print(f'    explore_end={explore_end}  use_cyclic={use_cyclic_lr}  use_ef={use_ef_decay}')

    for step in range(total_steps):

        # ── LR schedule ───────────────────────────────────────────────────────
        if use_cyclic_lr:
            if step < explore_end:
                lr = cyclic_lr(step, T_CYCLE, LR_MAX, LR_MIN)
            else:
                exploit_step  = step - explore_end
                exploit_total = total_steps - explore_end
                lr = cosine_decay_lr(exploit_step, exploit_total,
                                     LR_EXPLOIT_START, LR_EXPLOIT_END)
        else:
            lr = cosine_decay_lr(step, total_steps, LR_COSINE_START, LR_COSINE_END)

        for pg in opt.param_groups:
            pg['lr'] = lr

        # ── Exploration factor φ(t) ───────────────────────────────────────────
        ef = (exploration_factor(step, total_steps, EF_DECAY_START, EF_FLOOR)
              if use_ef_decay else 1.0)

        # ── Gradient step ─────────────────────────────────────────────────────
        xb, yb = get_batch('train', rng)
        opt.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        g_np = get_flat_params(model)   # grab params before step for grad later
        # actually grab grad after backward, before step
        g_np = np.concatenate([
            p.grad.detach().cpu().float().view(-1).numpy() if p.grad is not None
            else np.zeros(p.numel(), np.float32)
            for p in model.parameters()
        ]).astype(np.float32)
        opt.step()
        loss_val = loss.item()

        # ── Sharpness probe — Eq. 6 ───────────────────────────────────────────
        if step % SHARP_UPDATE_INT == 0 and step > 0:
            sharpness = fd_curvature_nn(model, xb, yb, g_np)

        # ── Tangent step with trust region — Eq. 9-10, 12-15 ─────────────────
        if step % TANG_INTERVAL == 0 and step > TANG_WARMUP:

            # Gate 1: sharpness ceiling (dynamic)
            dyn_kmax   = 0.8 * (2.0 / max(lr, 1e-8))
            sharp_gate = sharpness < dyn_kmax

            if not sharp_gate:
                tang_block_sharp += 1
            else:
                # Proposed tangent step magnitude — Eq. 9
                sharp_safe = max(abs(sharpness), 1.0)
                eps_t      = TANG_EPS_BASE * math.sqrt(TANG_SIGMA / sharp_safe) * ef
                eps_t      = float(np.clip(eps_t, 1e-6, TR_DELTA_MAX))

                # Cap to trust radius — Eq. 12
                eps_t = min(eps_t, tr_delta)

                direction = tang_mom.get_direction(g_np)
                delta_np  = direction * eps_t
                delta_t   = torch.tensor(delta_np, dtype=torch.float32, device=DEVICE)

                # Trust-region evaluation: ρ = actual / predicted reduction
                with torch.no_grad():
                    _, l_before = model(xb, yb)
                    f_before = l_before.item()

                # Apply tentative step
                params0 = get_flat_params(model)
                set_flat_params(model, params0 + delta_np)

                with torch.no_grad():
                    _, l_after = model(xb, yb)
                    f_after = l_after.item()

                reduction_actual    = f_before - f_after
                predicted_reduction = 0.5 * abs(sharpness) * eps_t ** 2

                if predicted_reduction < 1e-30:
                    rho = 1.0 if reduction_actual >= 0 else -1.0
                else:
                    rho = reduction_actual / predicted_reduction

                # Accept / reject — Eq. 15
                if rho >= TR_ETA0:
                    # Accept: keep new params
                    trust_accept += 1
                    tang_exec    += 1
                    if rho >= TR_ETA1:
                        tr_delta = min(TR_GAMMA_PLUS * tr_delta, TR_DELTA_MAX)
                else:
                    # Reject: restore params, contract trust radius
                    set_flat_params(model, params0)
                    tr_delta = max(TR_GAMMA_MINUS * tr_delta, TR_DELTA_MIN)
                    trust_reject += 1

        # ── Logging & best checkpoint ─────────────────────────────────────────
        if step % LOG_EVERY == 0:
            val_loss = estimate_val_loss(model, rng)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = copy.deepcopy(model.state_dict())
            phase = 'EXPLORE' if (use_cyclic_lr and step < explore_end) else 'EXPLOIT'
            history.append({
                'step': step, 'train_loss': float(loss_val),
                'val_loss': float(val_loss), 'lr': float(lr),
                'ef': float(ef), 'sharpness': float(sharpness),
                'tr_delta': float(tr_delta), 'phase': phase,
            })
            print(f'    step {step:5d}/{total_steps}  train={loss_val:.4f}  '
                  f'val={val_loss:.4f}  lr={lr:.2e}  ef={ef:.2f}  '
                  f'sharp={sharpness:.2f}  Δ={tr_delta:.2e}  '
                  f'tang={tang_exec}(✓{trust_accept}/✗{trust_reject})  {phase}')

    model.load_state_dict(best_state)
    final_val = estimate_val_loss(model, rng)
    ppl       = math.exp(min(final_val, 20))
    context   = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    sample    = decode(model.generate(context, 300, temperature=0.8)[0].tolist())

    return {
        'optimizer': label, 'seed': seed,
        'final_val_loss': float(final_val),
        'best_val_loss':  float(best_val_loss),
        'perplexity':     float(ppl),
        'tang_exec':      tang_exec,
        'tang_block_sharp': tang_block_sharp,
        'tang_block_loss':  tang_block_loss,
        'trust_accept':   trust_accept,
        'trust_reject':   trust_reject,
        'history':        history,
        'sample':         sample,
    }


def run_adam_single(seed, lr, total_steps):
    torch.manual_seed(seed); np.random.seed(seed)
    rng   = np.random.default_rng(seed)
    model = NanoGPT().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                               betas=(0.9, 0.999), weight_decay=0.0)
    history = []
    for step in range(total_steps):
        xb, yb = get_batch('train', rng)
        opt.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % LOG_EVERY == 0:
            val_loss = estimate_val_loss(model, rng)
            history.append({'step': step, 'train_loss': float(loss.item()),
                            'val_loss': float(val_loss)})
    final_val = estimate_val_loss(model, rng)
    ppl       = math.exp(min(final_val, 20))
    context   = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    sample    = decode(model.generate(context, 300, temperature=0.8)[0].tolist())
    return {'optimizer': f'adam_lr{lr:.0e}', 'lr': lr, 'seed': seed,
            'final_val_loss': float(final_val), 'perplexity': float(ppl),
            'history': history, 'sample': sample}

def run_adam_best(seed, total_steps):
    best = None
    for lr in ADAM_LR_GRID:
        r = run_adam_single(seed, lr, total_steps)
        print(f'  [Adam seed={seed} lr={lr:.0e}]  val_loss={r["final_val_loss"]:.4f}  ppl={r["perplexity"]:.2f}')
        if best is None or r['final_val_loss'] < best['final_val_loss']:
            best = r
    return best

print('Runner functions ready.')


# ── Cell 7: RUN ───────────────────────────────────────────────────────────────
print('=' * 72)
print('  TANGO BENCHMARK — WIKITEXT-2 NANOGPT')
print(f'  {TOTAL_STEPS} steps | {N_SEEDS} seeds | device={DEVICE}')
print(f'  Model: {N_LAYER}L/{N_HEAD}H/{N_EMBD}E | block={BLOCK_SIZE}')
print(f'  Variants: Tango-Full | Tango-NoEF | AdamW-Best')
print('=' * 72)

full_all, noef_all, adam_all = [], [], []
full_losses, noef_losses, adam_losses = [], [], []
full_ppls,   noef_ppls,   adam_ppls   = [], [], []
grand_start = time.time()

for seed in range(N_SEEDS):
    print(f'\n{"─"*72}\n  SEED {seed}\n{"─"*72}')
    t0 = time.time()

    # (A) Tango-Full
    rf = run_tango(seed, TOTAL_STEPS, use_cyclic_lr=True,  use_ef_decay=True,  label='Tango-Full')
    full_all.append(rf); full_losses.append(rf['final_val_loss']); full_ppls.append(rf['perplexity'])

    # (B) Tango-NoEF
    rn = run_tango(seed, TOTAL_STEPS, use_cyclic_lr=False, use_ef_decay=False, label='Tango-NoEF')
    noef_all.append(rn); noef_losses.append(rn['final_val_loss']); noef_ppls.append(rn['perplexity'])

    # (C) AdamW-Best
    ar = run_adam_best(seed, TOTAL_STEPS)
    adam_all.append(ar); adam_losses.append(ar['final_val_loss']); adam_ppls.append(ar['perplexity'])

    def tag(a, b, c):
        best = min(a, b, c)
        return 'Full' if best == a else ('NoEF' if best == b else 'Adam')

    w = tag(rf['final_val_loss'], rn['final_val_loss'], ar['final_val_loss'])
    print(f'\n  seed={seed}  Full={rf["final_val_loss"]:.4f}  NoEF={rn["final_val_loss"]:.4f}  '
          f'Adam={ar["final_val_loss"]:.4f}  [{w}]  {time.time()-t0:.1f}s')

print(f'\nTotal wall time: {time.time()-grand_start:.1f}s')


# ── Cell 8: Summary & plot ────────────────────────────────────────────────────
def stats(arr):
    return float(np.mean(arr)), float(np.std(arr))

fm, fs = stats(full_losses);  fp_m, fp_s = stats(full_ppls)
nm, ns = stats(noef_losses);  np_m, np_s = stats(noef_ppls)
am, as_ = stats(adam_losses); ap_m, ap_s = stats(adam_ppls)

wins_full = sum(f < min(n, a) for f, n, a in zip(full_losses, noef_losses, adam_losses))
wins_noef = sum(n < min(f, a) for f, n, a in zip(full_losses, noef_losses, adam_losses))
wins_adam = sum(a < min(f, n) for f, n, a in zip(full_losses, noef_losses, adam_losses))

print('=' * 72)
print('  FINAL RESULTS — WikiText-2 NanoGPT')
print('=' * 72)
print(f'  {"Optimizer":<14} {"val_loss":>10} {"±std":>7}  {"ppl":>8} {"±std":>7}  {"wins":>5}')
print('  ' + '─' * 55)
print(f'  {"Tango-Full":<14} {fm:>10.4f} {fs:>7.4f}  {fp_m:>8.2f} {fp_s:>7.2f}  {wins_full:>5}/{N_SEEDS}')
print(f'  {"Tango-NoEF":<14} {nm:>10.4f} {ns:>7.4f}  {np_m:>8.2f} {np_s:>7.2f}  {wins_noef:>5}/{N_SEEDS}')
print(f'  {"AdamW-Best":<14} {am:>10.4f} {as_:>7.4f}  {ap_m:>8.2f} {ap_s:>7.2f}  {wins_adam:>5}/{N_SEEDS}')
print(f'\n  Δ ppl Full−Adam : {fp_m - ap_m:+.2f}  (negative = Tango-Full wins)')
print(f'  Δ ppl Full−NoEF : {fp_m - np_m:+.2f}  (ef-decay contribution)')
print('=' * 72)

# ── ef decay contribution (mirrors Rosenbrock ablation output) ────────────────
if fm > 1e-300 and nm > 1e-300:
    ratio_ef = nm / fm
    oom_ef   = math.log10(nm) - math.log10(fm)
    print(f'\n  ef-decay impact (Full vs NoEF): ratio={ratio_ef:.2f}×  OoM={oom_ef:+.2f}')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)

colors = {'Tango-Full': '#55A868', 'Tango-NoEF': '#DD8452', 'AdamW': '#4C72B0'}

# 1. Val loss convergence
ax1 = fig.add_subplot(gs[0, :])
for runs, color, label in [
    (full_all, colors['Tango-Full'], 'Tango-Full'),
    (noef_all, colors['Tango-NoEF'], 'Tango-NoEF'),
    (adam_all, colors['AdamW'],      'AdamW-Best'),
]:
    for r in runs:
        xs = [h['step'] for h in r['history']]
        ys = [h['val_loss'] for h in r['history']]
        ax1.plot(xs, ys, color=color, alpha=0.18, linewidth=1)
    min_len = min(len(r['history']) for r in runs)
    mean_y  = np.mean([[h['val_loss'] for h in r['history']][:min_len] for r in runs], axis=0)
    xs_ref  = [runs[0]['history'][j]['step'] for j in range(min_len)]
    ax1.plot(xs_ref, mean_y, color=color, linewidth=2.5, label=f'{label} (mean)')

# shade explore / exploit boundary for Tango-Full
explore_end = int(TOTAL_STEPS * EXPLORE_FRAC)
ax1.axvline(explore_end, color=colors['Tango-Full'], linestyle='--', alpha=0.4, linewidth=1)
ax1.text(explore_end + 30, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 1.5,
         '← explore | exploit →', fontsize=8, color=colors['Tango-Full'], alpha=0.7)

ax1.set_xlabel('Training step'); ax1.set_ylabel('Validation loss')
ax1.set_title('Val Loss Convergence — WikiText-2 NanoGPT', fontweight='bold')
ax1.legend(); ax1.grid(alpha=0.3)

# 2. Per-seed final val loss
ax2 = fig.add_subplot(gs[1, 0])
seeds = list(range(N_SEEDS))
ax2.plot(seeds, full_losses, 's-', color=colors['Tango-Full'], label='Tango-Full', linewidth=2, markersize=7)
ax2.plot(seeds, noef_losses, 'D-', color=colors['Tango-NoEF'], label='Tango-NoEF', linewidth=2, markersize=7)
ax2.plot(seeds, adam_losses, 'o-', color=colors['AdamW'],      label='AdamW-Best', linewidth=2, markersize=7)
ax2.set_xlabel('Seed'); ax2.set_ylabel('Final val loss')
ax2.set_title('Per-Seed Final Val Loss', fontweight='bold')
ax2.legend(); ax2.grid(alpha=0.3)

# 3. PPL bar chart
ax3 = fig.add_subplot(gs[1, 1])
x  = np.arange(N_SEEDS)
w  = 0.25
ax3.bar(x - w,   full_ppls, w, label='Tango-Full', color=colors['Tango-Full'], alpha=0.85)
ax3.bar(x,       noef_ppls, w, label='Tango-NoEF', color=colors['Tango-NoEF'], alpha=0.85)
ax3.bar(x + w,   adam_ppls, w, label='AdamW-Best', color=colors['AdamW'],      alpha=0.85)
ax3.set_xlabel('Seed'); ax3.set_ylabel('Perplexity')
ax3.set_title('Perplexity per Seed', fontweight='bold')
ax3.set_xticks(x); ax3.set_xticklabels([f'seed {s}' for s in seeds])
ax3.legend(); ax3.grid(axis='y', alpha=0.3)

# 4. LR schedule (first seed Tango-Full)
ax4 = fig.add_subplot(gs[1, 2])
if full_all:
    hs = full_all[0]['history']
    ax4.plot([h['step'] for h in hs], [h['lr']  for h in hs],
             color=colors['Tango-Full'], linewidth=2, label='Tango-Full LR')
if noef_all:
    hs_n = noef_all[0]['history']
    ax4.plot([h['step'] for h in hs_n], [h['lr'] for h in hs_n],
             color=colors['Tango-NoEF'], linewidth=2, linestyle='--', label='Tango-NoEF LR')
ax4.axvline(explore_end, color='gray', linestyle=':', alpha=0.5)
ax4.set_xlabel('Step'); ax4.set_ylabel('Learning rate')
ax4.set_title('LR Schedule (seed 0)', fontweight='bold')
ax4.legend(); ax4.grid(alpha=0.3)

plt.suptitle(
    f'Tango (Full vs NoEF) vs AdamW — WikiText-2 NanoGPT\n'
    f'({N_LAYER}L {N_EMBD}d, {n_params:,} params, {TOTAL_STEPS} steps, {N_SEEDS} seeds, char-level)',
    fontsize=12, fontweight='bold')
plt.savefig('tango_wikitext2_results.png', dpi=150, bbox_inches='tight')
plt.show()
print('Plot saved → tango_wikitext2_results.png')


# ── Cell 9: Save results ──────────────────────────────────────────────────────
out = 'tango_wikitext2_results.json'
with open(out, 'w') as f:
    json.dump({
        'suite': 'tango_wikitext2',
        'config': {
            'total_steps': TOTAL_STEPS, 'n_seeds': N_SEEDS,
            'n_layer': N_LAYER, 'n_head': N_HEAD, 'n_embd': N_EMBD,
            'block_size': BLOCK_SIZE, 'batch_size': BATCH_SIZE,
            'explore_frac': EXPLORE_FRAC, 't_cycle': T_CYCLE,
            'lr_max': LR_MAX, 'lr_min': LR_MIN,
            'lr_exploit_start': LR_EXPLOIT_START, 'lr_exploit_end': LR_EXPLOIT_END,
            'ef_decay_start': EF_DECAY_START, 'ef_floor': EF_FLOOR,
            'tr_delta_init': TR_DELTA_INIT, 'tr_eta0': TR_ETA0, 'tr_eta1': TR_ETA1,
            'adam_lr_grid': ADAM_LR_GRID,
        },
        'summary': {
            'tango_full': {'val_loss_mean': fm, 'val_loss_std': fs, 'ppl_mean': fp_m, 'ppl_std': fp_s, 'wins': wins_full},
            'tango_noef': {'val_loss_mean': nm, 'val_loss_std': ns, 'ppl_mean': np_m, 'ppl_std': np_s, 'wins': wins_noef},
            'adamw_best': {'val_loss_mean': am, 'val_loss_std': as_, 'ppl_mean': ap_m, 'ppl_std': ap_s, 'wins': wins_adam},
            'delta_ppl_full_vs_adam': fp_m - ap_m,
            'delta_ppl_full_vs_noef': fp_m - np_m,
        },
        'all_runs': full_all + noef_all + adam_all,
    }, f, indent=2)
print(f'Saved → {out}')
try:
    from google.colab import files
    files.download(out)
    files.download('tango_wikitext2_results.png')
    print('Downloads triggered.')
except ImportError:
    pass
