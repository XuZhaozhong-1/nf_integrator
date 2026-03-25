# nfmodel/scripts/zg_pipeline_costh.py
import math
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from nfmodel.flows.zg_costh_flow import ZGCosthFlow
from nfmodel.physics.zg_phase_space import build_event_zg, dphi2_dcosth_dphi, MZ_DEFAULT
from nfmodel.physics.zg_me import me2
from nfmodel.physics.cuts import passes_cuts

TWOPI = 2.0 * math.pi
PHI0 = 0.0  # fixed at LO rotational symmetry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -----------------------
# Utils
# -----------------------
def pick_device():
    # return "mps" if torch.backends.mps.is_available() else "cpu"
    return "cpu"


def sample_uniform_costh(n, rng):
    return rng.uniform(-1.0, 1.0, size=n)


def m2_value(Ecm, mZ, costh, phi):
    """Compute M^2 with cuts; return 0 if cut fail or invalid."""
    p_all = build_event_zg(Ecm, float(costh), float(phi), mZ=mZ)
    if not passes_cuts(p_all):
        return 0.0
    val = me2(p_all)
    return float(val)


# -----------------------
# TRAIN (cosθ only)
# -----------------------
@torch.no_grad()
def compute_weights_costh_batch(costh_t, Ecm: float, mZ: float):
    """
    Weights for training:
      w_i ∝ |M(c_i)|^2  (φ fixed at LO)
    costh_t: shape (B,)
    Returns: w: shape (B,), float32 on same device as costh_t
    """
    B = costh_t.shape[0]
    w = torch.zeros(B, dtype=torch.float32, device=costh_t.device)
    # compute on CPU for MG calls
    costh_cpu = costh_t.detach().cpu().numpy()
    for i in range(B):
        w[i] = float(m2_value(Ecm, mZ, float(costh_cpu[i]), PHI0))
    return w


def train_flow_costh(
    Ecm=1000.0,
    mZ=MZ_DEFAULT,
    steps=2500,
    batch_size=10000,
    lr=2e-4,
    n_blocks=16,         # 8 blocks => 16 layers total (coupling + perm)
    hidden=16,
    permute="reverse",
    seed=0,
    model_path=None,
):
    device = pick_device()
    print(f"[train] device={device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    flow = ZGCosthFlow(n_blocks=n_blocks, hidden=hidden, permute=permute, seed=seed).to(device)
    opt = optim.Adam(flow.parameters(), lr=lr)

    losses = []
    best = float("inf")

    for step in range(1, steps + 1):
        # uniform c in (-1,1)
        c = 2.0 * torch.rand(batch_size, device=device) - 1.0

        w = compute_weights_costh_batch(c, Ecm=Ecm, mZ=mZ)  # (B,)
        ws = float(w.sum().item())
        if ws <= 0:
            continue
        w = w / ws

        logq = flow.logprob_c(c)  # (B,)
        lvari

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        lval = float(loss.item())
        losses.append(lval)
        best = min(best, lval)

        if step % 100 == 0:
            frac = float((w > 0).float().mean().item())
            print(f"[train] step {step:5d}  loss {lval:.6f}  best {best:.6f}  w_nonzero {frac:.3f}")

    if model_path is not None:
        out = Path(model_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": flow.state_dict(),
                "n_blocks": n_blocks,
                "hidden": hidden,
                "permute": permute,
                "seed": seed,
                "Ecm": Ecm,
                "mZ": mZ,
                "phi": float(PHI0),
            },
            out,
        )
        print(f"[train] saved: {out.resolve()}")

    return flow, losses


# -----------------------
# INTEGRATION
# -----------------------
def integrate_baseline_uniform_costh(Ecm, mZ, n=20000, seed=0):
    """
    Uniform in c only (φ integrated out analytically at LO):
      I = J * (2π) * ∫_{-1}^1 dc |M(c)|^2
    """
    rng = np.random.default_rng(seed)
    s = Ecm * Ecm
    flux = 2.0 * s
    jac = dphi2_dcosth_dphi(Ecm, mZ)

    c = sample_uniform_costh(n, rng)

    vals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        vals[i] = m2_value(Ecm, mZ, c[i], PHI0)

    mean = vals.mean()
    err = vals.std(ddof=1) / np.sqrt(n)

    # ∫_{-1}^1 dc f(c) ≈ 2 * mean
    I = jac * (2.0 * math.pi) * (2.0 * mean)
    dI = jac * (2.0 * math.pi) * (2.0 * err)

    sigma = I / flux
    dsigma = dI / flux
    return I, dI, sigma, dsigma


@torch.no_grad()
def integrate_nf_costh(flow: ZGCosthFlow, Ecm, mZ, n=20000, seed=0):
    """
    NF importance sampling in c only:
      ∫_{-1}^1 dc f(c) = E_{c~q}[ f(c) / q(c) ]
      I = J*(2π) * E[ f/q ]
    """
    device = next(flow.parameters()).device
    s = Ecm * Ecm
    flux = 2.0 * s
    jac = dphi2_dcosth_dphi(Ecm, mZ)

    c = flow.sample_c(n, device=device)          # (n,)
    logq = flow.logprob_c(c)                     # (n,)
    q = torch.exp(logq).detach().cpu().numpy()
    c_np = c.detach().cpu().numpy()

    contrib = np.zeros(n, dtype=np.float64)
    for i in range(n):
        val = m2_value(Ecm, mZ, c_np[i], PHI0)
        if q[i] <= 0 or not np.isfinite(q[i]):
            contrib[i] = 0.0
        else:
            contrib[i] = val / q[i]

    mean = contrib.mean()
    err = contrib.std(ddof=1) / np.sqrt(n)

    I = jac * (2.0 * math.pi) * mean
    dI = jac * (2.0 * math.pi) * err

    sigma = I / flux
    dsigma = dI / flux
    return I, dI, sigma, dsigma


# -----------------------
# PLOTTING
# -----------------------
def plot_training_curve(losses, out_dir: Path, Ecm: float, steps: int, batch_size: int, lr: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title(f"Training loss | Ecm={Ecm:.0f} GeV | steps={steps} | batch={batch_size} | lr={lr:g}")
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=150)
    plt.close()


def plot_costh_target_vs_nf(flow, Ecm, mZ, out_dir: Path, n_target=120000, n_nf=120000, seed=0):
    """
    Target p*(c) ∝ |M(c)|^2 estimated by uniform c with weights
    vs NF samples from q(c).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # target via weighted uniform
    c_u = sample_uniform_costh(n_target, rng)
    w = np.zeros(n_target, dtype=np.float64)
    for i in range(n_target):
        w[i] = m2_value(Ecm, mZ, c_u[i], PHI0)
    w = w / (w.sum())

    # NF samples
    device = next(flow.parameters()).device
    c_nf = flow.sample_c(n_nf, device=device).detach().cpu().numpy()

    bins = np.linspace(-1.0, 1.0, 61)
    tgt_hist, _ = np.histogram(c_u, bins=bins, weights=w)
    nf_hist, _ = np.histogram(c_nf, bins=bins)
    nf_hist = nf_hist / (nf_hist.sum())

    centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure()
    plt.plot(centers, tgt_hist, label=r"Target $p^\star(c)\propto |\mathcal{M}(c)|^2$ (weighted)")
    plt.plot(centers, nf_hist, label=r"NF samples $q_\theta(c)$")
    plt.xlabel(r"$c=\cos\theta$")
    plt.ylabel("normalized probability per bin")
    plt.title(f"Target vs NF | Ecm={Ecm:.0f} GeV | phi={PHI0}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "hist_costh_target_vs_nf.png", dpi=150)
    plt.close()


def plot_flow_transforms(flow: ZGCosthFlow, Ecm, mZ, out_dir: Path, n=120000, seed=0):
    """
    Two validations:
      (A) forward: z ~ N(0,1)  -> y -> c
      (B) inverse: c ~ q(c) -> y -> z   (z should look ~N(0,1))
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    device = next(flow.parameters()).device

    # ---------- (A) forward ----------
    z = torch.randn(n, 1, device=device)
    z_np = z.detach().cpu().numpy().reshape(-1)

    y = flow.z_to_y(z)  # (n,1)
    y_np = y.detach().cpu().numpy().reshape(-1)

    c_from_z = torch.tanh(y[:, 0]).detach().cpu().numpy()

    # target curve for c
    c_u = rng.uniform(-1.0, 1.0, size=n)
    w = np.zeros(n, dtype=np.float64)
    for i in range(n):
        w[i] = m2_value(Ecm, mZ, c_u[i], PHI0)
    w = w / (w.sum())

    # ---------- (B) inverse ----------
    c_u2 = rng.uniform(-1.0, 1.0, size=n)
    w2 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        w2[i] = m2_value(Ecm, mZ, c_u2[i], phi=PHI0)
    w2 = w2 / (w2.sum())
    idx = rng.choice(n, size=n, replace=True, p=w2)
    c_tgt_np = c_u2[idx]  # numpy samples ~ p*(c)
    c_tgt = torch.from_numpy(c_tgt_np).to(device=device, dtype=y.dtype if 'y' in locals() else torch.float32)
    y_from_c = flow.c_to_y(c_tgt).unsqueeze(-1)             # (n,1)
    z_from_c = flow.y_to_z(y_from_c)                       # (n,1)
    z_from_c_np = z_from_c.detach().cpu().numpy().reshape(-1)

    # ---------- plot ----------
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)

    # Row 1: z -> y -> c
    axes[0, 0].hist(z_np, bins=80, density=True)
    axes[0, 0].set_title(r"Forward: $z\sim\mathcal{N}(0,1)$")

    axes[0, 1].hist(y_np, bins=80, density=True)
    axes[0, 1].set_title(r"Forward: $y=f_\theta(z)$")

    bins_c = np.linspace(-1.0, 1.0, 80)
    axes[0, 2].hist(c_from_z, bins=bins_c, density=True, alpha=0.6, label=r"NF samples $q_\theta(c)$")
    tgt_hist, _ = np.histogram(c_u, bins=bins_c, weights=w)
    binw = bins_c[1] - bins_c[0]
    tgt_density = tgt_hist / (tgt_hist.sum() * binw)
    centers = 0.5 * (bins_c[:-1] + bins_c[1:])
    axes[0, 2].plot(centers, tgt_density, lw=2, label=r"Target $\propto|\mathcal{M}(c)|^2$")
    axes[0, 2].set_title(r"Forward: $c=\tanh(y)$")
    axes[0, 2].legend(frameon=False)

    # Row 2: c -> y -> z
    #c_nf_np = c_tgt.detach().cpu().numpy()
    axes[1, 0].hist(c_tgt_np, bins=bins_c, density=True)
    axes[1, 0].set_title(r"Inverse: $c\sim p_{target}(c)$")

    axes[1, 1].hist(y_from_c.detach().cpu().numpy().reshape(-1), bins=80, density=True)
    axes[1, 1].set_title(r"Inverse: $y=\operatorname{atanh}(c)$")

    axes[1, 2].hist(z_from_c_np, bins=80, density=True)
    axes[1, 2].set_title(r"Inverse: $z=f_\theta^{-1}(y)$ (should be normal)")

    for ax in axes.flatten():
        ax.set_ylabel("density")

    fig.suptitle(f"Ecm={Ecm:.0f} GeV | n={n} | phi={PHI0}", fontsize=10)
    fig.savefig(out_dir / "flow_forward_inverse_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def hist_lhe_density_with_err(x, edges):
    """
    Unweighted events: p_hat = count/N, Var(p_hat) ≈ p(1-p)/N.
    Density = p_hat / Δ.
    """
    N = len(x)
    h, _ = np.histogram(x, bins=edges)
    bw = np.diff(edges)

    p_hat = h / max(N, 1)
    dens = p_hat / bw

    se_p = np.sqrt(np.maximum(p_hat * (1.0 - p_hat), 0.0) / max(N, 1))
    se_dens = se_p / bw

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, dens, se_dens


def hist_nf_density_with_err(x, w, edges):
    """
    Weighted (importance) events: estimator p_hat = S_b / S
      S_b = sum w_i 1_i,  S = sum w_i
    Var(p_hat) ≈ (1/S^2) * (1/(N-1)) * sum (w_i (1_i - p_hat))^2
    Density = p_hat / Δ.
    """
    x = np.asarray(x)
    w = np.asarray(w, dtype=np.float64)
    N = len(x)
    bw = np.diff(edges)

    # bin assignment
    bin_idx = np.digitize(x, edges) - 1  # bins 0..B-1
    B = len(edges) - 1

    S = np.sum(w)
    centers = 0.5 * (edges[:-1] + edges[1:])

    dens = np.zeros(B, dtype=np.float64)
    se_dens = np.zeros(B, dtype=np.float64)

    if S <= 0 or N <= 1:
        return centers, dens, se_dens

    for b in range(B):
        mask = (bin_idx == b)
        Sb = np.sum(w[mask])
        p_hat = Sb / S

        # g_i = w_i (I_i - p_hat)
        I = mask.astype(np.float64)
        g = w * (I - p_hat)

        var_p = (np.sum(g * g) / (N - 1)) / (S * S)   # <-- key formula
        se_p = np.sqrt(max(var_p, 0.0))

        dens[b] = p_hat / bw[b]
        se_dens[b] = se_p / bw[b]

    return centers, dens, se_dens
# -----------------------
# SAVE RESULTS
# -----------------------
def save_comparison(out_path: Path, baseline: dict, nf: dict, meta: dict):
    dIb = float(baseline["dI"])
    dIn = float(nf["dI"])
    vr = (dIb / dIn) ** 2 if dIn > 0 else float("inf")

    record = {
        "baseline": baseline,
        "nf_costh": nf,
        "variance_reduction": vr,
        "meta": meta,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[saved] comparison → {out_path.resolve()}")


# -----------------------
# MAIN PIPELINE
# -----------------------
def main():
    Ecm = 1000.0
    mZ = MZ_DEFAULT

    # Train settings
    steps = 1000
    batch_size = 1000000
    lr = 2e-4
    hidden = 16
    n_blocks = 16         # => 16 layers total (coupling + perm)
    permute = "reverse"

    # Eval settings
    n_eval = 20000

    model_path = PROJECT_ROOT / "nfmodel" / "models" / "zg_costh_flow.pt"
    plot_dir = PROJECT_ROOT / "nfmodel" / "plots" / "zg_pipeline_costh"
    results_path = PROJECT_ROOT / "nfmodel" / "results" / "zg_costh_comparison.json"

    # 1) Train
    flow, losses = train_flow_costh(
        Ecm=Ecm,
        mZ=mZ,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        hidden=hidden,
        n_blocks=n_blocks,
        permute=permute,
        model_path=model_path,
    )

    # 2) Validation plots: forward + inverse transforms
    plot_flow_transforms(flow, Ecm, mZ, plot_dir, n=120000, seed=0)

    # 3) Evaluate baseline vs NF (with timing)
    t0 = time.time()
    Ib, dIb, sb, dsb = integrate_baseline_uniform_costh(Ecm, mZ, n=n_eval, seed=1)
    t1 = time.time()
    In, dIn, sn, dsn = integrate_nf_costh(flow, Ecm, mZ, n=n_eval, seed=2)
    t2 = time.time()

    print("\n=== Results (same N) ===")
    print(f"[baseline cosθ] I = {Ib:.6e} ± {dIb:.2e} | sigma_hat = {sb:.6e} ± {dsb:.2e}")
    print(f"[NF costh]      I = {In:.6e} ± {dIn:.2e} | sigma_hat = {sn:.6e} ± {dsn:.2e}")

    vr = (dIb / dIn) ** 2
    print(f"[improvement] variance reduction factor ≈ {vr:.2f}×")
    print(f"[timing] baseline integration: {t1 - t0:.2f}s")
    print(f"[timing] NF integration:       {t2 - t1:.2f}s")

    save_comparison(
        results_path,
        baseline={
            "I": float(Ib),
            "dI": float(dIb),
            "sigma_hat": float(sb),
            "dsigma_hat": float(dsb),
            "N": int(n_eval),
            "time_sec": float(t1 - t0),
        },
        nf={
            "I": float(In),
            "dI": float(dIn),
            "sigma_hat": float(sn),
            "dsigma_hat": float(dsn),
            "N": int(n_eval),
            "time_sec": float(t2 - t1),
        },
        meta={
            "Ecm": float(Ecm),
            "mZ": float(mZ),
            "phi": float(PHI0),
            "steps": int(steps),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "hidden": int(hidden),
            "n_blocks": int(n_blocks),
            "layers_total": int(2 * n_blocks),
            "permute": permute,
            "seed": 0,
        },
    )

    # 4) Remaining plots
    plot_training_curve(losses, plot_dir, Ecm=Ecm, steps=steps, batch_size=batch_size, lr=lr)
    plot_costh_target_vs_nf(flow, Ecm, mZ, plot_dir, n_target=120000, n_nf=120000, seed=3)

    print(f"\n[done] plots saved to: {plot_dir.resolve()}")
    print("[done] files: training_loss.png, hist_costh_target_vs_nf.png, flow_forward_inverse_validation.png")
    print(f"[done] results saved to: {results_path.resolve()}")


if __name__ == "__main__":
    main()