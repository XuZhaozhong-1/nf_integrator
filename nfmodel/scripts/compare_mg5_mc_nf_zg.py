#!/usr/bin/env python3
"""
Compare MG5 vs your own MC (uniform) vs NF importance sampling for u u~ -> Z g at fixed Ecm.

Outputs:
  - Printed cross sections (pb) + MC uncertainties
  - Shape plot: dP/dc from LHE vs normalized |M|^2(c) vs NF target-weighted
  - (Optional) Saves a JSON summary

Usage example:
  python -m nfmodel.scripts.compare_mg5_mc_nf_zg \
    --lhe "/path/to/unweighted_events.lhe.gz" \
    --model "nfmodel/models/zg_costh_flow.pt" \
    --Ecm 1000 \
    --N 20000 \
    --bins 60 \
    --outdir "nfmodel/plots/mg5_mc_nf"

Notes:
  - MG5 LHE is unweighted => its cosθ histogram is dP/dc (shape only).
  - Cross section comparison is meaningful only if:
      * same process (u u~ > z g)
      * same cuts (ptj, etaj, etc.)
      * same coupling/scale choices (you already fixed MG5 to match).
"""

import math
import json
import gzip
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from nfmodel.flows.zg_costh_flow import ZGCosthFlow
from nfmodel.physics.zg_phase_space import build_event_zg, dphi2_dcosth_dphi, MZ_DEFAULT
from nfmodel.physics.zg_me import me2
from nfmodel.physics.cuts import passes_cuts

PHI0 = 0.0

# GeV^-2 to pb
GEV2_TO_PB = 3.89379e8


# -----------------------
# IO helpers
# -----------------------
def open_maybe_gz(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def project_root() -> Path:
    # robust even if cwd changes inside MG wrappers
    return Path(__file__).resolve().parents[2]


def parse_mg5_sigma_from_lhe_header(lhe_path: Path):
    """
    Extract MG5 "Integrated weight (pb)" from the LHE header.
    Returns: sigma_pb or None.
    """
    sigma_pb = None
    with open_maybe_gz(lhe_path) as f:
        for line in f:
            if "Integrated weight (pb)" in line:
                # Example: "#  Integrated weight (pb)  :       4.4684"
                try:
                    sigma_pb = float(line.split(":")[-1].strip().split()[0])
                except Exception:
                    pass
            if "<event" in line:
                break
    return sigma_pb


# -----------------------
# LHE -> cos(theta)
# -----------------------
def parse_lhe_costh(lhe_path: Path):
    """
    Read unweighted LHE and compute cosθ of the Z in the partonic CM frame.
    For u u~ -> Z g, incoming partons are status -1, Z is status 1 and pid=23.
    """
    cos_list = []
    with open_maybe_gz(lhe_path) as f:
        in_event = False
        event_lines = []
        for line in f:
            s = line.strip()
            if s.startswith("<event"):
                in_event = True
                event_lines = []
                continue

            if not in_event:
                continue

            if s.startswith("</event"):
                if not event_lines:
                    in_event = False
                    continue

                header = event_lines[0].split()
                try:
                    nup = int(header[0])
                except Exception:
                    in_event = False
                    continue

                parts = event_lines[1 : 1 + nup]

                pZ = None
                for pl in parts:
                    cols = pl.split()
                    if len(cols) < 10:
                        continue
                    pid = int(cols[0])
                    ist = int(cols[1])
                    px, py, pz, E = map(float, cols[6:10])

                    if ist == 1 and pid == 23:
                        pZ = (E, px, py, pz)

                if pZ is None:
                    in_event = False
                    continue

                p = math.sqrt(pZ[1] ** 2 + pZ[2] ** 2 + pZ[3] ** 2) + 1e-30
                c = pZ[3] / p
                cos_list.append(c)

                in_event = False
                continue

            # inside <event> ... </event>
            if s and (not s.startswith("#")):
                event_lines.append(line)

    return np.asarray(cos_list, dtype=np.float64)


# -----------------------
# Physics: |M|^2(c)
# -----------------------
def m2_at_c(Ecm, mZ, c, use_cuts=True):
    """
    Compute |M|^2 at (Ecm, c=cosθ, phi fixed).
    If use_cuts=True, applies passes_cuts on the built event.
    Returns 0.0 for cut-fail or invalid me2.
    """
    p_all = build_event_zg(Ecm, float(c), PHI0, mZ=mZ)
    if use_cuts and (not passes_cuts(p_all)):
        return 0.0
    val = float(me2(p_all))
    if (not np.isfinite(val)) or val < 0.0:
        return 0.0
    return val


# -----------------------
# Cross section estimators
# -----------------------
def sigma_from_uniform_costh(Ecm, mZ, N, seed=0, use_cuts=True):
    """
    Baseline MC:
      c ~ Uniform(-1,1)
      ∫_{-1}^1 dc f(c) ≈ 2 * mean(f)
    """
    rng = np.random.default_rng(seed)
    c = rng.uniform(-1.0, 1.0, size=N)
    vals = np.array([m2_at_c(Ecm, mZ, ci, use_cuts=use_cuts) for ci in c], dtype=np.float64)

    mean = vals.mean()
    stderr = vals.std(ddof=1) / math.sqrt(N)

    s = Ecm * Ecm
    flux = 2.0 * s
    jac = dphi2_dcosth_dphi(Ecm, mZ)

    I = jac * (2.0 * math.pi) * (2.0 * mean)
    dI = jac * (2.0 * math.pi) * (2.0 * stderr)

    sigma_gev2 = I / flux
    dsigma_gev2 = dI / flux

    sigma_pb = sigma_gev2 * GEV2_TO_PB
    dsigma_pb = dsigma_gev2 * GEV2_TO_PB
    return sigma_pb, dsigma_pb


@torch.no_grad()
def sigma_from_nf(flow: ZGCosthFlow, Ecm, mZ, N, seed=0, use_cuts=True):
    """
    NF importance sampling:
      c ~ q(c)
      ∫ dc f(c) = E_q[ f(c)/q(c) ]
    """
    # seed for reproducible sampling from torch
    torch.manual_seed(seed)
    device = next(flow.parameters()).device

    c = flow.sample_c(N, device=device)     # (N,)
    logq = flow.logprob_c(c)               # (N,)
    q = torch.exp(logq).detach().cpu().numpy()
    c_np = c.detach().cpu().numpy()

    contrib = np.zeros(N, dtype=np.float64)
    for i in range(N):
        qi = float(q[i])
        if (not np.isfinite(qi)) or qi <= 0.0:
            contrib[i] = 0.0
            continue
        fi = m2_at_c(Ecm, mZ, c_np[i], use_cuts=use_cuts)
        contrib[i] = fi / qi

    mean = contrib.mean()
    stderr = contrib.std(ddof=1) / math.sqrt(N)

    s = Ecm * Ecm
    flux = 2.0 * s
    jac = dphi2_dcosth_dphi(Ecm, mZ)

    I = jac * (2.0 * math.pi) * mean
    dI = jac * (2.0 * math.pi) * stderr

    sigma_gev2 = I / flux
    dsigma_gev2 = dI / flux

    sigma_pb = sigma_gev2 * GEV2_TO_PB
    dsigma_pb = dsigma_gev2 * GEV2_TO_PB
    return sigma_pb, dsigma_pb


# -----------------------
# Shape plot: dP/dc
# -----------------------

def make_shape_plot(
    lhe_path,
    flow,
    Ecm,
    mZ,
    bins,
    out_png,
    seed=0,
    use_cuts=True,
    nf_shape_N=200000,
    ratio_ylim=(0.9, 1.1),
    pull_ylim=(-4.0, 4.0),
):
    """
    Shape plot with per-bin uncertainty + ratio + pull.

    Panels:
      (1) density dP/dc : MG5 LHE (unweighted) + NF(reweighted) + |M|^2 curve
      (2) ratio: (NF / MG5) with error band
      (3) pull:  (NF - MG5) / sqrt(se_nf^2 + se_lhe^2)

    Notes:
      - LHE is unweighted => binomial/multinomial SE for p_hat ~ count/N
      - NF uses self-normalized IS variance estimate per bin:
          Var(p_hat(b)) ≈ (1/S^2)*(1/(N-1))*sum_i ( w_i (I_i(b) - p_hat(b)) )^2
        then density = p_hat / Δ
    """

    # -----------------------
    # helpers
    # -----------------------
    def hist_lhe_density_with_err(x, edges):
        N = len(x)
        h, _ = np.histogram(x, bins=edges)
        bw = np.diff(edges)

        # p_hat = count/N, density = p_hat/Δ
        p_hat = h / max(N, 1)
        dens = p_hat / bw

        # binomial approx (fine for large N; multinomial is similar scale)
        se_p = np.sqrt(np.maximum(p_hat * (1.0 - p_hat), 0.0) / max(N, 1))
        se_dens = se_p / bw

        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, dens, se_dens

    def hist_nf_density_with_err(x, w, edges):
        x = np.asarray(x)
        w = np.asarray(w, dtype=np.float64)
        N = len(x)
        bw = np.diff(edges)
        B = len(edges) - 1

        centers = 0.5 * (edges[:-1] + edges[1:])
        dens = np.zeros(B, dtype=np.float64)
        se_dens = np.zeros(B, dtype=np.float64)

        S = float(np.sum(w))
        if S <= 0.0 or N <= 1:
            return centers, dens, se_dens

        bin_idx = np.digitize(x, edges) - 1  # 0..B-1, out of range -> <0 or >=B

        for b in range(B):
            mask = (bin_idx == b)
            Sb = float(np.sum(w[mask]))
            p_hat = Sb / S

            I = mask.astype(np.float64)
            g = w * (I - p_hat)

            var_p = (np.sum(g * g) / (N - 1)) / (S * S)
            se_p = math.sqrt(max(var_p, 0.0))

            dens[b] = p_hat / bw[b]
            se_dens[b] = se_p / bw[b]

        return centers, dens, se_dens

    # -----------------------
    # (1) LHE -> cosθ + density + err
    # -----------------------
    c_lhe = parse_lhe_costh(lhe_path)

    edges = np.linspace(-1.0, 1.0, int(bins) + 1)
    centers, dens_lhe, se_lhe = hist_lhe_density_with_err(c_lhe, edges)
    bw = edges[1] - edges[0]

    # -----------------------
    # (2) |M|^2 curve on centers (normalized)
    # -----------------------
    m2_vals = np.array([m2_at_c(Ecm, mZ, c, use_cuts=use_cuts) for c in centers], dtype=np.float64)
    dens_m2 = m2_vals / (np.sum(m2_vals) * bw + 1e-30)

    # -----------------------
    # (3) NF sample -> weights w = |M|^2/q
    # -----------------------
    torch.manual_seed(seed)
    device = next(flow.parameters()).device

    with torch.no_grad():
        c_nf = flow.sample_c(int(nf_shape_N), device=device)
        logq_nf = flow.logprob_c(c_nf)

    c_nf_np = c_nf.detach().cpu().numpy()
    q_nf = np.exp(logq_nf.detach().cpu().numpy())

    w_nf = np.zeros_like(c_nf_np, dtype=np.float64)
    for i in range(len(c_nf_np)):
        qi = float(q_nf[i])
        if (not np.isfinite(qi)) or qi <= 0.0:
            w_nf[i] = 0.0
            continue
        fi = m2_at_c(Ecm, mZ, float(c_nf_np[i]), use_cuts=use_cuts)
        w_nf[i] = fi / qi

    centers_nf, dens_nf, se_nf = hist_nf_density_with_err(c_nf_np, w_nf, edges)

    # sanity: should match centers
    assert np.allclose(centers_nf, centers)

    # -----------------------
    # ratio + pull (binwise)
    # -----------------------
    eps = 1e-30
    valid = dens_lhe > 0

    ratio = np.full_like(dens_lhe, np.nan, dtype=np.float64)
    ratio_se = np.full_like(dens_lhe, np.nan, dtype=np.float64)
    ratio[valid] = dens_nf[valid] / dens_lhe[valid]

    # error propagation for ratio assuming independence
    # r = a/b => (σr/r)^2 = (σa/a)^2 + (σb/b)^2
    valid2 = valid & (dens_nf > 0)
    ratio_se[valid2] = ratio[valid2] * np.sqrt(
        (se_nf[valid2] / (dens_nf[valid2] + eps)) ** 2
        + (se_lhe[valid2] / (dens_lhe[valid2] + eps)) ** 2
    )

    pull_den = np.sqrt(se_nf**2 + se_lhe**2)
    pull = np.full_like(dens_lhe, np.nan, dtype=np.float64)
    pull[pull_den > 0] = (dens_nf[pull_den > 0] - dens_lhe[pull_den > 0]) / pull_den[pull_den > 0]

    # -----------------------
    # plotting (3-panel)
    # -----------------------
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3, 1, figsize=(8.5, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.3, 1.3], "hspace": 0.08}
    )
    ax0, ax1, ax2 = axes

    # Panel 1: density with error bars + curve
    ax0.errorbar(
        centers, dens_lhe, yerr=se_lhe,
        fmt="o", ms=3, capsize=2, elinewidth=1,
        label="MG5 LHE (unweighted)", zorder=5
    )
    ax0.errorbar(
        centers, dens_nf, yerr=se_nf,
        fmt="o", ms=3, capsize=2, elinewidth=1,
        label="NF (reweighted by |M|^2/q)", zorder=6
    )
    ax0.plot(centers, dens_m2, lw=2, label="|M|^2(c) (normalized)", zorder=4)

    ax0.set_ylabel("density")
    ax0.set_title(f"Shape check (cuts={'ON' if use_cuts else 'OFF'}) | Ecm={Ecm:.0f} GeV")
    ax0.legend(frameon=False)

    # Option 2: expand y-range so error bars become visible
    ymax = np.nanmax(np.concatenate([dens_lhe + se_lhe, dens_nf + se_nf, dens_m2]))
    if np.isfinite(ymax) and ymax > 0:
        ax0.set_ylim(0.0, 1.35 * ymax)

    # Panel 2: ratio with band
    ax1.axhline(1.0, lw=1)
    ax1.errorbar(centers, ratio, yerr=ratio_se, fmt="o", ms=3, capsize=2, elinewidth=1)
    ax1.set_ylabel("NF / MG5")
    # autoscale ratio y
    finite = np.isfinite(ratio) & np.isfinite(ratio_se)
    if np.any(finite):
        k = 3.0
        lo = np.nanmin(ratio[finite] - k * ratio_se[finite])
        hi = np.nanmax(ratio[finite] + k * ratio_se[finite])
        pad = 0.05 * (hi - lo + 1e-12)
        ax1.set_ylim(lo - pad, hi + pad)
    else:
        ax1.set_ylim(0.95, 1.05)

    # Panel 3: pull
    ax2.axhline(0.0, lw=1)
    ax2.plot(centers, pull, marker="o", ms=3, lw=0)
    ax2.set_ylabel("pull")
    ax2.set_xlabel(r"$c=\cos\theta$")
    pull_text = r"$\mathrm{pull}_b=\dfrac{\hat f^{\mathrm{NF}}_b-\hat f^{\mathrm{MG}}_b}{\sqrt{\mathrm{SE}(\hat f^{\mathrm{NF}}_b)^2+\mathrm{SE}(\hat f^{\mathrm{MG}}_b)^2}}$"
    ax2.text(
        0.02, 0.95, pull_text,
        transform=ax2.transAxes,
        va="top", ha="left",
        fontsize=5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, edgecolor="none")
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lhe", required=True, type=str, help="path to unweighted_events.lhe or .lhe.gz")
    ap.add_argument("--model", required=True, type=str, help="path to NF checkpoint (.pt)")
    ap.add_argument("--Ecm", type=float, default=1000.0)
    ap.add_argument("--mZ", type=float, default=MZ_DEFAULT)
    ap.add_argument("--N", type=int, default=20000, help="MC/NF sample size for sigma estimators")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--outdir", type=str, default="nfmodel/plots/mg5_mc_nf")
    ap.add_argument("--no_cuts", action="store_true", help="disable passes_cuts in our M2 evaluation")
    ap.add_argument("--save_json", action="store_true", help="also save a summary JSON")
    args = ap.parse_args()

    use_cuts = (not args.no_cuts)

    root = project_root()
    lhe_path = Path(args.lhe).expanduser()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    Ecm = float(args.Ecm)
    mZ = float(args.mZ)
    N = int(args.N)
    seed = int(args.seed)

    # ----- load NF -----
    device = "cpu"
    ckpt = torch.load(args.model, map_location=device)
    n_blocks = int(ckpt.get("n_blocks", 8))
    hidden = int(ckpt.get("hidden", 16))
    permute = ckpt.get("permute", "reverse")
    flow_seed = int(ckpt.get("seed", 0))

    flow = ZGCosthFlow(n_blocks=n_blocks, hidden=hidden, permute=permute, seed=flow_seed).to(device)
    flow.load_state_dict(ckpt["state_dict"])
    flow.eval()

    # ----- MG5 sigma (pb) from LHE header -----
    sigma_mg5 = parse_mg5_sigma_from_lhe_header(lhe_path)

    # ----- baseline sigma -----
    sigma_mc, dsigma_mc = sigma_from_uniform_costh(Ecm, mZ, N=N, seed=seed, use_cuts=use_cuts)

    # ----- NF sigma -----
    sigma_nf, dsigma_nf = sigma_from_nf(flow, Ecm, mZ, N=N, seed=seed + 123, use_cuts=use_cuts)

    print("\n=== Cross sections (pb) ===")
    if sigma_mg5 is not None:
        print(f"[MG5 LHE header] sigma = {sigma_mg5:.6f} pb")
    else:
        print("[MG5 LHE header] sigma = (not found in header)")

    print(f"[Uniform MC]     sigma = {sigma_mc:.6f} ± {dsigma_mc:.6f} pb   (N={N})")
    print(f"[NF importance]  sigma = {sigma_nf:.6f} ± {dsigma_nf:.6f} pb   (N={N})")

    vr = (dsigma_mc / dsigma_nf) ** 2 if dsigma_nf > 0 else float("inf")
    print(f"[Variance reduction] ≈ {vr:.2f}×  (based on sigma errors)")

    if sigma_mg5 is not None:
        diff_mc = sigma_mc - sigma_mg5
        diff_nf = sigma_nf - sigma_mg5
        print("\n=== Differences vs MG5 (pb) ===")
        print(f"[Uniform MC - MG5]  {diff_mc:+.6f} pb  (~{diff_mc/(dsigma_mc+1e-30):.2f}σ of MC)")
        print(f"[NF - MG5]          {diff_nf:+.6f} pb  (~{diff_nf/(dsigma_nf+1e-30):.2f}σ of NF)")

    # ----- shape plot -----
    shape_png = outdir / "shape_costh_mg5_m2_nf.png"
    make_shape_plot(
        lhe_path=lhe_path,
        flow=flow,
        Ecm=Ecm,
        mZ=mZ,
        bins=args.bins,
        out_png=shape_png,
        seed=seed,
        use_cuts=use_cuts,
    )
    print(f"\n[saved] shape plot: {shape_png}")

    # ----- optional JSON -----
    if args.save_json:
        summary = {
            "Ecm": Ecm,
            "mZ": mZ,
            "cuts": bool(use_cuts),
            "N": N,
            "seed": seed,
            "sigma_mg5_pb": float(sigma_mg5) if sigma_mg5 is not None else None,
            "sigma_uniform_pb": float(sigma_mc),
            "dsigma_uniform_pb": float(dsigma_mc),
            "sigma_nf_pb": float(sigma_nf),
            "dsigma_nf_pb": float(dsigma_nf),
            "variance_reduction": float(vr),
            "shape_plot": str(shape_png),
        }
        out_json = outdir / "summary_mg5_mc_nf.json"
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[saved] summary json: {out_json}")


if __name__ == "__main__":
    main()