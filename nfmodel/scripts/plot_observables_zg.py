#!/usr/bin/env python3
"""
nfmodel/scripts/plot_observables_zg.py

Compare LO parton-level observable *shapes* between:
  (A) NF-generated events (cosθ sampled from qθ, then *reweighted* by w ∝ |M|^2 / qθ(c))
  (B) MG5 unweighted LHE events (already distributed like LO dσ)

Outputs:
  - zg_observables_compare.png  (2x2 densities with error bars)
  - zg_observables_ratio.png    (2x2 NF/LHE ratio with error bars)
  - zg_observables_pull.png     (2x2 pulls)

Usage example:
  python -m nfmodel.scripts.plot_observables_zg \
    --model nfmodel/models/zg_costh_flow.pt \
    --lhe  /abs/path/to/unweighted_events.lhe.gz \
    --Ecm 1000 --N 200000 \
    --jet_pt_min 20 --jet_eta_max 5 \
    --ht_mode jets \
    --use_cuts \
    --device cpu
"""

import math
import gzip
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from nfmodel.flows.zg_costh_flow import ZGCosthFlow
from nfmodel.physics.zg_phase_space import build_event_zg, MZ_DEFAULT
from nfmodel.physics.zg_me import me2
from nfmodel.physics.cuts import passes_cuts

PHI0 = 0.0  # LO rotational symmetry


# -----------------------
# basic kinematics
# -----------------------
def pt(px: float, py: float) -> float:
    return float(math.sqrt(px * px + py * py))


def eta_from_p(px: float, py: float, pz: float) -> float:
    p = math.sqrt(px * px + py * py + pz * pz) + 1e-30
    return 0.5 * math.log((p + pz) / (p - pz + 1e-30))


def is_parton_jet(pdgid: int) -> bool:
    a = abs(int(pdgid))
    return (1 <= a <= 6) or (a == 21)  # quarks or gluon


# -----------------------
# file I/O
# -----------------------
def open_maybe_gz(path: Path):
    path = Path(path)
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


# -----------------------
# histogram + uncertainties
# -----------------------
def hist_density_with_se_unweighted(x: np.ndarray, bins: np.ndarray):
    """
    Unweighted LHE events: treat counts as multinomial.
    For each bin b:
      p_hat = n_b / N
      dens_hat = p_hat / Δ
      SE(dens_hat) ≈ sqrt(p_hat(1-p_hat)/N) / Δ
    """
    x = np.asarray(x)
    N = len(x)
    h, edges = np.histogram(x, bins=bins)
    bw = np.diff(edges)

    p_hat = h / max(N, 1)
    dens = p_hat / bw

    se_p = np.sqrt(np.maximum(p_hat * (1.0 - p_hat), 0.0) / max(N, 1))
    se_dens = se_p / bw

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, dens, se_dens


def hist_density_with_se_snis(x: np.ndarray, w: np.ndarray, bins: np.ndarray):
    """
    Self-normalized IS (NF shapes):
      p_hat(b) = S_b / S,  S_b = sum_i w_i 1_i(b),  S = sum_i w_i

    Delta-method variance estimator:
      Var(p_hat(b)) ≈ (1/S^2) * (1/(N-1)) * sum_i [ w_i (1_i(b) - p_hat(b)) ]^2
    density = p_hat / Δ
    """
    x = np.asarray(x)
    w = np.asarray(w, dtype=np.float64)
    N = len(x)

    edges = np.asarray(bins, dtype=np.float64)
    bw = np.diff(edges)
    B = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    dens = np.zeros(B, dtype=np.float64)
    se_dens = np.zeros(B, dtype=np.float64)

    S = float(np.sum(w))
    if (S <= 0.0) or (N <= 1):
        return centers, dens, se_dens

    bin_idx = np.digitize(x, edges) - 1  # 0..B-1 (outside -> <0 or >=B)

    for b in range(B):
        I = (bin_idx == b).astype(np.float64)
        Sb = float(np.sum(w[I > 0.5]))
        p_hat = Sb / S

        g = w * (I - p_hat)
        var_p = (np.sum(g * g) / (N - 1)) / (S * S)
        se_p = math.sqrt(max(var_p, 0.0))

        dens[b] = p_hat / bw[b]
        se_dens[b] = se_p / bw[b]

    return centers, dens, se_dens


def compute_ratio_pull(x_nf, w_nf, x_lhe, bins):
    c, d_nf, se_nf = hist_density_with_se_snis(x_nf, w_nf, bins)
    _, d_lhe, se_lhe = hist_density_with_se_unweighted(x_lhe, bins)

    ratio = np.divide(d_nf, d_lhe, out=np.full_like(d_nf, np.nan), where=(d_lhe > 0))

    ratio_se = np.full_like(ratio, np.nan)
    mask = (d_nf > 0) & (d_lhe > 0)
    ratio_se[mask] = ratio[mask] * np.sqrt(
        (se_nf[mask] / d_nf[mask])**2 + (se_lhe[mask] / d_lhe[mask])**2
    )

    denom = np.sqrt(se_nf**2 + se_lhe**2)
    pull = np.divide(d_nf - d_lhe, denom, out=np.full_like(d_nf, np.nan), where=(denom > 0))

    return c, ratio, ratio_se, pull


# -----------------------
# LHE parsing -> observables
# -----------------------
def read_lhe_observables(
    lhe_path: Path,
    jet_pt_min: float,
    jet_eta_max: float,
    ht_mode: str = "jets",
):
    lhe_path = Path(lhe_path).expanduser()
    if not lhe_path.exists():
        raise FileNotFoundError(f"LHE not found: {lhe_path}")

    HT_list, ptZ_list, njet_list, leadpt_list = [], [], [], []

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

                Z_p = None
                jets_pt = []

                for pl in parts:
                    cols = pl.split()
                    if len(cols) < 10:
                        continue
                    pid = int(cols[0])
                    ist = int(cols[1])
                    px, py, pz, E = map(float, cols[6:10])

                    if ist != 1:
                        continue

                    if pid == 23:
                        Z_p = (E, px, py, pz)

                    if is_parton_jet(pid):
                        ptj = pt(px, py)
                        etaj = eta_from_p(px, py, pz)
                        if (ptj >= jet_pt_min) and (abs(etaj) <= jet_eta_max):
                            jets_pt.append(ptj)

                if Z_p is None:
                    in_event = False
                    continue

                ptZ = pt(Z_p[1], Z_p[2])
                njet = len(jets_pt)
                lead_pt = max(jets_pt) if njet > 0 else 0.0

                if ht_mode == "jets":
                    HT = float(sum(jets_pt))
                elif ht_mode == "zjets":
                    HT = float(ptZ + sum(jets_pt))
                else:
                    raise ValueError("ht_mode must be 'jets' or 'zjets'")

                HT_list.append(HT)
                ptZ_list.append(ptZ)
                njet_list.append(njet)
                leadpt_list.append(lead_pt)

                in_event = False
                continue

            if s and (not s.startswith("#")):
                event_lines.append(line)

    HT = np.asarray(HT_list, dtype=np.float64)
    ptZ_arr = np.asarray(ptZ_list, dtype=np.float64)
    njet = np.asarray(njet_list, dtype=np.int64)
    leadpt = np.asarray(leadpt_list, dtype=np.float64)

    if len(HT) == 0:
        raise RuntimeError("LHE: no usable events found.")

    w = np.ones(len(HT), dtype=np.float64)
    w /= w.sum()
    return w, HT, ptZ_arr, njet, leadpt, len(HT)


# -----------------------
# NF generation -> observables
# -----------------------
def load_flow(model_path: Path, device: str):
    ckpt = torch.load(model_path, map_location=device)
    n_blocks = int(ckpt.get("n_blocks", 8))
    hidden = int(ckpt.get("hidden", 16))
    permute = ckpt.get("permute", "reverse")
    seed = int(ckpt.get("seed", 0))

    flow = ZGCosthFlow(n_blocks=n_blocks, hidden=hidden, permute=permute, seed=seed).to(device)
    flow.load_state_dict(ckpt["state_dict"])
    flow.eval()
    return flow, ckpt


def nf_generate_observables(
    flow: ZGCosthFlow,
    Ecm: float,
    mZ: float,
    N: int,
    jet_pt_min: float,
    jet_eta_max: float,
    ht_mode: str = "jets",
    use_cuts: bool = False,
):
    device = next(flow.parameters()).device
    with torch.no_grad():
        c = flow.sample_c(N, device=device)
        logq = flow.logprob_c(c)

    c_np = c.detach().cpu().numpy()
    q_np = np.exp(logq.detach().cpu().numpy())

    w_list, HT_list, ptZ_list, njet_list, leadpt_list = [], [], [], [], []

    for i in range(N):
        p_all = build_event_zg(Ecm, float(c_np[i]), PHI0, mZ=mZ)

        if use_cuts and (not passes_cuts(p_all)):
            continue

        try:
            m2 = float(me2(p_all))
        except Exception:
            continue
        if (not np.isfinite(m2)) or m2 < 0.0:
            continue

        q = float(q_np[i])
        if (not np.isfinite(q)) or q <= 0.0:
            continue

        w = m2 / q

        pZ = p_all[2]
        pg = p_all[3]

        ptZ = pt(pZ[1], pZ[2])
        ptg = pt(pg[1], pg[2])
        etag = eta_from_p(pg[1], pg[2], pg[3])

        jets_pt = []
        if (ptg >= jet_pt_min) and (abs(etag) <= jet_eta_max):
            jets_pt.append(ptg)

        njet = len(jets_pt)
        lead_pt = max(jets_pt) if njet > 0 else 0.0

        if ht_mode == "jets":
            HT = float(sum(jets_pt))
        elif ht_mode == "zjets":
            HT = float(ptZ + sum(jets_pt))
        else:
            raise ValueError("ht_mode must be 'jets' or 'zjets'")

        w_list.append(w)
        HT_list.append(HT)
        ptZ_list.append(ptZ)
        njet_list.append(njet)
        leadpt_list.append(lead_pt)

    w = np.asarray(w_list, dtype=np.float64)
    HT = np.asarray(HT_list, dtype=np.float64)
    ptZ_arr = np.asarray(ptZ_list, dtype=np.float64)
    njet = np.asarray(njet_list, dtype=np.int64)
    leadpt = np.asarray(leadpt_list, dtype=np.float64)

    if len(w) == 0:
        raise RuntimeError("NF: no events survived (cuts / me2 / q(c) issues).")

    # normalize weights for *shape* plots
    w_sum = w.sum()
    w = w / w_sum

    ess = 1.0 / (np.sum(w * w))
    return w, HT, ptZ_arr, njet, leadpt, ess, len(w)


# -----------------------
# plotting
# -----------------------
def _set_plot_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })


def plot_density_grid(out_path: Path, nf_data, lhe_data, title: str, jet_pt_min: float, jet_eta_max: float, ht_mode: str):
    w_nf, HT_nf, ptZ_nf, njet_nf, lead_nf = nf_data
    w_lhe, HT_lhe, ptZ_lhe, njet_lhe, lead_lhe = lhe_data

    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    def linbins(x1, x2, n=60):
        hi = max(50.0, np.percentile(np.concatenate([x1, x2]), 99.5))
        return np.linspace(0.0, hi, n)

    bins_ht   = linbins(HT_nf, HT_lhe)
    bins_ptz  = linbins(ptZ_nf, ptZ_lhe)
    bins_lead = linbins(lead_nf, lead_lhe)

    def panel(ax, x_nf, x_lhe, bins, xlabel, subtitle):
        c, d_nf, se_nf = hist_density_with_se_snis(x_nf, w_nf, bins)
        _, d_lhe, se_lhe = hist_density_with_se_unweighted(x_lhe, bins)
        ax.errorbar(c, d_lhe, yerr=se_lhe, fmt="o", ms=2.5, capsize=2, label="MG5 LHE")
        ax.errorbar(c, d_nf,  yerr=se_nf,  fmt="o", ms=2.5, capsize=2, label="NF (reweighted)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_title(subtitle)
        ax.legend(frameon=False)

    panel(axes[0, 0], HT_nf,   HT_lhe,   bins_ht,   "HT [GeV]",
          f"HT ({'sum jet pT' if ht_mode=='jets' else 'pT(Z)+sum jet pT'})")
    panel(axes[0, 1], ptZ_nf,  ptZ_lhe,  bins_ptz,  "pT(Z) [GeV]", "Z transverse momentum")
    panel(axes[1, 1], lead_nf, lead_lhe, bins_lead, "leading jet pT [GeV]", "Leading jet pT")

    # Njets (probabilities)
    ax = axes[1, 0]
    bins_n = np.arange(-0.5, 6.5, 1.0)
    h_nf, e = np.histogram(njet_nf, bins=bins_n, weights=w_nf)
    h_lhe, _ = np.histogram(njet_lhe, bins=bins_n, weights=w_lhe)
    centers = 0.5 * (e[:-1] + e[1:])
    ax.bar(centers - 0.2, h_lhe, width=0.4, label="MG5 LHE")
    ax.bar(centers + 0.2, h_nf,  width=0.4, label="NF (reweighted)")
    ax.set_xlabel("N jets (parton jets)")
    ax.set_ylabel("probability")
    ax.set_title(f"Jet multiplicity (pT>{jet_pt_min:.0f} GeV, |eta|<{jet_eta_max:.1f})")
    ax.legend(frameon=False)

    fig.suptitle(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_ratio_grid(out_path: Path, nf_data, lhe_data, title: str, ht_mode: str):
    w_nf, HT_nf, ptZ_nf, njet_nf, lead_nf = nf_data
    w_lhe, HT_lhe, ptZ_lhe, njet_lhe, lead_lhe = lhe_data

    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    def linbins(x1, x2, n=60):
        hi = max(50.0, np.percentile(np.concatenate([x1, x2]), 99.5))
        return np.linspace(0.0, hi, n)

    bins_ht   = linbins(HT_nf, HT_lhe)
    bins_ptz  = linbins(ptZ_nf, ptZ_lhe)
    bins_lead = linbins(lead_nf, lead_lhe)

    def ratio_panel(ax, x_nf, x_lhe, bins, xlabel, subtitle):
        c, ratio, ratio_se, _ = compute_ratio_pull(x_nf, w_nf, x_lhe, bins)
        ax.errorbar(c, ratio, yerr=ratio_se, fmt="o", ms=2.5, capsize=2, label="NF/LHE")
        ax.axhline(1.0, lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NF / LHE")
        ax.set_title(subtitle)
        ax.set_ylim(0.5, 1.5)
        ax.legend(frameon=False)

    ratio_panel(axes[0, 0], HT_nf,   HT_lhe,   bins_ht,   "HT [GeV]",
                f"HT ({'jets' if ht_mode=='jets' else 'z+jets'})")
    ratio_panel(axes[0, 1], ptZ_nf,  ptZ_lhe,  bins_ptz,  "pT(Z) [GeV]", "pT(Z)")
    ratio_panel(axes[1, 1], lead_nf, lead_lhe, bins_lead, "leading jet pT [GeV]", "Leading jet pT")

    # Njets ratio
    ax = axes[1, 0]
    bins_n = np.arange(-0.5, 6.5, 1.0)
    p_nf, e = np.histogram(njet_nf, bins=bins_n, weights=w_nf)
    p_lhe, _ = np.histogram(njet_lhe, bins=bins_n, weights=w_lhe)
    centers = 0.5 * (e[:-1] + e[1:])
    ratio = np.divide(p_nf, p_lhe, out=np.full_like(p_nf, np.nan), where=(p_lhe > 0))
    ax.bar(centers, ratio, width=0.9, label="NF/LHE")
    ax.axhline(1.0, lw=1)
    ax.set_xlabel("N jets")
    ax.set_ylabel("NF / LHE")
    ax.set_title("Njets ratio")
    ax.set_ylim(0.5, 1.5)
    ax.legend(frameon=False)

    fig.suptitle(title + " (ratio)", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_pull_grid(out_path: Path, nf_data, lhe_data, title: str, ht_mode: str):
    w_nf, HT_nf, ptZ_nf, njet_nf, lead_nf = nf_data
    w_lhe, HT_lhe, ptZ_lhe, njet_lhe, lead_lhe = lhe_data

    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    def linbins(x1, x2, n=60):
        hi = max(50.0, np.percentile(np.concatenate([x1, x2]), 99.5))
        return np.linspace(0.0, hi, n)

    bins_ht   = linbins(HT_nf, HT_lhe)
    bins_ptz  = linbins(ptZ_nf, ptZ_lhe)
    bins_lead = linbins(lead_nf, lead_lhe)

    def pull_panel(ax, x_nf, x_lhe, bins, xlabel, subtitle):
        c, _, _, pull = compute_ratio_pull(x_nf, w_nf, x_lhe, bins)
        ax.step(c, pull, where="mid", label=r"pull $(\rho_{\rm NF}-\rho_{\rm LHE})/\sqrt{\sigma_{\rm NF}^2+\sigma_{\rm LHE}^2}$")
        ax.axhline(0.0, lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("pull")
        ax.set_title(subtitle)
        ax.set_ylim(-5, 5)
        ax.legend(frameon=False)

    pull_panel(axes[0, 0], HT_nf,   HT_lhe,   bins_ht,   "HT [GeV]",
               f"HT ({'jets' if ht_mode=='jets' else 'z+jets'})")
    pull_panel(axes[0, 1], ptZ_nf,  ptZ_lhe,  bins_ptz,  "pT(Z) [GeV]", "pT(Z)")
    pull_panel(axes[1, 1], lead_nf, lead_lhe, bins_lead, "leading jet pT [GeV]", "Leading jet pT")

    # Njets pull: approximate
    ax = axes[1, 0]
    bins_n = np.arange(-0.5, 6.5, 1.0)
    p_nf, e = np.histogram(njet_nf, bins=bins_n, weights=w_nf)
    p_lhe, _ = np.histogram(njet_lhe, bins=bins_n, weights=w_lhe)
    centers = 0.5 * (e[:-1] + e[1:])

    denom = np.sqrt(np.maximum(p_nf, 0) + np.maximum(p_lhe, 0))
    pull = np.divide(p_nf - p_lhe, denom, out=np.full_like(p_nf, np.nan, dtype=float), where=(denom > 0))

    ax.bar(centers, pull, width=0.9, label="pull (approx)")
    ax.axhline(0.0, lw=1)
    ax.set_xlabel("N jets")
    ax.set_ylabel("pull")
    ax.set_title("Njets pull (approx)")
    ax.set_ylim(-5, 5)
    ax.legend(frameon=False)

    fig.suptitle(title + " (pull)", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# -----------------------
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="nfmodel/models/zg_costh_flow.pt",
                    help="Path to saved NF checkpoint (.pt)")
    ap.add_argument("--lhe", type=str, required=True,
                    help="Path to MG5 unweighted_events.lhe or .lhe.gz")
    ap.add_argument("--Ecm", type=float, default=1000.0,
                    help="Partonic COM energy [GeV] for NF event construction")
    ap.add_argument("--mZ", type=float, default=MZ_DEFAULT, help="Z mass [GeV]")
    ap.add_argument("--N", type=int, default=200000, help="Number of NF samples (before filtering)")
    ap.add_argument("--outdir", type=str, default="nfmodel/plots/zg_observables",
                    help="Output directory relative to project root")
    ap.add_argument("--jet_pt_min", type=float, default=20.0, help="Jet pT threshold [GeV]")
    ap.add_argument("--jet_eta_max", type=float, default=5.0, help="Jet |eta| max")
    ap.add_argument("--ht_mode", type=str, default="jets", choices=["jets", "zjets"],
                    help="HT definition: 'jets' or 'zjets'")
    ap.add_argument("--use_cuts", action="store_true",
                    help="Apply nfmodel.physics.cuts.passes_cuts on NF events")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Device for flow evaluation (cpu/mps).")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = (project_root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    lhe_path = Path(args.lhe).expanduser().resolve()

    print(f"[compare] device={args.device}")
    print(f"[compare] model={model_path}")
    print(f"[compare] lhe={lhe_path}")

    flow, _ = load_flow(model_path, device=args.device)

    # NF
    w_nf, HT_nf, ptZ_nf, njet_nf, lead_nf, ess, kept_nf = nf_generate_observables(
        flow=flow,
        Ecm=float(args.Ecm),
        mZ=float(args.mZ),
        N=int(args.N),
        jet_pt_min=float(args.jet_pt_min),
        jet_eta_max=float(args.jet_eta_max),
        ht_mode=args.ht_mode,
        use_cuts=bool(args.use_cuts),
    )
    print(f"[NF] kept {kept_nf}/{args.N}, ESS≈{ess:.1f}")

    # LHE
    w_lhe, HT_lhe, ptZ_lhe, njet_lhe, lead_lhe, kept_lhe = read_lhe_observables(
        lhe_path,
        jet_pt_min=float(args.jet_pt_min),
        jet_eta_max=float(args.jet_eta_max),
        ht_mode=args.ht_mode,
    )
    print(f"[LHE] kept {kept_lhe} events")

    nf_data = (w_nf, HT_nf, ptZ_nf, njet_nf, lead_nf)
    lhe_data = (w_lhe, HT_lhe, ptZ_lhe, njet_lhe, lead_lhe)

    title = (f"Z+g observables | NF(reweighted) vs MG5 LHE | "
             f"Ecm={args.Ecm:.0f} GeV | jet pT>{args.jet_pt_min:.0f}, |eta|<{args.jet_eta_max:.1f} | "
             f"HT={args.ht_mode}")

    out_density = out_dir / "zg_observables_compare.png"
    out_ratio   = out_dir / "zg_observables_ratio.png"
    out_pull    = out_dir / "zg_observables_pull.png"

    plot_density_grid(out_density, nf_data, lhe_data, title, args.jet_pt_min, args.jet_eta_max, args.ht_mode)
    plot_ratio_grid(out_ratio, nf_data, lhe_data, title, args.ht_mode)
    plot_pull_grid(out_pull, nf_data, lhe_data, title, args.ht_mode)

    print(f"[saved] {out_density.resolve()}")
    print(f"[saved] {out_ratio.resolve()}")
    print(f"[saved] {out_pull.resolve()}")


if __name__ == "__main__":
    main()
