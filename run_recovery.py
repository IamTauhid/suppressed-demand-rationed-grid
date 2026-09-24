"""Driver for the controlled synthetic-censoring recovery experiment.

Grid: holdout in {2019,2020,2021} x admitted in {40,60,100}% x seed in {0,1,2}
      x spec in {blind, tobit, icg}, all under the fixed standardisation.
Plus: the 2020 cut re-run under the legacy (sd + 1e-8) standardisation, to test
      the manuscript's claim that all three specifications failed there.

Writes recovery_raw.csv (one row per fit).
"""
import csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recovery import FEATS, standardise, fit_predict, impose_censoring, ALPHA_MIN

HOLDOUTS = (2019, 2020, 2021)
ADMITTED = (0.40, 0.60, 1.00)
SEEDS = (0, 1, 2)
SPECS = ("blind", "tobit", "icg")


def load_panel(path="panel_daily.csv"):
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        names = rd.fieldnames
    dates = np.array([r["date"][:10] for r in rows], dtype="datetime64[D]")
    conv = lambda v: 1.0 if v == "True" else (0.0 if v == "False" else (float("nan") if v == "" else float(v)))
    return dates, {c: np.array([conv(r[c]) for r in rows]) for c in names if c != "date"}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dates, P = load_panel(os.path.join(here, "results", "panel_daily.csv"))
    pool = np.load(os.path.join(here, "results", "depth_pool.npy"))
    yr = dates.astype("datetime64[Y]").astype(int) + 1970
    era = (yr >= 2016) & (yr <= 2021)
    X_all = np.column_stack([P[c] for c in FEATS])
    # Latent series for the synthetic experiment: the identity-generated total G + S
    # (served energy plus admitted shortfall), NOT the published demand column. In the
    # 2016-2021 era used here the two agree to within 1 MW in 99.8% of hours; using
    # G + S makes the driver match the argument of Section 3.3 exactly.
    Dlat, trend = P["G"] + P["S"], P["trend"]
    yr_era, X_era, D_era, t_era = yr[era], X_all[era], Dlat[era], trend[era]

    def build(hold, alpha, seed, mode):
        # one synthetic censoring realisation per (holdout, seed), shared across
        # admitted levels and specifications so comparisons are paired
        rng = np.random.default_rng(1000 * seed + hold)
        G, S, Dt, cens = impose_censoring(D_era, pool, alpha, rng)
        lo = np.log(Dt)
        hi = np.log(G + S / ALPHA_MIN)
        itr, ite = yr_era < hold, yr_era == hold
        Xtr, Xte = standardise(X_era[itr], X_era[ite], mode)
        ttr, tte = t_era[itr], t_era[ite]
        tm, ts = ttr.mean(), ttr.std()
        return dict(Xtr=Xtr, ttr=(ttr - tm) / ts, y=lo[itr], lo=lo[itr], hi=hi[itr],
                    cens=cens[itr], Xte=Xte, tte=(tte - tm) / ts,
                    Dstar_te=D_era[ite], Dpub_te=Dt[ite], n_tr=int(itr.sum()),
                    n_te=int(ite.sum()), n_cens_tr=int(cens[itr].sum()))

    jobs = [(h, a, s, "fixed") for h in HOLDOUTS for a in ADMITTED for s in SEEDS]
    jobs += [(2020, a, s, "legacy") for a in ADMITTED for s in SEEDS]

    out = open(os.path.join(here, "results", "recovery_raw.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["holdout", "admitted", "seed", "mode", "spec", "n_train", "n_test",
                "n_cens_train", "true_supp_pp", "bias_pp", "final_nll", "diverged",
                "max_abs_std_test", "secs"])
    t_start = time.time()
    for k, (hold, alpha, seed, mode) in enumerate(jobs, 1):
        d = build(hold, alpha, seed, mode)
        true_supp = 100.0 * (d["Dstar_te"].sum() - d["Dpub_te"].sum()) / d["Dstar_te"].sum()
        mx = float(np.abs(d["Xte"]).max())
        for spec in SPECS:
            t0 = time.time()
            mu, nll = fit_predict(spec, d["Xtr"], d["ttr"], d["y"], d["lo"], d["hi"],
                                  d["cens"], d["Xte"], d["tte"], seed)
            if mu is None or not np.all(np.isfinite(mu)):
                bias, div = float("nan"), 1
            else:
                pred = np.exp(np.clip(mu, -50, 50)).sum()
                bias = 100.0 * (pred - d["Dstar_te"].sum()) / d["Dstar_te"].sum()
                div = int(abs(bias) > 100.0 or not np.isfinite(nll))
            w.writerow([hold, alpha, seed, mode, spec, d["n_tr"], d["n_te"],
                        d["n_cens_tr"], f"{true_supp:.4f}", f"{bias:.4f}",
                        f"{nll:.6f}", div, f"{mx:.6g}", f"{time.time()-t0:.1f}"])
            out.flush()
        print(f"[{k}/{len(jobs)}] {hold} a={alpha:.2f} s={seed} {mode} "
              f"({time.time()-t_start:.0f}s elapsed)", flush=True)
    out.close()


if __name__ == "__main__":
    main()
