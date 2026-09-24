"""Stage 3c: Tobit stabilising-penalty sweep (Section 6.1, Appendix A).

Refits the Tobit specification on one synthetic draw (holdout 2019, admitted 40%, seed 0, fixed
standardisation; latent series G + S as in run_recovery.py) with the penalty weight lambda in
{0.1, 1, 10, 100} and records the correction defined as in Table 7: the Tobit bias relative to the
latent series less the censoring-blind bias on the same draw. Writes results/tobit_penalty_sweep.csv.
Runtime about 2-3 minutes on 8 CPU threads.
"""
import csv, os, sys, functools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recovery
from recovery import FEATS, standardise, fit_predict, impose_censoring, ALPHA_MIN
from run_recovery import load_panel

HOLD, ALPHA, SEED, MODE = 2019, 0.40, 0, "fixed"
LAMBDAS = (0.1, 1.0, 10.0, 100.0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dates, P = load_panel(os.path.join(here, "results", "panel_daily.csv"))
    pool = np.load(os.path.join(here, "results", "depth_pool.npy"))
    yr = dates.astype("datetime64[Y]").astype(int) + 1970
    era = (yr >= 2016) & (yr <= 2021)
    X_all = np.column_stack([P[c] for c in FEATS])
    Dlat, trend = P["G"] + P["S"], P["trend"]
    yr_era, X_era, D_era, t_era = yr[era], X_all[era], Dlat[era], trend[era]
    rng = np.random.default_rng(1000 * SEED + HOLD)
    G, S, Dt, cens = impose_censoring(D_era, pool, ALPHA, rng)
    lo, hi = np.log(Dt), np.log(G + S / ALPHA_MIN)
    itr, ite = yr_era < HOLD, yr_era == HOLD
    Xtr, Xte = standardise(X_era[itr], X_era[ite], MODE)
    ttr, tte = t_era[itr], t_era[ite]; tm, ts = ttr.mean(), ttr.std()
    true_supp = 100.0 * (D_era[ite].sum() - Dt[ite].sum()) / D_era[ite].sum()
    args = (Xtr, (ttr - tm) / ts, lo[itr], lo[itr], hi[itr], cens[itr], Xte, (tte - tm) / ts, SEED)
    Dstar_te = D_era[ite].sum()
    def bias(mu): return 100.0 * (np.exp(np.clip(mu, -50, 50)).sum() - Dstar_te) / Dstar_te
    mu_b, _ = fit_predict("blind", *args); blind_bias = bias(mu_b)
    print(f"blind bias {blind_bias:+.2f} pp; true suppression {true_supp:.2f} pp", flush=True)
    base = recovery.nll_tobit
    rows = []
    for lam in LAMBDAS:
        recovery.SPECS["tobit"] = functools.partial(base, lam=lam)
        mu, nll = fit_predict("tobit", *args)
        corr = bias(mu) - blind_bias                           # Table 7 definition
        rows.append((lam, round(corr, 2), round(true_supp, 2), round(blind_bias, 2)))
        print(f"lambda={lam:g}: tobit correction {corr:+.2f} pp, nll {nll:.4f}", flush=True)
    recovery.SPECS["tobit"] = base
    with open(os.path.join(here, "results", "tobit_penalty_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lambda", "tobit_correction_pp", "true_supp_pp", "blind_bias_pp"]); w.writerows(rows)


if __name__ == "__main__":
    main()
