"""
Stage 4: structural counterfactual, rolling-origin placebo, block bootstrap, welfare
arithmetic and Table 3 (Sections 3.3, 6.2-6.7, 7).

Inputs   results/panel_daily.csv, results/hourly_clean.csv, data/pgcb_V1_raw.csv
Outputs  results/spec_comparison.csv      one row per specification (gaps, in-sample, OOS RMSE)
         results/placebo_detail.csv       every rolling-origin evaluation, three specifications
         results/fitted_year_gaps.csv     2016-2021 annual gaps per specification
         results/structural.csv           gaps, bootstrap p05/p95, bounds, TWh, m$ (3 VOLLs)
         results/table3_excess_clean.csv  excess hours on the cleaned record
         results/alpha_sweep.csv          upper bound vs admission floor, with m$ at VOLL 90
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import structural as S

DATA = os.path.join(HERE, "data"); RES = os.path.join(HERE, "results")
SPECS = ["none", "+growth", "+level"]
VOLL = (40, 90, 180); FX = 122.0; ALPHA_MIN = 0.30
S.BASE[:] = ["trend", "cdd", "cdd2", "cdd_rh", "sin1", "cos1", "sin2", "cos2",
             "dow1", "dow2", "dow3", "dow4", "dow5", "dow6", "covid"]


def usd(pp, G_MWh, v=90):
    return pp / 100 * G_MWh * 1e3 * v / FX / 1e6


def main():
    P = pd.read_csv(os.path.join(RES, "panel_daily.csv"), index_col="date", parse_dates=True)
    tr = P[(P.index.year >= 2016) & (P.index.year <= 2021)]
    ev = P[P.index.year >= 2022]
    yG = P.groupby(P.index.year).G.sum(); yS = P.groupby(P.index.year).S.sum()

    rows, plac, fitted = [], [], {}
    for spec in SPECS:
        m = S.fit(tr, spec); st = S.in_sample_stats(m, tr)
        gaps = S.annual_gap(m, P, S.CRISIS_YEARS)
        pl = S.rolling_placebo(P, spec); plac.append(pl)
        fitted[spec] = st["fitted_gaps"]
        rows.append({"spec": spec, **{f"gap_{y}": gaps[y] for y in S.CRISIS_YEARS},
                     "insample_annual_gap_sd": st["annual_gap_sd_pp"], "insample_daily_resid_sd": st["daily_resid_sd_pp"],
                     "R2": st["r2"], "trend_pct_yr": st["trend_pct_per_yr"], "cond": st["cond"], "n_train": len(tr),
                     "oos_rmse_pooled": S.rmse(pl.gap_pp), "oos_rmse_normal": S.rmse(pl[pl.stratum == "normal"].gap_pp),
                     "oos_rmse_covid": S.rmse(pl[pl.stratum == "covid_break"].gap_pp),
                     "oos_mean_normal": pl[pl.stratum == "normal"].gap_pp.mean(),
                     "oos_mean_covid": pl[pl.stratum == "covid_break"].gap_pp.mean(), "oos_mean_pooled": pl.gap_pp.mean(),
                     "oos_max_normal": pl[pl.stratum == "normal"].gap_pp.abs().max(),
                     "oos_max_covid": pl[pl.stratum == "covid_break"].gap_pp.abs().max(), "oos_max_pooled": pl.gap_pp.abs().max()})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "spec_comparison.csv"), index=False)
    pd.concat(plac).to_csv(os.path.join(RES, "placebo_detail.csv"), index=False)
    pd.DataFrame(fitted).rename_axis("year").to_csv(os.path.join(RES, "fitted_year_gaps.csv"))

    out = []
    for spec in SPECS:
        B = S.block_bootstrap(tr, ev, spec, list(S.CRISIS_YEARS), n_rep=2000, block=28, seed=0)
        m = S.fit(tr, spec); g = S.annual_gap(m, P, S.CRISIS_YEARS)
        for j, y in enumerate(S.CRISIS_YEARS):
            lo, hi = np.percentile(B[:, j], [5, 95])
            bl, bu = 100 * yS[y] / yG[y], 100 * yS[y] / (ALPHA_MIN * yG[y])
            out.append({"spec": spec, "year": y, "gap_pp": g[y], "boot_p05": lo, "boot_p95": hi,
                        "bound_lower_pp": bl, "bound_upper_pp": bu, "inside_bound": bool(bl <= g[y] <= bu),
                        "G_TWh": yG[y] / 1e6, "unserved_TWh": g[y] / 100 * yG[y] / 1e6,
                        "bound_lower_TWh": bl / 100 * yG[y] / 1e6, "bound_upper_TWh": bu / 100 * yG[y] / 1e6,
                        **{f"cost_mUSD_voll{v}": usd(g[y], yG[y], v) for v in VOLL},
                        **{f"bound_lower_mUSD_voll{v}": usd(bl, yG[y], v) for v in VOLL},
                        **{f"bound_upper_mUSD_voll{v}": usd(bu, yG[y], v) for v in VOLL},
                        "boot_p05_mUSD_voll90": usd(lo, yG[y]), "boot_p95_mUSD_voll90": usd(hi, yG[y])})
    W = pd.DataFrame(out); W.to_csv(os.path.join(RES, "structural.csv"), index=False)

    # alpha sweep (Appendix B), in pp and in m$ at VOLL 90, from the same yS / yG
    rows = []
    for a in (0.20, 0.30, 0.50, 0.70, 1.00):
        r = {"alpha_min": a}
        for y in S.CRISIS_YEARS:
            r[f"upper_pp_{y}"] = 100 * yS[y] / (a * yG[y]); r[f"upper_mUSD_{y}"] = usd(r[f"upper_pp_{y}"], yG[y])
        rows.append(r)
    A = pd.DataFrame(rows)
    for y in S.CRISIS_YEARS:
        A[f"lower_pp_{y}"] = 100 * yS[y] / yG[y]; A[f"lower_mUSD_{y}"] = usd(A[f"lower_pp_{y}"], yG[y])
    A.to_csv(os.path.join(RES, "alpha_sweep.csv"), index=False)

    # Table 3 on the cleaned hourly record (pre-interpolation observed hours only)
    h0 = pd.read_csv(os.path.join(RES, "hourly_clean.csv"), index_col="dt", parse_dates=True).dropna()
    h0["excess"] = h0.demand_mw - h0.generation_mw - h0.load_shedding
    t3 = []
    for y in (2022, 2023, 2024, 2025):
        g_ = h0[h0.index.year == y]; ex = g_[g_.excess > 0]
        t3.append({"year": y, "hours_with_excess_pct": 100 * len(ex) / len(g_), "median_excess_MW": ex.excess.median(),
                   "excess_pct_of_generation": 100 * ex.excess.sum() / g_.generation_mw.sum(), "excess_GWh": ex.excess.sum() / 1e3,
                   "excess_pct_of_shed_same_hours": 100 * ex.excess.sum() / ex.load_shedding.sum(),
                   "identity_exact_pct": 100 * (g_.excess == 0).mean()})
    T3 = pd.DataFrame(t3); T3.to_csv(os.path.join(RES, "table3_excess_clean.csv"), index=False)
    tt = h0[(h0.index.year >= 2022) & (h0.index.year <= 2024)]; ext = tt[tt.excess > 0]
    print("excess 2022-2024 clean: %.1f GWh = %.1f%% of shedding in those hours" % (ext.excess.sum() / 1e3, 100 * ext.excess.sum() / ext.load_shedding.sum()))
    print(W[["spec", "year", "gap_pp", "boot_p05", "boot_p95", "inside_bound", "unserved_TWh", "cost_mUSD_voll90"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
