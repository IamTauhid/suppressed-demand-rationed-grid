"""
Structural OLS counterfactual (paper Section 6.2 / Route 2) and its validation
(Section 6.4 rolling-origin placebo; Section 6.5 activity controls).

Outcome:  log daily served energy, fitted on the uncensored 2016-2021 era.
Design:   linear trend, quadratic humidity-adjusted cooling response, two annual
          harmonic pairs, day-of-week terms, plus an optional activity control:
              none     -- no control
              +growth  -- annual real GDP growth (%), constant within year
              +level   -- log real GDP level, constant within year
Gap:      100 * (sum exp(pred) - sum G) / sum G over a calendar year, in pp of served
          energy.  In the uncensored era served energy equals published demand to
          within rounding, so the fit is to G.
"""
import numpy as np
import pandas as pd

BASE = ["trend", "cdd", "cdd2", "cdd_rh", "sin1", "cos1", "sin2", "cos2",
        "dow1", "dow2", "dow3", "dow4", "dow5", "dow6"]
CONTROLS = {"none": [], "+growth": ["gdp_growth"], "+level": ["log_gdp"]}
FIT_YEARS = (2016, 2021)
CRISIS_YEARS = (2022, 2023, 2024, 2025)


def add_macro(panel, gdp):
    """Attach calendar-year GDP growth and log real GDP level to a daily panel."""
    p = panel.copy()
    yr = p.index.year
    p["gdp_growth"] = pd.Series(yr, index=p.index).map(gdp["gdp_growth_pct"]).values
    p["log_gdp"] = np.log(pd.Series(yr, index=p.index).map(gdp["gdp_const_usd"]).values)
    return p


def _X(p, spec):
    cols = BASE + CONTROLS[spec]
    X = p[cols].values.astype(float)
    return np.column_stack([np.ones(len(X)), X]), cols


def fit(p_train, spec):
    X, cols = _X(p_train, spec)
    y = np.log(p_train["G"].values)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return {"beta": beta, "cols": cols, "resid": resid, "y": y, "X": X,
            "r2": 1 - resid.var() / y.var(),
            "trend_pct_per_yr": 100 * (np.exp(beta[1]) - 1),
            "cond": np.linalg.cond(X)}


def predict(model, p):
    X, _ = _X(p, _spec_from_cols(model["cols"]))
    return np.exp(X @ model["beta"])


def _spec_from_cols(cols):
    for k, v in CONTROLS.items():
        if cols == BASE + v:
            return k
    raise ValueError(cols)


def annual_gap(model, p, years):
    pred = predict(model, p)
    out = {}
    for y in years:
        m = p.index.year == y
        if m.sum() == 0:
            continue
        out[y] = 100 * (pred[m].sum() - p["G"].values[m].sum()) / p["G"].values[m].sum()
    return out


def in_sample_stats(model, p_train):
    """Two different 'in-sample sd' definitions, both reported so the 0.66 / 1.43
    discrepancy can be traced to a definition rather than a specification."""
    gaps = annual_gap(model, p_train, range(FIT_YEARS[0], FIT_YEARS[1] + 1))
    g = np.array(list(gaps.values()))
    return {"annual_gap_sd_pp": g.std(ddof=1),          # sd of the 6 fitted-year gaps
            "annual_gap_sd_pp_ddof0": g.std(ddof=0),
            "daily_resid_sd_pp": 100 * model["resid"].std(ddof=1),   # daily log-residual sd
            "r2": model["r2"], "trend_pct_per_yr": model["trend_pct_per_yr"],
            "cond": model["cond"], "n_train": len(p_train), "fitted_gaps": gaps}


def rolling_placebo(p, spec, origins=(2017, 2018, 2019, 2020), max_h=3,
                    clean_sg_pct=0.03, first_year=2016):
    """Refit on first_year..origin, evaluate h years ahead on years with S/G below
    clean_sg_pct (approx. zero true suppression)."""
    rows = []
    for o in origins:
        tr = p[(p.index.year >= first_year) & (p.index.year <= o)]
        m = fit(tr, spec)
        for h in range(1, max_h + 1):
            tgt = o + h
            te = p[p.index.year == tgt]
            if len(te) == 0 or tgt > FIT_YEARS[1]:
                continue
            sg = 100 * te["S"].sum() / te["G"].sum()
            if sg >= clean_sg_pct:
                continue
            gap = annual_gap(m, te, [tgt])[tgt]
            rows.append({"spec": spec, "origin": o, "n_train": len(tr), "h": h,
                         "target": tgt, "S_over_G_pct": sg, "gap_pp": gap,
                         "stratum": "covid_break" if tgt in (2020, 2021) else "normal"})
    return pd.DataFrame(rows)


def rmse(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def block_bootstrap(p_train, p_eval, spec, years, n_rep=2000, block=28, seed=0):
    """Moving-block residual bootstrap: resample training residuals in blocks,
    rebuild y* = fitted + resid*, refit, re-extrapolate. Returns gaps[rep, year]."""
    rng = np.random.default_rng(seed)
    m = fit(p_train, spec)
    fitted = m["X"] @ m["beta"]
    r = m["resid"]
    n = len(r)
    nblk = int(np.ceil(n / block))
    Xe, _ = _X(p_eval, spec)
    G = p_eval["G"].values
    yrs = p_eval.index.year.values
    out = np.empty((n_rep, len(years)))
    for k in range(n_rep):
        starts = rng.integers(0, n - block + 1, size=nblk)
        rs = np.concatenate([r[s:s + block] for s in starts])[:n]
        ystar = fitted + rs
        b, *_ = np.linalg.lstsq(m["X"], ystar, rcond=None)
        pred = np.exp(Xe @ b)
        for j, y in enumerate(years):
            mm = yrs == y
            out[k, j] = 100 * (pred[mm].sum() - G[mm].sum()) / G[mm].sum()
    return out
