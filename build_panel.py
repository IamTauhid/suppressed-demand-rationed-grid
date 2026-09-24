"""
Stage 1: clean the raw PGCB hourly record and build the daily design matrix.

Inputs   data/pgcb_V1_raw.csv         PGCB hourly record (Mendeley doi:10.17632/vpk8spw2mm.1)
         data/nasa_power_raw.json     NASA POWER daily T2M, RH2M, T2M_MAX, T2M_MIN for the
                                      eight divisional HQs (fetch_inputs.py)
         data/wdi_bgd.json            World Bank WDI GDP growth / level / CPI (fetch_inputs.py)
Outputs  results/panel_daily.csv      one row per retained day
         results/depth_pool.npy       empirical curtailment depth S/D on shed days, 2022-2024
         results/cleaning_fidelity.csv  cleaning quantities for comparison with the manuscript
         results/hourly_clean.csv     cleaned hourly grid before interpolation (for the audit)

Cleaning rules (Section 3.1):
  * strict hourly grid: rows stamped :30 are dropped, never averaged in
  * reject hours where any source, generation, published demand or shedding exceeds
    installed capacity + 15% headroom, or where the energy balance
    |sum(sources) - generation| exceeds 10% of generation
  * time-interpolate gaps of at most 3 hours; keep days with >= 20 valid slots
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

SRC = ["gas", "liquid_fuel", "coal", "hydro", "solar", "wind",
       "india_bheramara_hvdc", "india_tripura", "india_adani", "nepal"]
CAP = {"gas": 11500, "liquid_fuel": 6300, "coal": 5200, "hydro": 230, "solar": 1200, "wind": 60,
       "india_bheramara_hvdc": 1160, "india_tripura": 160, "india_adani": 1600, "nepal": 40}
SYS_CEIL = 20000
DIV = {"Dhaka": 44215000, "Chattogram": 33202000, "Rajshahi": 20353000, "Khulna": 17416000,
       "Barishal": 9325000, "Sylhet": 11033000, "Rangpur": 17610000, "Mymensingh": 12369000}
COVID = [("2020-03-26", "2020-05-30"), ("2021-04-05", "2021-08-10")]


def main():
    raw = pd.read_csv(os.path.join(DATA, "pgcb_V1_raw.csv"))
    raw["dt"] = pd.to_datetime(raw["datetime"])
    d = raw.set_index("dt").sort_index()

    ceil_bad = pd.Series(False, index=d.index)
    for c, cap in CAP.items():
        ceil_bad |= d[c].fillna(0) > cap * 1.15
    ceil_bad |= (d.generation_mw > SYS_CEIL) | (d.demand_mw > SYS_CEIL) | (d.load_shedding > SYS_CEIL)
    ssum = d[SRC].fillna(0).sum(axis=1)
    relres = (ssum - d.generation_mw).abs() / d.generation_mw
    bal_bad = (relres > 0.10) & (ssum > 0) & ~ceil_bad
    keep = ~(ceil_bad | bal_bad)

    h = d.loc[keep, ["generation_mw", "demand_mw", "load_shedding"]]
    h = h[h.index.minute == 0]
    grid = pd.date_range(h.index.min().floor("D"), h.index.max().floor("D") + pd.Timedelta(hours=23), freq="h")
    h0 = h.reindex(grid)
    h0.to_csv(os.path.join(RES, "hourly_clean.csv"), index_label="dt")
    h = h0.interpolate(method="time", limit=3, limit_area="inside")

    g = h.groupby(h.index.normalize())
    daily = pd.DataFrame({"G": g.generation_mw.sum(min_count=1), "Dpub": g.demand_mw.sum(min_count=1),
                          "S": g.load_shedding.sum(min_count=1), "slots": g.generation_mw.count()})
    daily = daily[daily.slots >= 20]
    daily["shed"] = daily.S > 0

    # weather, population-weighted
    nasa = json.load(open(os.path.join(DATA, "nasa_power_raw.json")))
    w = np.array([DIV[k] for k in DIV], float); w /= w.sum()
    wx = {}
    for v in ["T2M", "RH2M", "T2M_MAX"]:
        M = pd.DataFrame({k: pd.Series(nasa[k][v]) for k in DIV})
        M.index = pd.to_datetime(M.index, format="%Y%m%d")
        wx[v] = (M.replace(-999.0, np.nan) * w).sum(axis=1)
    wx = pd.DataFrame(wx)

    pan = daily.join(wx, how="inner")
    doy = pan.index.dayofyear.values
    pan["trend"] = (pan.index - pan.index.min()).days / 365.25
    pan["cdd"] = np.maximum(pan.T2M - 18.0, 0.0); pan["cdd2"] = pan.cdd ** 2
    pan["cdd_rh"] = pan.cdd * pan.RH2M / 100.0; pan["tmax"] = pan.T2M_MAX
    for k in (1, 2):
        pan[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25); pan[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    dow = pan.index.dayofweek.values
    pan["weekend"] = np.isin(dow, [4, 5]).astype(float)
    for k in range(1, 7):
        pan[f"dow{k}"] = (dow == k).astype(float)
    cov = np.zeros(len(pan))
    for a, b in COVID:
        cov[(pan.index >= a) & (pan.index <= b)] = 1.0
    pan["covid"] = cov

    wdi = json.load(open(os.path.join(DATA, "wdi_bgd.json")))
    yr = pd.Series(pan.index.year, index=pan.index)
    pan["gdp_growth"] = yr.map({int(k): v for k, v in wdi["NY.GDP.MKTP.KD.ZG"].items()}).values
    pan["log_gdp"] = np.log(yr.map({int(k): v for k, v in wdi["NY.GDP.MKTP.KD"].items()}).values)
    pan.to_csv(os.path.join(RES, "panel_daily.csv"), index_label="date")

    c = daily[(daily.index.year >= 2022) & (daily.index.year <= 2024)]
    depth = (c.S / c.Dpub)[c.shed].values
    np.save(os.path.join(RES, "depth_pool.npy"), depth)

    off = raw.dt.dt.minute != 0
    fid = pd.DataFrame([
        ("Raw observations", len(raw)), (":30-stamped rows", int(off.sum())),
        ("Hour-00 records 2016+", int(((raw.dt.dt.hour == 0) & (raw.dt.dt.year >= 2016)).sum())),
        ("Hours rejected: ceilings", int(ceil_bad.sum())), ("Hours rejected: energy balance", int(bal_bad.sum())),
        ("Retained days", len(daily)), ("Days with 24 slots", int((daily.slots == 24).sum())),
        ("Censored share %", round(100 * daily.shed.mean(), 2)), ("Censored days", int(daily.shed.sum())),
        ("Pr(shed) 2022-2024", round(c.shed.mean(), 4)),
        ("Depth mean (S/D | shed)", round(depth.mean(), 4)), ("Depth sd", round(depth.std(), 4)),
        ("Uncensored-era days 2016-2021", int(((daily.index.year >= 2016) & (daily.index.year <= 2021)).sum())),
    ], columns=["quantity", "value"])
    fid.to_csv(os.path.join(RES, "cleaning_fidelity.csv"), index=False)
    print(fid.to_string(index=False))


if __name__ == "__main__":
    main()
