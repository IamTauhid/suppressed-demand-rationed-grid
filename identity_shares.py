"""Recompute Table 2 (tab:identity): share of raw hourly rows on which published demand equals
generation plus admitted shedding, by era and tolerance. Raw record, no cleaning.

Input : data/pgcb_V1_raw.csv     Output: results/identity_shares.csv     Log: logs/identity_shares.log
"""
import pandas as pd, numpy as np

raw = pd.read_csv("data/pgcb_V1_raw.csv")
raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
ok = raw.dropna(subset=["datetime", "generation_mw", "demand_mw", "load_shedding"]).copy()
ok["res"] = ok["demand_mw"] - (ok["generation_mw"] + ok["load_shedding"])
ok["yr"] = ok["datetime"].dt.year
print("raw rows", len(raw), "| rows with D, G, S all present", len(ok))

eras = {"pre2022": ok[ok.yr < 2022], "2022_2024": ok[(ok.yr >= 2022) & (ok.yr <= 2024)]}
rows = []
for era, d in eras.items():
    r = dict(era=era, hours=len(d),
             exact_pct=100 * (d.res == 0).mean(),
             within1MW_pct=100 * (d.res.abs() <= 1).mean(),
             within50MW_pct=100 * (d.res.abs() <= 50).mean(),
             within100MW_pct=100 * (d.res.abs() <= 100).mean(),
             excess_pct=100 * (d.res > 0).mean())
    rows.append(r)
    print(f"{era:10s} n={r['hours']:6d} exact={r['exact_pct']:.1f} ±1MW={r['within1MW_pct']:.1f} ±50MW={r['within50MW_pct']:.1f} ±100MW={r['within100MW_pct']:.1f} D>G+S={r['excess_pct']:.1f}")
pd.DataFrame(rows).to_csv("results/identity_shares.csv", index=False)
