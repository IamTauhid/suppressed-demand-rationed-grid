"""Cost side of the modelled cost-revenue gap (Section 7.1), rebuilt from the hourly source mix.

Inputs : data/pgcb_V1_raw.csv            hourly generation by source (MW, spot readings -> MWh per hour)
         data/table1_params.csv          per-unit costs by source (BDT/kWh, FY2025) and loss rates, as in Table 1
         data/tariff_series.csv          OPTIONAL: year, tariff_bdt_kwh (nominal average retail tariff, BERC orders).
                                         Not included in the package (see RUN.md); a template with the Table 1
                                         range and no yearly values is provided as data/tariff_series_TEMPLATE.csv.
Outputs: results/cost_model.csv          per year: energy by source, cost by source, weighted average cost,
                                         delivered energy, and (if a tariff series is supplied) revenue and gap
Log    : logs/cost_model.log

Checks the manuscript's calibration statement: weighted average generation cost built up from Table 1 for 2025
against the published FY2025 average purchase cost of 12.10 BDT/kWh (deviation reported).
"""
import pandas as pd, numpy as np, os

raw = pd.read_csv("data/pgcb_V1_raw.csv", parse_dates=["datetime"])
prm = pd.read_csv("data/table1_params.csv")
uc = prm[prm.kind == "unit_cost"]
cost = dict(zip(uc.parameter, uc.value.astype(float)))
loss_tx = float(prm[prm.parameter == "transmission_loss_mid"].value.iloc[0])
loss_dx = float(prm[prm.parameter == "distribution_loss_mid"].value.iloc[0])
SRC = ["gas", "liquid_fuel", "coal", "hydro", "solar", "wind", "india_bheramara_hvdc", "india_tripura", "india_adani", "nepal"]
missing = [c for c in SRC if c not in raw.columns]
assert not missing, missing

# Spot MW readings at hourly stamps -> MWh per hour. Rows with :30 stamps are excluded (Section 3.1 cleaning)
# and rows failing the system ceiling are excluded, consistent with build_panel.py.
h = raw[raw.datetime.dt.minute == 0].copy()
h = h[(h.generation_mw <= 20000) & (h.demand_mw <= 20000) & (h.load_shedding <= 20000)]
h["yr"] = h.datetime.dt.year
# Source-level energy balance: the source columns contain data-entry errors (e.g. liquid_fuel =
# 29,222,897 MW on 2022-03-05 09:00). An hour is "balanced" if the source sum is within 5% of the
# generation column; the fuel mix and the weighted average cost are computed on balanced hours and
# the annual cost is that average applied to the year's total served energy (generation column).
h["src_sum"] = h[SRC].fillna(0).sum(axis=1)
h["balanced"] = (h.src_sum - h.generation_mw).abs() <= 0.05 * h.generation_mw
rows = []
for yr, d in h.groupby("yr"):
    b = d[d.balanced]
    e = {c: b[c].fillna(0).sum() / 1e3 for c in SRC}            # GWh by source, balanced hours
    Eb = sum(e.values()); Cb = sum(e[c] * 1e6 * cost[c] for c in SRC)
    avg = Cb / (Eb * 1e6) if Eb else np.nan                       # BDT/kWh
    G_rec = d.generation_mw.fillna(0).sum() / 1e3                 # GWh, all valid hours
    r = dict(year=yr, hours=len(d), balanced_hours=len(b), unbalanced_pct=round(100 * (1 - len(b) / len(d)), 2),
             G_GWh=round(G_rec, 1), avg_cost_BDT_kWh=round(avg, 3), cost_bn_BDT=round(avg * G_rec * 1e6 / 1e9, 2),
             delivered_GWh=round(G_rec * (1 - loss_tx) * (1 - loss_dx), 1))
    r.update({f"share_{c}_pct": round(100 * e[c] / Eb, 2) for c in SRC})
    rows.append(r)
out = pd.DataFrame(rows)

tariff_path = "data/tariff_series.csv"
if os.path.exists(tariff_path):
    tr = pd.read_csv(tariff_path)
    out = out.merge(tr[["year", "tariff_bdt_kwh"]], on="year", how="left")
    out["revenue_bn_BDT"] = out.delivered_GWh * 1e6 * out.tariff_bdt_kwh / 1e9
    out["gap_bn_BDT"] = out.cost_bn_BDT - out.revenue_bn_BDT
    out["gap_mUSD_at_122"] = out.gap_bn_BDT * 1e9 / 122 / 1e6
    print("tariff series supplied: gap computed")
else:
    print("no data/tariff_series.csv supplied: cost side only (revenue and gap not computed)")
out.to_csv("results/cost_model.csv", index=False)
print(out[["year", "hours", "balanced_hours", "unbalanced_pct", "G_GWh", "avg_cost_BDT_kWh", "cost_bn_BDT", "delivered_GWh"]].to_string(index=False))
c25 = float(out.loc[out.year == 2025, "avg_cost_BDT_kWh"].iloc[0])
print(f"calibration: 2025 weighted average cost {c25:.2f} BDT/kWh vs published FY2025 purchase cost 12.10 -> deviation {100*(c25/12.10-1):+.1f}% (manuscript states 11.54, -4.6%)")
