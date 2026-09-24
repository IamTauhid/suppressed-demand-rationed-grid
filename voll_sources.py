"""Re-express published Bangladesh outage-cost / lost-output figures in 2024 BDT per kWh.

Inputs  : data/wdi_conversion_series.json  (World Bank WDI: FP.CPI.TOTL for BGD and USA,
          PA.NUS.FCRF for BGD, NY.GDP.DEFL.ZS for BGD; fetched 2026-09-23 via api.worldbank.org)
Outputs : results/voll_source_conversions.csv
          tables/tab_vollsources_body.tex
Log     : logs/voll_sources.log (stdout)

Source figures (quantity, unit, price year, coverage) are entered verbatim from the documents
named in the manuscript bibliography; nothing here is estimated. Conversion routes:
  A  deflate in USD with US CPI to 2024, then convert at the 2024 average BDT/USD rate
  B  convert at the price-year BDT/USD rate, then inflate with Bangladesh CPI to 2024
  C  as B but with the Bangladesh GDP deflator
Route B/C is the natural one for a Bangladesh-domestic cost; route A is reported to show
how much the answer depends on the route. The manuscript quotes the B-route value and the
A-to-C range.
"""
import json, csv, os

os.makedirs("results", exist_ok=True); os.makedirs("tables", exist_ok=True)
S = json.load(open("data/wdi_conversion_series.json"))
cpi_b = {int(k): v for k, v in S["cpi_bgd"].items()}
cpi_u = {int(k): v for k, v in S["cpi_usa"].items()}
fx    = {int(k): v for k, v in S["fx_bgd"].items()}
defl  = {int(k): v for k, v in S["defl_bgd"].items()}
T = 2024  # target price year (last year with all four series)

def route_A_usd(v, py): return v * cpi_u[T] / cpi_u[py] * fx[T]
def route_B_usd(v, py): return v * fx[py] * cpi_b[T] / cpi_b[py]
def route_C_usd(v, py): return v * fx[py] * defl[T] / defl[py]
def route_B_bdt(v, py): return v * cpi_b[T] / cpi_b[py]
def route_C_bdt(v, py): return v * defl[T] / defl[py]

# (key, source label, quantity measured, coverage, value, unit, price year, currency)
SRC = [
    ("wj_planned",   "Wijayatunga and Jayalath (2008)", "Industrial outage cost, planned interruptions",
     "Industrial consumers, survey 2001--2003", 0.34, "US\\$/kWh", 2001, "USD"),
    ("wj_unplanned", "Wijayatunga and Jayalath (2008)", "Industrial outage cost, unplanned interruptions",
     "Industrial consumers, survey 2001--2003", 0.83, "US\\$/kWh", 2001, "USD"),
    ("mc_low",  "Mujeri and Chowdhury (2013), as cited by ADB (2015)", "Economic output per unit of additional supply, low",
     "Economy-wide", 46.0, "Tk/kWh", 1996, "BDT"),
    ("mc_high", "Mujeri and Chowdhury (2013), as cited by ADB (2015)", "Economic output per unit of additional supply, high",
     "Economy-wide", 107.0, "Tk/kWh", 1996, "BDT"),
    ("adb_wtp", "ADB (2013)", "Willingness to pay, lower bound set at the tariff",
     "Household, agriculture and commercial consumers, western Bangladesh", 5.78, "Tk/kWh", 2013, "BDT"),
]
rows = []
for key, src, qty, cov, v, unit, py, cur in SRC:
    if cur == "USD":
        a, b, c = route_A_usd(v, py), route_B_usd(v, py), route_C_usd(v, py)
    else:
        a, b, c = float("nan"), route_B_bdt(v, py), route_C_bdt(v, py)
    rows.append(dict(key=key, source=src, quantity=qty, coverage=cov, value=v, unit=unit, price_year=py,
                     bdt2024_routeA=round(a, 1) if a == a else "", bdt2024_routeB=round(b, 1), bdt2024_routeC=round(c, 1)))

with open("results/voll_source_conversions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def rng(r):
    vals = [x for x in (r["bdt2024_routeA"], r["bdt2024_routeB"], r["bdt2024_routeC"]) if x != ""]
    lo, hi = min(vals), max(vals)
    return f"{lo:.0f}" if round(lo) == round(hi) else f"{lo:.0f}--{hi:.0f}"

SHORT = {  # compact labels for the typeset table; full text is in the CSV
    "wj_planned":   ("\\cite{wijayatunga2008}", "Outage cost, planned interruptions", "Industry (survey 2001--03)"),
    "wj_unplanned": ("\\cite{wijayatunga2008}", "Outage cost, unplanned interruptions", "Industry (survey 2001--03)"),
    "mc_low":       ("\\cite{mujeri2013} via \\cite{adb2015ea}", "Output per kWh of added supply, low", "Economy-wide"),
    "mc_high":      ("\\cite{mujeri2013} via \\cite{adb2015ea}", "Output per kWh of added supply, high", "Economy-wide"),
    "adb_wtp":      ("\\cite{adb2013efa}", "WTP lower bound (= tariff)", "Non-industrial, western BD"),
}
lines = []
for r in rows:
    src, qty, cov = SHORT[r["key"]]
    lines.append(f"{src} & {qty} & {cov} & {r['value']:g} {r['unit']} ({r['price_year']}) & {r['bdt2024_routeB']:.0f} & {rng(r)} \\\\")
open("tables/tab_vollsources_body.tex", "w").write("\n".join(lines) + "\n")

print("target price year", T)
print("factors: US CPI 2024/2001 = %.3f ; BGD CPI 2024/2001 = %.3f ; BGD CPI 2024/1996 = %.3f ; BGD deflator 2024/1996 = %.3f ; BGD CPI 2024/2013 = %.3f"
      % (cpi_u[T]/cpi_u[2001], cpi_b[T]/cpi_b[2001], cpi_b[T]/cpi_b[1996], defl[T]/defl[1996], cpi_b[T]/cpi_b[2013]))
print("FX BDT/USD 2001 = %.2f ; 2024 = %.2f" % (fx[2001], fx[T]))
for r in rows:
    print(f"{r['key']:12s} {r['value']:>7g} {r['unit']:8s} {r['price_year']}  A={r['bdt2024_routeA']} B={r['bdt2024_routeB']} C={r['bdt2024_routeC']}")
