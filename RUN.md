# Replication package: Suppressed Electricity Demand in a Rationed Grid

One command sequence regenerates every result-bearing table body and in-text figure in
`main.tex` from the raw inputs. Run from the package root.

## Inputs (data/)
| File | Source |
|---|---|
| `pgcb_V1_raw.csv` | PGCB hourly record, Mendeley Data doi:10.17632/vpk8spw2mm.1 (CC BY 4.0) |
| `nasa_power_raw.json` | NASA POWER daily API, parameters T2M, RH2M, T2M_MAX, T2M_MIN, eight divisional HQs, 2015-04-01..2026-03-08 |
| `wdi_bgd.json` | World Bank WDI `NY.GDP.MKTP.KD.ZG`, `NY.GDP.MKTP.KD`, `FP.CPI.TOTL.ZG` for BGD, retrieved 2026-09-22 |
| `table1_params.csv` | Per-unit costs, loss rates and other parameters exactly as printed in Table 1 (transcribed; sources named per row) |
| `tariff_series_TEMPLATE.csv` | Template for the annual nominal average retail tariff (BERC); **values not included**, see below |
| `wdi_conversion_series.json` | World Bank WDI `FP.CPI.TOTL` (BGD, USA), `NY.GDP.DEFL.ZS` (BGD), `PA.NUS.FCRF` (BGD), 1995-2025, retrieved 2026-09-23; used only by `voll_sources.py` |

`fetch_inputs.py` re-downloads the second and third (network required); the first is
included as supplied.

## Command sequence
```
python build_panel.py          # stage 1: clean, aggregate, design matrix        -> results/panel_daily.csv, depth_pool.npy, hourly_clean.csv
python timestamp_audit.py      # stage 2: Section 3.1 audit + midnight sensitivity -> results/timestamp_audit.csv, hour_label_convention_test.csv, midnight_treatment_sensitivity.csv
python run_recovery.py         # stage 3: 108 torch fits (~50 min on 4 CPU threads); latent series = G + S -> results/recovery_raw.csv
                               #   Table 7 = the 2026-09-23 run of this driver (torch 2.14.0+cpu). The earlier published-total run and a torch-version control are kept under results/ and logs/ with dated names.
python summarise_recovery.py   # stage 3b: Table 7 inputs + standardisation diagnostics
python tobit_penalty_sweep.py  # stage 3c: Tobit penalty sweep (Section 6.1) -> results/tobit_penalty_sweep.csv (~2 min)
python run_structural.py       # stage 4: OLS counterfactual, placebo, bootstrap, welfare, Table 3, alpha sweep
python make_tables.py          # stage 5 (run AFTER stage 9; reads results/cost_model.csv): tables/*.tex bodies + tables/numbers.tex macros
python make_figure1.py         # stage 6: figs/fig1_censoring.pdf
python identity_shares.py      # stage 7: Table 2 identity shares on the raw record -> results/identity_shares.csv
python voll_sources.py         # stage 8: Table 12 conversions of published outage-cost figures -> results/voll_source_conversions.csv, tables/tab_vollsources_body.tex
python cost_model.py           # stage 9: modelled cost-revenue gap (Section 7.1, Table 13) from Table 1 parameters, hourly source mix and BERC tariff series -> results/cost_model.csv
pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```
Logs of the run that produced the submitted numbers are in `logs/`.

## What `main.tex` reads from disk
* `tables/numbers.tex`, `\newcommand` macros for in-text figures (gaps, bounds, TWh, m$, RMSEs)
* `tables/tab_{recovery,ident,placebo,placebodetail,activity,welfare,excess,alphasens,depth,intensity,vollsources}_body.tex`, table bodies

Nothing numerical in those tables is typed by hand. In-text numbers outside the macro
set (e.g. the cleaning counts in Section 3.1, the 28/77/85% depth declines) are copied
from `results/*.csv` and listed with their source file in `CHANGELOG_revision.md`.

## Model specification (structural.py)
Outcome `log G` (daily served energy). Design: intercept, trend (years), `cdd`, `cdd^2`,
`cdd x RH`, two annual harmonic pairs, six day-of-week dummies, COVID-lockdown indicator
(2020-03-26..2020-05-30, 2021-04-05..2021-08-10), plus activity control
(`none` / `+growth` = annual real GDP growth / `+level` = log real GDP). Fit 2016-01-01..
2021-12-31 (n = 2,148). Gap = 100 (sum exp(pred) - sum G) / sum G per calendar year.
Rolling placebo: origins 2017..2020, horizons 1..3, targets with S/G < 0.03%.
Bootstrap: moving-block residual, 28-day blocks, 2000 replications, refit each.

## Recovery experiment (recovery.py / run_recovery.py)
Appendix A specification: shared trunk 96 -> 48, SiLU, dropout 0.10, separate mu and
log-sigma heads, explicit linear trend, AdamW lr 8e-3 wd 1e-4, cosine annealing, 1,500
full-batch steps, mu-head bias at the training mean. Tobit penalty lambda = 1 on censored
days. Censoring: Pr(shed) = 0.650, depth resampled from the empirical 2022-2024 S/D~ pool (depth is defined on the published total; see Section 6.1). Latent series over 2016-2021 is G + S, the admitted total, not the published column.
`mode="legacy"` reproduces the sd + 1e-8 standardisation that broke the 2020 holdout. Executed 2026-09-23 with torch 2.14.0+cpu; a torch-version control against the 2026-09-22 run (torch 2.13) showed per-fit differences <= 0.011 pp.

## Review build vs submission build
`build/` contains compatibility shims (`elsarticle.cls`, `siunitx.sty`, `lineno.sty`)
used only because the compile host had no CTAN access. They reproduce the constructs
`main.tex` uses; the Elsevier layout is obtained by compiling the same `main.tex` with
the real `elsarticle` class. `main.tex` itself is unchanged between the two builds.

## Not part of the replication (working-archive only)
`revise_manuscript.py`, `revise_round4.py` ... `revise_round7.py` are the exact-replacement scripts that
transformed each pre-submission draft of `main.tex` into the next; `revision_memo_round{4,5,6}.md`,
`CHANGELOG_revision.md`, `SUBMISSION_ASSESSMENT.md` and `claim_evidence_inventory.md` document the
pre-submission revision history. None of these is needed to regenerate any result.

## Inputs transcribed from external documents (and how to verify them)

| Input file | Content | Source to verify against |
|---|---|---|
| `data/table1_params.csv` | Per-unit costs by source (BDT/kWh, FY2025), loss-rate ranges, FX, scenario values, exactly as in Table 1 | BPDB Annual Report 2024-25 (https://bpdb.gov.bd, Monthly/Annual Report page, listed 09 Oct 2025); PGCB Annual Report 2023-24 (https://pgcb.gov.bd). Page references still to be added by the authors. |
| `data/berc_tariff_orders.csv` | Every change in the BERC weighted-average retail tariff, Sep 2015 - Feb 2024, with effective month, size of change and the press/BPDB source for each | BERC tariff orders (https://berc.org.bd, Tariff Orders); BPDB tariff-rate page (https://bpdb.gov.bd, "Tariff Rate: Effect from ..."); sources quoted row by row in the file |
| `data/tariff_series.csv` | Calendar-year averages of that schedule (2026 = Jan-Mar, the sample end) | Derived from the previous file; arithmetic shown in the `basis` column |
| `data/wdi_conversion_series.json`, `data/wdi_bgd.json` | World Bank WDI series | `fetch_inputs.py` re-downloads them |

`cost_model.py` reads the first three and writes the modelled cost-revenue gap (Table 13 last column), the break-even values and the real-tariff change used in the text (via `make_tables.py` macros).

## Quantity in the manuscript that this package does NOT regenerate

| Item | Missing input | Exact retrieval | What to run once obtained |
|---|---|---|---|
| Table `tab:second` (identity in the BPDB zone-day series; 16,650 zone-days, 99.6%) | Hossain, Nafs, Yeaser & Newaz (2025) dataset | Mendeley Data doi:10.17632/x7r7wdb39k.1 (V1); described in *Data in Brief* 62:112014, doi:10.1016/j.dib.2025.112014. Not reachable from the build host (data.mendeley.com resolves to a blocked address range), so it could not be bundled. | For each zone-day compute reported demand minus (reported supply + reported shedding); Table `tab:second` reports the share of days with zero residual and with residual within 0.2% and 0.5% of the zone's own peak. The authors' original script for this table is not in the package; the definition above is sufficient to recompute it. |

See `claim_evidence_inventory.md` for a claim-by-claim list.

## Not part of the replication (working-archive only)
`revise_manuscript.py`, `revise_round4.py` ... `revise_round7.py` are the exact-replacement scripts that
transformed each pre-submission draft of `main.tex` into the next; `revision_memo_round{4,5,6}.md`,
`CHANGELOG_revision.md`, `SUBMISSION_ASSESSMENT.md` and `claim_evidence_inventory.md` document the
pre-submission revision history. None of these is needed to regenerate any result.

## Quantities in the manuscript that this package does NOT regenerate, and how to obtain the inputs

| Item | Missing input | Exact retrieval | What to run once obtained |
|---|---|---|---|
| Table `tab:second` (identity in the BPDB zone-day series; 16,650 zone-days, 99.6%) | Hossain, Nafs, Yeaser & Newaz (2025) dataset | Mendeley Data doi:10.17632/x7r7wdb39k.1 (V1); described in *Data in Brief* 62:112014, doi:10.1016/j.dib.2025.112014. Not reachable from the build host (data.mendeley.com resolves to a blocked range), so it could not be bundled. | For each zone-day compute reported demand minus (reported supply + reported shedding); Table `tab:second` reports the share of days with zero residual and with residual within 0.2% and 0.5% of the zone's own peak. The authors' original script for this table is not in the package; the definition above is sufficient to recompute it. |
| Table 1 per-unit generation costs and loss rates | Source documents behind the transcribed values | BPDB Annual Report 2024-25 (https://bpdb.gov.bd, Monthly/Annual Report page, listed 09 Oct 2025); PGCB Annual Report 2023-24 (https://pgcb.gov.bd). Page references are still to be added by the authors. | Values are in `data/table1_params.csv`; `cost_model.py` consumes them. |
| Revenue side of the modelled cost-revenue gap (Table `tab:welfare`, last column) and the break-even values | Annual nominal average retail tariff, 2015-2026 | Bangladesh Energy Regulatory Commission tariff orders (https://berc.org.bd, Tariff Orders); the manuscript's Table 1 gives the range 6.10-8.95 BDT/kWh. Fill `data/tariff_series_TEMPLATE.csv`, save as `data/tariff_series.csv`. | `python cost_model.py` then also writes revenue, gap (bn BDT) and gap (m USD at 122). Consistency check: the carried gap column implies tariffs of about 6.7 (2022), 7.5 (2023), 8.1 (2024) and 8.5 (2025) BDT/kWh when combined with the regenerated cost side, all inside the Table 1 range. |
| Real tariff decline of 13.7% (Section 8.3) | Same tariff series plus BBS CPI | As above; CPI from `data/wdi_bgd.json` (`FP.CPI.TOTL.ZG`). | Arithmetic on the series. |

See `claim_evidence_inventory.md` for a claim-by-claim list.

## Independent re-execution (2026-09-24)

The package was re-run from the raw inputs on a different machine, into an empty
`results/` directory, and the output compared file by file with the committed set.
Environment differences: numpy 2.5.2 (pinned 2.4.6), pandas 3.0.5 (pinned 2.3.3),
torch 2.13.0+cpu (the committed recovery grid used 2.14.0+cpu), 4 threads rather
than 8.

| Stage | Outcome |
|---|---|
| `build_panel.py` | identical: 3,928 retained days, 1,076 censored, 4,439 :30-stamped rows, zero hour-00 records from 2016 |
| `identity_shares.py` | `identity_shares.csv` byte-identical |
| `timestamp_audit.py` | `timestamp_audit.csv`, `hour_label_convention_test.csv`, `midnight_treatment_sensitivity.csv` byte-identical |
| `cost_model.py` | `cost_model.csv` byte-identical |
| `voll_sources.py` | conversions reproduce |
| `run_structural.py` | every printed gap, bound and bootstrap limit reproduces; `structural.csv`, `placebo_detail.csv`, `spec_comparison.csv`, `fitted_year_gaps.csv` agree to a maximum relative difference of **4.0e-10** (last-bit float, attributable to the numpy/pandas version change) |
| `run_recovery.py` | 62 of 108 fits completed at time of writing. `blind` 21/21 and `tobit` 21/21 bit-identical; `icg` 15/20 bit-identical, remaining five deviating by at most 0.0087 pp in `bias_pp` (mean absolute deviation 0.0011 pp). This is the torch 2.13 vs 2.14 difference, two orders of magnitude below the ICG mean absolute calibration error of 0.84 pp. The interval arm is not bit-reproducible across torch minor versions and we do not claim that it is; the likely cause is reduction order in the multi-threaded tail evaluations it uses and the other two arms do not. |

Independent verification of the manuscript against these files is in
`audit_numbers.py`: 56 quantities recomputed from `results/*.csv` and compared with
the printed values and with `tables/numbers.tex`, all matching.
