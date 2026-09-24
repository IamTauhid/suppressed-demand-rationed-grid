# Suppressed Electricity Demand in a Rationed Grid

Replication package for *Suppressed Electricity Demand in a Rationed Grid: A Reporting
Audit and Conditional Bounds. Evidence from Bangladesh* — the Bangladesh national grid,
2015–2026. Manuscript under review at *Utilities Policy*.

Every table body and in-text figure in the paper is regenerated from the raw inputs by
the command sequence in [`RUN.md`](RUN.md). The manuscript source is not included here;
it is supplied to the journal.

## What the paper claims

**The published demand series is, before the crisis, a restatement of supply.** It
equals served generation plus the admitted shortfall in **99.8%** of pre-2022 hours to
within 1 MW, and in **90.7%** exactly. The same identity holds on **99.6%** of 16,650
zone-days in an independently compiled series from a different reporting authority.
Under rationing it degrades but does not break: it holds exactly in **53.0%** of
2022–2024 hours, and in **46.3%** of hours published demand exceeds the identity.

**The identity is a statement about how the number is produced, not about intent.**
Whether the operator's shedding estimate carries independent information about unmet
demand depends on how it is constructed, which the published record does not disclose.
The paper does not claim deliberate concealment.

**Suppressed demand is bounded, not measured, and the bounds are conditional.** For
2023 the range runs from **2.97 pp** of served energy (the reported shortfall) to
**9.91 pp** under an assumed admission floor of 0.30 — conditional on the operator
neither overstating the shortfall nor omitting it entirely. The upper limit is the
lower limit divided by that floor and carries no information the reported shortfall
does not.

**The independent counterfactual is reported as a conditional scenario, not a damage
estimate.** It gives **7.9 pp** for 2023, inside the bound, but exceeds the upper limit
in 2024 and 2025, and rolling-origin testing shows spurious gaps of 6 to 11 pp across a
structural break — comparable to the quantity being estimated. Monetised figures
(2.0 to 6.7 bn USD for 2023) are stated against a **modelled cost–revenue gap**, not an
audited fiscal subsidy.

**Curtailment was close to uniform across the day and shallower at peak.** Peak hours
were shed marginally more often (occurrence ratio **1.075**) but materially less deeply
(intensity ratio **0.868**, 229 MW at peak against 264 MW off peak). Both are
consistent with an energy rather than a peak-capacity constraint, which is a supported
hypothesis rather than an established fact: the record does not report unit
availability.

## Layout

```
*.py            pipeline, run in the order given in RUN.md
data/           raw and transcribed inputs (PGCB hourly record, NASA POWER, WDI,
                BERC tariff orders, Table 1 parameters)
results/        every derived series and result file behind the tables
tables/         LaTeX table bodies and the \newcommand macros for in-text numbers
figs/           Figure 1
logs/           logs of the runs that produced the submitted numbers
citation_audit  every citation checked against its source
```

## Reproducing

See [`RUN.md`](RUN.md) for the full sequence, inputs and per-stage outputs. In short:

```bash
python build_panel.py        # clean, aggregate, design matrix
python timestamp_audit.py    # Section 3.1 aggregation audit
python run_recovery.py       # 108 torch fits, ~50 min on 4 CPU threads
python summarise_recovery.py
python tobit_penalty_sweep.py
python run_structural.py     # OLS counterfactual, placebo, bootstrap, welfare
python make_tables.py        # run after cost_model.py
python make_figure1.py
python identity_shares.py
python voll_sources.py
python cost_model.py
```

`environment.txt` pins the interpreter and package versions. `fetch_inputs.py`
re-downloads the NASA POWER and World Bank series; the PGCB record is included as
supplied.

## Known limitations, stated in the paper

- **Provenance of the reported shortfall is not established.** Whether shedding is
  metered, inferred from feeder loads, forecast, or computed as a residual is not
  disclosed in the published record. This is the paper's principal limitation, and the
  lower bound is a bound only if the operator does not overstate.
- **Midnight is never observed.** Hour 00 carries no records from 2016 onward; the
  value in every daily total for that slot is interpolated. 4.68% of raw rows sit at
  :30, all at 18:30 or 19:30, and are dropped by the hourly reindex rather than
  double-counted. The differential effect on shed versus non-shed days is −0.025 pp of
  daily energy, against an S/G of 2.1–3.0 pp.
- **No audited subsidy figures.** Every fiscal comparison is against a modelled
  cost–revenue gap and is labelled as such.
- **The operational record contains no monetary variables.** Every cost, tariff and
  welfare figure rests on exogenous published parameters, listed with sources in
  `data/table1_params.csv` and swept in sensitivity analysis.

## Data

- **Primary**: Shekh & Rafi, *Hourly Electricity Generation, Demand, Load Shedding, and
  Fuel Mix Dataset for Bangladesh (2015–2026)*, Mendeley Data V1,
  [doi:10.17632/vpk8spw2mm.1](https://doi.org/10.17632/vpk8spw2mm.1), CC BY 4.0.
  Included here as supplied, under that licence.
- **Weather**: NASA POWER daily API, eight divisional headquarters, population-weighted
  by the 2022 census.
- **Macroeconomic**: World Bank WDI.
- **Tariffs**: BERC tariff orders, transcribed with effective months and sources.

## Licence

Code and derived results: MIT (see [`LICENSE`](LICENSE)). The PGCB hourly record in
`data/pgcb_V1_raw.csv` is redistributed under CC BY 4.0 per its original licence.

## Citing

See [`CITATION.cff`](CITATION.cff). Please cite both the paper and the underlying
dataset.
