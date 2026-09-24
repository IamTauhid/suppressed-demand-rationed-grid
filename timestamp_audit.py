"""
Stage 2: timestamp audit (Section 3.1) and midnight-treatment sensitivity.

Inputs   data/pgcb_V1_raw.csv, results/hourly_clean.csv
Outputs  results/timestamp_audit.csv              parse / duplicate / 24:00 / :30 / gap counts
         results/hour_label_convention_test.csv   per-day solar centroid vs solar noon
         results/midnight_treatment_sensitivity.csv  annual S/G bounds under four treatments
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); RES = os.path.join(HERE, "results")


def solar_noon_local(dates, lon=90.4125, tz=6.0):
    n = dates.dayofyear.values
    B = 2 * np.pi * (n - 81) / 364.0
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    return 12.0 + (tz * 15.0 - lon) * 4.0 / 60.0 - eot / 60.0


def build_daily(hh, treatment):
    x = hh.interpolate(method="time", limit=3, limit_area="inside")
    if treatment == "equal_weight":
        x.loc[x.index.hour == 0] = np.nan
        g = x.groupby(x.index.normalize()); out = g.mean() * 24.0; n = g.generation_mw.count(); minslots = 19
    else:
        if treatment in ("ffill_23", "bfill_01"):
            mid = x.index.hour == 0
            src = x.shift(1) if treatment == "ffill_23" else x.shift(-1)
            x.loc[mid] = src.loc[mid].values
        g = x.groupby(x.index.normalize()); out = g.sum(min_count=1); n = g.generation_mw.count(); minslots = 20
    out["slots"] = n
    out = out[out.slots >= minslots]
    out.columns = ["G", "Dpub", "S", "slots"]
    return out


def main():
    raw = pd.read_csv(os.path.join(DATA, "pgcb_V1_raw.csv"))
    s = raw["datetime"].astype(str)
    parsed = pd.to_datetime(s, errors="coerce")
    full = pd.date_range(parsed.min().floor("D"), parsed.max().floor("D") + pd.Timedelta(hours=23), freq="h")
    on_hour = parsed[parsed.dt.minute == 0]
    missing = full.difference(pd.DatetimeIndex(on_hour))
    m2 = missing[missing.hour != 0]
    grp = (pd.Series(m2).diff() != pd.Timedelta(hours=1)).cumsum()
    runs = pd.Series(1, index=m2).groupby(grp.values).size()
    audit = {
        "raw_rows": len(raw), "parse_failures": int(parsed.isna().sum()),
        "rows_containing_24:00": int(s.str.contains(r"\b24:00").sum()),
        "duplicate_timestamps": int(parsed.duplicated().sum()),
        "half_hour_rows": int((parsed.dt.minute == 30).sum()),
        "other_nonzero_minute_rows": int(((parsed.dt.minute != 0) & (parsed.dt.minute != 30)).sum()),
        "out_of_order_rows": int((parsed.diff().dt.total_seconds() < 0).sum()),
        "grid_hours": len(full), "missing_grid_hours": len(missing),
        "missing_grid_hours_excl_hour00": len(m2),
        "missing_hour00_2016plus": int(((missing.hour == 0) & (missing.year >= 2016)).sum()),
        "gap_runs_excl_hour00": int(len(runs)), "gap_runs_len_le3": int((runs <= 3).sum()),
        "longest_gap_hours": int(runs.max()),
    }
    pd.Series(audit).to_frame("value").to_csv(os.path.join(RES, "timestamp_audit.csv"))

    # hour-label convention: solar-weighted centroid of labels vs solar noon
    df = raw.assign(dt=parsed)
    sol = df[(df.solar.fillna(0) > 0) & (df.dt.dt.minute == 0)].copy()
    sol["hour"] = sol.dt.dt.hour; sol["date"] = sol.dt.dt.normalize()
    cnt = sol.groupby("date").hour.count(); sol = sol[sol.date.isin(cnt[cnt >= 8].index)]
    cen = sol.groupby("date").apply(lambda g: np.average(g.hour, weights=g.solar), include_groups=False).rename("label_centroid").to_frame()
    cen["solar_noon"] = solar_noon_local(cen.index)
    cen["err_begin"] = cen.label_centroid + 0.5 - cen.solar_noon
    cen["err_end"] = cen.label_centroid - 0.5 - cen.solar_noon
    cen["err_spot"] = cen.label_centroid - cen.solar_noon
    cen.to_csv(os.path.join(RES, "hour_label_convention_test.csv"))
    print("days %d | mean err: begin %+.3f  end %+.3f  spot %+.3f" % (len(cen), cen.err_begin.mean(), cen.err_end.mean(), cen.err_spot.mean()))

    h0 = pd.read_csv(os.path.join(RES, "hourly_clean.csv"), index_col="dt", parse_dates=True)
    rows = []
    for tr in ["interp", "equal_weight", "ffill_23", "bfill_01"]:
        dd = build_daily(h0, tr)
        yy = dd.groupby(dd.index.year).agg(G=("G", "sum"), S=("S", "sum"), n=("G", "size")).loc[2022:2025]
        for y, r in yy.iterrows():
            rows.append({"treatment": tr, "year": y, "days": int(r.n), "lower_SG_pp": 100 * r.S / r.G,
                         "upper_pp": 100 * r.S / (0.30 * r.G)})
    R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "midnight_treatment_sensitivity.csv"), index=False)
    print(R.pivot(index="year", columns="treatment", values="lower_SG_pp").round(3).to_string())


if __name__ == "__main__":
    main()
