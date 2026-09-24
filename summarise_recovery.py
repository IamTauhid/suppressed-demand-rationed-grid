"""
Stage 3b: summarise the recovery grid (results/recovery_raw.csv from run_recovery.py)
into the Table 7 body inputs and the standardisation diagnostics.

Outputs  results/table5_recovery.csv          mean correction over seeds, per holdout x admitted
         results/legacy_2020_failures.csv     failures of 9 draws per spec under sd+1e-8 scaling
         results/standardisation_diagnostic.csv  max |standardised test value| by cut and mode
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
FEATS = ["cdd", "cdd2", "cdd_rh", "tmax", "RH2M", "sin1", "cos1", "sin2", "cos2", "weekend",
         "dow1", "dow2", "dow3", "dow4", "dow5", "dow6", "covid"]


def main():
    rr = pd.read_csv(os.path.join(RES, "recovery_raw.csv"))
    w = rr.pivot_table(index=["mode", "holdout", "admitted", "seed", "n_train", "true_supp_pp"],
                       columns="spec", values=["bias_pp", "diverged"]).reset_index()
    w.columns = ["_".join([c for c in t if c]).strip("_") for t in w.columns]
    for s in ("tobit", "icg"):
        w[f"corr_{s}"] = w[f"bias_pp_{s}"] - w["bias_pp_blind"]
    fx = w[w["mode"] == "fixed"]
    tab = (fx.groupby(["holdout", "n_train", "admitted"])
             .agg(true_supp=("true_supp_pp", "mean"), tobit=("corr_tobit", "mean"), tobit_sd=("corr_tobit", "std"),
                  icg=("corr_icg", "mean"), icg_sd=("corr_icg", "std"), blind_bias=("bias_pp_blind", "mean"),
                  n_seeds=("seed", "size")).reset_index())
    tab["admitted"] = (100 * tab["admitted"]).round().astype(int)
    tab.to_csv(os.path.join(RES, "table5_recovery.csv"), index=False)
    lg = w[w["mode"] == "legacy"]
    lg[["diverged_blind", "diverged_tobit", "diverged_icg"]].sum().rename("n_failed_of_9").to_frame().T \
        .to_csv(os.path.join(RES, "legacy_2020_failures.csv"), index=False)

    P = pd.read_csv(os.path.join(RES, "panel_daily.csv"), index_col="date", parse_dates=True)
    era = P[(P.index.year >= 2016) & (P.index.year <= 2021)]
    rows = []
    for hold in (2019, 2020, 2021):
        tr, te = era[era.index.year < hold], era[era.index.year == hold]
        Xtr, Xte = tr[FEATS].values, te[FEATS].values
        m, sd = Xtr.mean(0), Xtr.std(0)
        for mode, sc in [("legacy", sd + 1e-8), ("fixed", np.where(sd < 1e-10, 1.0, sd))]:
            Bm = (Xte - m) / sc
            rows.append({"holdout": hold, "mode": mode, "n_train": len(tr), "n_test": len(te),
                         "const_train_cols": int((sd < 1e-10).sum()), "max_abs_std_test": float(np.abs(Bm).max()),
                         "worst_column": FEATS[int(np.abs(Bm).max(0).argmax())]})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "standardisation_diagnostic.csv"), index=False)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print("MAE tobit %.2f  icg %.2f" % ((tab.tobit - tab.true_supp).abs().mean(), (tab.icg - tab.true_supp).abs().mean()))
    print("fits:", len(rr), "| fixed diverged:", int(rr[rr["mode"] == "fixed"].diverged.sum()))


if __name__ == "__main__":
    main()
