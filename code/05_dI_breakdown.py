"""
MInA: decomposition of the information gain into microdomain contributions
==========================================================================
"""
#%%
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # Interactive session: the working directory must be code/.
    CODE_DIR = os.getcwd()

ROOT = os.path.dirname(CODE_DIR)
CSV_DIR = os.path.join(ROOT, "csv")
OUT_PREFIX = "MInA_dI_breakdown_"

# Tolerance on |sum_c dI_c - dI| used to check the decomposition.
ATOL = 1e-8


def contribution(w, H, H0):
    """dI_c = w_c (H0 - H_c); empty clusters (w_c = 0) contribute zero."""
    return np.where(w == 0, 0.0, w * (H0 - H))


def breakdown(df):
    n_max = max((int(m.group(1)) for c in df.columns
                 if (m := re.fullmatch(r"w_(\d+)", c))), default=0)
    H0 = df["H0"].to_numpy(float)

    out = pd.DataFrame(index=df.index)
    out["n_gaussians"] = df["n_gaussians"]
    out["H0"] = df["H0"]
    out["Hclust"] = df["Hclust"]
    out["dI"] = df["dI"]

    terms = []
    for i in range(1, n_max + 1):
        w = df[f"w_{i}"].fillna(0).to_numpy(float)
        dI_i = contribution(w, df[f"H_{i}"].to_numpy(float), H0)
        # Undefined for distributions with fewer than i fitted Gaussians.
        out[f"dI_{i}"] = np.where(df["n_gaussians"] >= i, dI_i, np.nan)
        terms.append(f"dI_{i}")
    w_ng = df["w_ng"].fillna(0).to_numpy(float)
    out["dI_ng"] = contribution(w_ng, df["H_ng"].to_numpy(float), H0)
    terms.append("dI_ng")

    out["dI_sum"] = out[terms].sum(axis=1, min_count=1)
    # Contributions as percentages of dI; undefined when dI = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in terms:
            out[f"{t}_pct"] = np.where(out["dI"] != 0, 100 * out[t] / out["dI"], np.nan)

    bad = (out["dI_sum"] - out["dI"]).abs() > ATOL
    if bad.any():
        warnings.warn(f"Decomposition does not match dI for: {list(out.index[bad])}")
    return out


def main():
    paths = sorted(p for p in glob.glob(os.path.join(CSV_DIR, "MInA_*.csv"))
                   if not os.path.basename(p).startswith(OUT_PREFIX))
    if not paths:
        raise FileNotFoundError(f"No MInA_*.csv files found in {CSV_DIR}")

    for path in paths:
        df = pd.read_csv(path, index_col=0)
        out = breakdown(df)
        name = os.path.basename(path)[len("MInA_"):]
        out.to_csv(os.path.join(CSV_DIR, OUT_PREFIX + name), index_label=df.index.name)


if __name__ == "__main__":
    main()
# %%
