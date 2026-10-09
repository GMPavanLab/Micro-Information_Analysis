"""
MInA: Micro-Information Analysis of synthetic distributions
===========================================================

"""
import glob
import importlib
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: figures are only saved to disk.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # Interactive session: the working directory must be code/.
    CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)

mina = importlib.import_module("00_MInA_functions")

ROOT = os.path.dirname(CODE_DIR)
NPY_DIR = os.path.join(ROOT, "array", "synthetic")
OUT_CSV = os.path.join(ROOT, "csv", "MInA_synthetic.csv")
FIG_DIR = os.path.join(ROOT, "figures", "synthetic")
BAR_DIR = os.path.join(FIG_DIR, "00_structure_diagram")
SCAT_DIR = os.path.join(FIG_DIR, "01_information_structure_plot")

# ==============================================================================
# ANALYSIS PARAMETERS
# ==============================================================================

FIT_PARAMS = dict(
    bw_adjust=0.3,
    prominence=0.003,
    peak_height_threshold=0.05,
    use_pad=False,
)
POP_PARAMS = dict(
    resid_thr=0.5,
    entropy_method="histo",
    n_bins=50,
)

AREA_KEYS = ["npeaks", "Ag", "Ang", "h_ng", "h_g", "Eguf", "Egef"]

# ==============================================================================
# PLOT SETTINGS
# ==============================================================================

LABELS = {
    "dI": r"$\Delta I$",
    "Eguf": r"$\sum guf$",
    "Egef": r"$\sum gef$",
    "Ew": r"$\sum gef$",
    "Ewng": r"$\sum guf$",
}

# (y, x) pairs, each drawn with both colour schemes.
PLOTS_BOTH = [("dI", "Egef"), ("dI", "Eguf"), ("dI", "Ewng"), ("dI", "Ew")]
# (y, x) pairs, drawn with one colour per array only.
PLOTS_ARRAY = [("Eguf", "Egef"), ("Ewng", "Ew")]

# Colour pairs (dark, light): gauss_n takes the dark colour, nongauss_n the light one.
PALETTE = [
    "#233d4d", "#457ea1", "#fe7f2d", "#dfb182", "#1A635B", "#39A295",
    "#58c29f", "#bad4aa", "#6d4c2e", "#c08552", "#432371", "#A670F3",
]
# Offsets (in points) of the letter labels, tuned on dI vs Eguf; arrays not
# listed use the default offset of scatter_plot.
LETTER_OFFSETS = {
    "gauss_1": (6, 11), "nongauss_1": (-3, -16), "gauss_2": (7, 8),
    "nongauss_2": (-8, 12), "gauss_3": (13, -6), "nongauss_3": (-12, 17),
    "gauss_4": (15, -13), "nongauss_4": (14, -10), "gauss_5": (-3, 15),
    "nongauss_5": (11, -9), "gauss_6": (-13, 2), "nongauss_6": (-11, 0),
}
NPEAKS_COLORS = {1: "#457592", 2: "#a7cc20", 3: "#2a9d8f",
                 4: "#A268E4", 5: "#7e3274", 6: "#9b2226", 7: "#9b2226"}


# ==============================================================================
# ANALYSIS
# ==============================================================================

def parse_name(name):
    """Return (kind, n_true) from a file stem of the form <kind>_<n_true>."""
    kind, _, n = name.rpartition("_")
    return (kind, int(n)) if n.isdigit() else (name, np.nan)


def run_analysis():
    npy_files = sorted(glob.glob(os.path.join(NPY_DIR, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {NPY_DIR}")

    rows, fractions = [], {}
    for path in npy_files:
        name = os.path.splitext(os.path.basename(path))[0]
        kind, n_true = parse_name(name)
        data = np.asarray(np.load(path), dtype=np.float64)

        pop, W = mina.mina_analysis(data, method="population", return_fractions=True,
                                    **FIT_PARAMS, **POP_PARAMS)
        *area, explained = mina.mina_analysis(data, method="area", return_fractions=True,
                                              **FIT_PARAMS)
        fractions[name] = {"area": explained, "population": W}

        rows.append({
            "array": name,
            "kind": kind,
            "n_true": n_true,
            **pop,
            **dict(zip(AREA_KEYS, area)),
        })

    df = pd.DataFrame(rows).set_index("array")
    n_max = int(df["n_gaussians"].max())
    cols = (["kind", "n_true", "n_gaussians", "H0", "Hclust", "dI", "dI_rel",
             "Ew", "Ewng", "w_ng", "H_ng"]
            + [c for k in range(1, n_max + 1) for c in (f"mu_{k}", f"w_{k}", f"H_{k}")]
            + AREA_KEYS)
    df = df.reindex(columns=cols)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index_label="array")
    return df, fractions


# ==============================================================================
# PLOTS
# ==============================================================================

def palette_index(df):
    """Position of each array in PALETTE: gauss_n -> 2(n-1), nongauss_n -> 2(n-1)+1."""
    return [2 * (n - 1) + int(k == "nongauss") for n, k in zip(df["n_true"], df["kind"])]


def point_colors(df, by_npeaks):
    if by_npeaks:
        return [NPEAKS_COLORS.get(n, mina.GREY) for n in df["n_true"]]
    return [PALETTE[i % len(PALETTE)] for i in palette_index(df)]


def scatter(df, y, x, by_npeaks):
    suffix = "_npeaks" if by_npeaks else ""
    # Letters (a, b, c, ...) identify the arrays in the plots coloured per array.
    letters = None if by_npeaks else [chr(ord("a") + i) for i in palette_index(df)]
    offsets = [LETTER_OFFSETS.get(name, (8, 8)) for name in df.index]
    fig, _ = mina.scatter_plot(
        df[x], df[y],
        colors=point_colors(df, by_npeaks),
        markers=np.where(df["kind"] == "gauss", ">", "o"),
        xlabel=LABELS[x], ylabel=LABELS[y],
        annotations=letters, offsets=offsets,
        save_path=os.path.join(SCAT_DIR, f"{y}_vs_{x}{suffix}.png"),
    )
    plt.close(fig)


def make_plots(df):
    df = df.copy()
    df["n_true"] = df["n_true"].astype(int)
    os.makedirs(SCAT_DIR, exist_ok=True)

    for y, x in PLOTS_BOTH:
        scatter(df, y, x, by_npeaks=False)
        scatter(df, y, x, by_npeaks=True)
    for y, x in PLOTS_ARRAY:
        scatter(df, y, x, by_npeaks=False)


def make_bars(fractions):
    """Cumulative explained-fraction bar plots, one per array and method."""
    os.makedirs(BAR_DIR, exist_ok=True)
    for name, per_method in fractions.items():
        for method, frac in per_method.items():
            if len(frac) == 0:  # No Gaussian fitted.
                continue
            fig, _ = mina.plot_cumulative_bars(
                frac, save_path=os.path.join(BAR_DIR, f"{name}_{method}.png"))
            plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    df, fractions = run_analysis()
    make_plots(df)
    make_bars(fractions)


if __name__ == "__main__":
    main()