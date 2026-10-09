"""
MInA: Micro-Information Analysis of descriptor distributions
============================================================

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
NPY_DIR = os.path.join(ROOT, "array", "descriptor")
OUT_CSV = os.path.join(ROOT, "csv", "MInA_descriptors.csv")
FIG_DIR = os.path.join(ROOT, "figures", "descriptors")
BAR_DIR = os.path.join(FIG_DIR, "00_structure_diagram")
SCAT_DIR = os.path.join(FIG_DIR, "01_information_structure_plot")

# ==============================================================================
# ANALYSIS PARAMETERS
# ==============================================================================

FIT_PARAMS = dict(
    bw_adjust=2,
    prominence=0.0001,
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

# Array name -> (legend label, colour); the order sets the order of the legend.
DESCRIPTORS = {
    "LENS_10":       ("LENS",     "#669900"),
    "sp_10_LENS_10": ("<LENS>",   "#a1c349"),
    "dist_5":        ("dist_5",   "#006699"),
    "sp_10_dist_5":  ("<dist_5>", "#3399cc"),
    "nn_10":         ("nn",       "#990066"),
    "sp_10_nn_10":   ("<nn>",     "#cc3399"),
    "OTO":           ("OTO",      "#ff6600"),
    "sp_10_OTO":     ("<OTO>",    "#ffcc00"),
    "vel":           ("vel",      "#ae2012"),
    "sp_10_vel":     ("<vel>",    "#62040a"),
}
NPEAKS_COLORS = {1: "#233d4d", 2: "#fe7f2d", 3: "#2a9d8f",
                 4: "#bad4aa", 5: "#c08552", 6: "#432371", 7: "#9b2226"}


# ==============================================================================
# ANALYSIS
# ==============================================================================

def run_analysis():
    npy_files = sorted(glob.glob(os.path.join(NPY_DIR, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {NPY_DIR}")

    rows, fractions = [], {}
    for path in npy_files:
        name = os.path.splitext(os.path.basename(path))[0]
        data = np.asarray(np.load(path), dtype=np.float64)

        pop, W = mina.mina_analysis(data, method="population", return_fractions=True,
                                    **FIT_PARAMS, **POP_PARAMS)
        *area, explained = mina.mina_analysis(data, method="area", return_fractions=True,
                                              **FIT_PARAMS)
        fractions[name] = {"area": explained, "population": W}

        rows.append({"array": name, **pop, **dict(zip(AREA_KEYS, area))})

    df = pd.DataFrame(rows).set_index("array")
    n_max = int(df["n_gaussians"].max())
    cols = (["n_gaussians", "H0", "Hclust", "dI", "dI_rel",
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

def marker(name):
    return "^" if name.startswith("sp_") else "o"


def point_colors(df, by_npeaks):
    if by_npeaks:
        return [NPEAKS_COLORS.get(int(n), mina.GREY) for n in df["n_gaussians"]]
    return [DESCRIPTORS.get(name, (name, mina.GREY))[1] for name in df.index]


def scatter(df, y, x, by_npeaks):
    suffix = "_npeaks" if by_npeaks else ""
    fig, _ = mina.scatter_plot(
        df[x], df[y],
        colors=point_colors(df, by_npeaks),
        markers=[marker(name) for name in df.index],
        xlabel=LABELS[x], ylabel=LABELS[y],
        save_path=os.path.join(SCAT_DIR, f"{y}_vs_{x}{suffix}.png"),
    )
    plt.close(fig)


def make_legends(df):
    # Legend of the plots coloured per array: listed arrays first, in the
    # order of DESCRIPTORS, then any unlisted array in grey.
    names = ([n for n in DESCRIPTORS if n in df.index]
             + [n for n in df.index if n not in DESCRIPTORS])
    fig = mina.legend_figure(
        labels=[DESCRIPTORS.get(n, (n, mina.GREY))[0] for n in names],
        colors=[DESCRIPTORS.get(n, (n, mina.GREY))[1] for n in names],
        markers=[marker(n) for n in names],
        save_path=os.path.join(SCAT_DIR, "legend.png"),
    )
    plt.close(fig)

    # Legend of the plots coloured by number of fitted Gaussians.
    npeaks = sorted(int(n) for n in df["n_gaussians"].unique())
    fig = mina.legend_figure(
        labels=[f"n = {n}" for n in npeaks],
        colors=[NPEAKS_COLORS.get(n, mina.GREY) for n in npeaks],
        markers=["o"] * len(npeaks),
        save_path=os.path.join(SCAT_DIR, "legend_npeaks.png"),
    )
    plt.close(fig)


def make_plots(df):
    os.makedirs(SCAT_DIR, exist_ok=True)
    for y, x in PLOTS_BOTH:
        scatter(df, y, x, by_npeaks=False)
        scatter(df, y, x, by_npeaks=True)
    for y, x in PLOTS_ARRAY:
        scatter(df, y, x, by_npeaks=False)
    make_legends(df)


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