"""
MInA: Micro-Information Analysis of the denoised SOAP components
================================================================

"""
import importlib
import os
import sys
import warnings

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
NPY_PATH = os.path.join(ROOT, "array", "soap", "soap_denoised.npy")
OUT_CSV = os.path.join(ROOT, "csv", "MInA_soap_denoised.csv")
FIG_DIR = os.path.join(ROOT, "figures", "soap_denoised")
BAR_DIR = os.path.join(FIG_DIR, "00_structure_diagram")
SCAT_DIR = os.path.join(FIG_DIR, "01_information_structure_plot")

COMP_AXIS = 2
# Maximum memory (GB) used by a block of components read from disk. Each block
# requires one pass over the file, so a larger budget means fewer passes.
MEM_BUDGET_GB = 2.0

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
# (y, x) pairs, drawn with the highlighted components only.
PLOTS_ARRAY = [("Eguf", "Egef"), ("Ewng", "Ew")]

# Highlighted component index -> colour; the order sets the order of the legend.
HIGHLIGHT = {
    3: "#001219", 62: "#A2067B", 63: "#5f4bb6", 113: "#8ecae6", 197: "#126782",
    237: "#a1c349", 328: "#6a8532", 423: "#f6b049", 458: "#eb5e28", 519: "#691810",
}
NPEAKS_COLORS = {1: "#233d4d", 2: "#fe7f2d", 3: "#2a9d8f", 4: "#bad4aa",
                 5: "#c08552", 6: "#432371", 7: "#9b2226", 8: "#f64d50"}

PLOT_STYLE = dict(labelsize=12, grid_alpha=0.5)


# ==============================================================================
# ANALYSIS
# ==============================================================================

def iter_components(path):
    """Yield (k, data[:, :, k]) for every component of the array at path.

    The file is memory-mapped and read in blocks of consecutive components
    whose size is set by MEM_BUDGET_GB. Each block is copied once into a
    component-major contiguous array, so that every component is a
    contiguous slice and no further copy is made when it is flattened.
    """
    arr = np.load(path, mmap_mode="r")
    if arr.ndim <= COMP_AXIS:
        raise ValueError(f"{path}: shape {arr.shape} has no axis {COMP_AXIS}")
    n_comp = arr.shape[COMP_AXIS]
    comp_bytes = arr.nbytes // n_comp
    chunk = max(1, int(MEM_BUDGET_GB * 1024**3 // comp_bytes))
    for k0 in range(0, n_comp, chunk):
        k1 = min(k0 + chunk, n_comp)
        block = np.ascontiguousarray(np.moveaxis(arr[:, :, k0:k1], COMP_AXIS, 0))
        for j in range(k1 - k0):
            yield k0 + j, block[j]
        del block


def n_components(path):
    return np.load(path, mmap_mode="r").shape[COMP_AXIS]


def progress(n, total, width=30):
    """Single-line progress bar on stderr."""
    frac = n / total
    bar = "#" * int(width * frac)
    sys.stderr.write(f"\r[{bar:<{width}}] {n}/{total} {100 * frac:5.1f}%")
    if n == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def run_analysis():
    if not os.path.isfile(NPY_PATH):
        raise FileNotFoundError(NPY_PATH)

    total = n_components(NPY_PATH)
    rows, fractions = [], {}
    for n, (k, data) in enumerate(iter_components(NPY_PATH), 1):
        pop, W = mina.mina_analysis(data, method="population", return_fractions=True,
                                    **FIT_PARAMS, **POP_PARAMS)
        *area, explained = mina.mina_analysis(data, method="area", return_fractions=True,
                                              **FIT_PARAMS)
        if k in HIGHLIGHT:
            fractions[k] = {"area": explained, "population": W}
        rows.append({"component": k, **pop, **dict(zip(AREA_KEYS, area))})
        progress(n, total)

    df = pd.DataFrame(rows).set_index("component")
    n_max = int(df["n_gaussians"].max())
    cols = (["n_gaussians", "H0", "Hclust", "dI", "dI_rel",
             "Ew", "Ewng", "w_ng", "H_ng"]
            + [c for k in range(1, n_max + 1) for c in (f"mu_{k}", f"w_{k}", f"H_{k}")]
            + AREA_KEYS)
    df = df.reindex(columns=cols)

    missing = [k for k in HIGHLIGHT if k not in df.index]
    if missing:
        warnings.warn(f"Highlighted components not in the array: {missing}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index_label="component")
    return df, fractions


# ==============================================================================
# PLOTS
# ==============================================================================

def point_style(df, by_npeaks):
    """Colours, sizes, opacities and drawing orders of the points."""
    n = len(df)
    if by_npeaks:
        colors = [NPEAKS_COLORS.get(int(p), mina.GREY) for p in df["n_gaussians"]]
        return colors, np.full(n, 0.8), np.full(n, 5)
    hl = np.array([k in HIGHLIGHT for k in df.index])
    colors = [HIGHLIGHT.get(k, mina.GREY) for k in df.index]
    return colors, np.where(hl, 1.0, 0.2), np.where(hl, 10, 5)


def scatter(df, y, x, by_npeaks):
    suffix = "_npeaks" if by_npeaks else ""
    colors, alphas, zorders = point_style(df, by_npeaks)
    fig, _ = mina.scatter_plot(
        df[x], df[y],
        colors=colors, markers=["o"] * len(df),
        sizes=70, alphas=alphas, zorders=zorders,
        xlabel=LABELS[x], ylabel=LABELS[y],
        save_path=os.path.join(SCAT_DIR, f"{y}_vs_{x}{suffix}.png"),
        **PLOT_STYLE,
    )
    plt.close(fig)


def make_legends(df):
    comps = [k for k in HIGHLIGHT if k in df.index]
    fig = mina.legend_figure(
        labels=[str(k) for k in comps],
        colors=[HIGHLIGHT[k] for k in comps],
        markers=["o"] * len(comps),
        save_path=os.path.join(SCAT_DIR, "legend.png"),
    )
    plt.close(fig)

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
    """Cumulative explained-fraction bar plots of the highlighted components."""
    os.makedirs(BAR_DIR, exist_ok=True)
    for k, per_method in fractions.items():
        for method, frac in per_method.items():
            if len(frac) == 0:  # No Gaussian fitted.
                continue
            fig, _ = mina.plot_cumulative_bars(
                frac, save_path=os.path.join(BAR_DIR, f"component_{k}_{method}.png"))
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