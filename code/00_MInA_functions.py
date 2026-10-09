"""
MInA: Micro-Information Analysis
================================

Core functions of the MInA framework for quantifying the non-Gaussian
character of a one-dimensional distribution.

Dependencies: numpy, scipy, matplotlib, seaborn, infomeasure.
"""
#%%
from __future__ import annotations

import glob
import os
from typing import Literal

import infomeasure as im
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from numpy import trapezoid
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Label assigned to samples that are not described by any fitted Gaussian.
RESIDUAL = -1


# ==============================================================================
# GAUSSIAN FIT
# ==============================================================================

def gaussian(x, h, mu, sigma):
    """Non-normalized Gaussian function.

    Parameters
    ----------
    x : array_like
        Points at which the function is evaluated.
    h : float
        Peak height.
    mu : float
        Mean.
    sigma : float
        Standard deviation.

    Returns
    -------
    ndarray
        Values of ``h * exp(-0.5 * ((x - mu) / sigma)**2)``.
    """
    return h * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def compute_kde(data_1d, bw_adjust):
    """Kernel density estimate of a one-dimensional dataset.

    The KDE is computed with ``seaborn.kdeplot`` and evaluated on the
    default seaborn grid; the figure used for the computation is closed
    before returning.

    Parameters
    ----------
    data_1d : ndarray
        One-dimensional input data.
    bw_adjust : float
        Multiplicative factor applied to the default bandwidth.

    Returns
    -------
    x_kde, y_kde : ndarray
        Evaluation grid and corresponding density values.
    """
    fig, ax = plt.subplots()
    sns.kdeplot(data_1d, bw_adjust=bw_adjust, ax=ax)
    x_kde = ax.lines[0].get_xdata()
    y_kde = ax.lines[0].get_ydata()
    plt.close(fig)
    return x_kde, y_kde


def find_kde_maxima(x_kde, y_kde, prominence=0.0001, peak_height_threshold=0.05):
    """Identify the significant maxima of a KDE.

    Local maxima are first selected according to their prominence; maxima
    whose height is lower than ``peak_height_threshold`` times the height of
    the highest maximum are then discarded.

    Parameters
    ----------
    x_kde, y_kde : ndarray
        KDE grid and density values.
    prominence : float, optional
        Minimum prominence required for a local maximum.
    peak_height_threshold : float, optional
        Minimum peak height, relative to the highest maximum.

    Returns
    -------
    ndarray
        Indices of the retained maxima, sorted by decreasing height.
    """
    peaks, _ = find_peaks(y_kde, prominence=prominence)
    if peaks.size == 0:
        return peaks
    heights = y_kde[peaks]
    peaks = peaks[heights >= peak_height_threshold * heights.max()]
    return peaks[np.argsort(y_kde[peaks])[::-1]]


def peak_fit_window(y_kde, pidx, use_pad=False):
    """Fitting window around a KDE maximum.

    Starting from the maximum, the window is extended to the left and to the
    right as long as the density is monotonically non-increasing, i.e. up to
    the nearest local minima on either side.

    Parameters
    ----------
    y_kde : ndarray
        KDE density values.
    pidx : int
        Index of the maximum.
    use_pad : bool, optional
        If True, the window is further extended on each side by one third of
        its width (at least one grid point), within the grid limits.

    Returns
    -------
    li, ri : int
        Left and right indices (inclusive) of the fitting window.
    """
    li = pidx
    while li > 0 and y_kde[li - 1] <= y_kde[li]:
        li -= 1
    ri = pidx
    while ri < len(y_kde) - 1 and y_kde[ri + 1] <= y_kde[ri]:
        ri += 1
    if use_pad:
        pad = max(1, (ri - li) // 3)
        li = max(0, li - pad)
        ri = min(len(y_kde) - 1, ri + pad)
    return li, ri


def fwhm_sigma(x_kde, y_kde, pidx):
    """Initial estimate of the standard deviation of a peak.

    The estimate is obtained from the full width at half maximum (FWHM) of
    the peak as ``sigma = FWHM / (2 * sqrt(2 * ln 2))``, with a lower bound
    of 1e-6.

    Parameters
    ----------
    x_kde, y_kde : ndarray
        KDE grid and density values.
    pidx : int
        Index of the maximum.

    Returns
    -------
    float
        Initial guess for the standard deviation.
    """
    ph = y_kde[pidx]
    lh = rh = pidx
    while lh > 0 and y_kde[lh] > ph / 2.0:
        lh -= 1
    while rh < len(y_kde) - 1 and y_kde[rh] > ph / 2.0:
        rh += 1
    return max((x_kde[rh] - x_kde[lh]) / (2.0 * np.sqrt(2 * np.log(2))), 1e-6)


def fit_single_gaussian(x_kde, y_kde, pidx, use_pad=False):
    """Least-squares fit of a single Gaussian to a KDE maximum.

    The fit is restricted to the window returned by ``peak_fit_window`` and
    initialized with the peak position, the peak height and the FWHM-based
    standard deviation. The mean is constrained within the KDE grid and the
    standard deviation within the grid range. If the optimization does not
    converge, the initial guess is returned.

    Parameters
    ----------
    x_kde, y_kde : ndarray
        KDE grid and density values.
    pidx : int
        Index of the maximum.
    use_pad : bool, optional
        Passed to ``peak_fit_window``.

    Returns
    -------
    mu, sigma, h : float
        Mean, standard deviation and height of the fitted Gaussian.
    """
    px, ph = x_kde[pidx], y_kde[pidx]
    li, ri = peak_fit_window(y_kde, pidx, use_pad)
    sigma0 = fwhm_sigma(x_kde, y_kde, pidx)
    try:
        popt, _ = curve_fit(
            gaussian, x_kde[li:ri + 1], y_kde[li:ri + 1],
            p0=[ph, px, sigma0],
            bounds=([0, x_kde[0], 1e-10], [np.inf, x_kde[-1], x_kde[-1] - x_kde[0]]),
            maxfev=5000,
        )
        h, mu, sigma = popt
    except RuntimeError:
        h, mu, sigma = ph, px, sigma0
    return float(mu), float(sigma), float(h)


def fit_gaussians(data, bw_adjust=2, prominence=0.0001,
                  peak_height_threshold=0.05, use_pad=False):
    """Fit one Gaussian to each significant maximum of the KDE of the data.

    Parameters
    ----------
    data : array_like
        Input data; flattened to one dimension.
    bw_adjust : float, optional
        Bandwidth adjustment factor of the KDE.
    prominence : float, optional
        Minimum prominence of the KDE maxima.
    peak_height_threshold : float, optional
        Minimum height of the KDE maxima, relative to the highest maximum.
    use_pad : bool, optional
        If True, the fitting windows are extended (see ``peak_fit_window``).

    Returns
    -------
    x_kde, y_kde : ndarray
        KDE grid and density values.
    gaussians : list of tuple
        Fitted parameters ``(mu, sigma, h)``, sorted by decreasing peak
        height.
    """
    data_1d = np.asarray(data, dtype=float).ravel()
    x_kde, y_kde = compute_kde(data_1d, bw_adjust)
    peaks = find_kde_maxima(x_kde, y_kde, prominence, peak_height_threshold)
    gaussians = [fit_single_gaussian(x_kde, y_kde, p, use_pad) for p in peaks]
    return x_kde, y_kde, gaussians


# ==============================================================================
# POPULATION ASSIGNMENT
# ==============================================================================
def _intersection_boundary(g_a, g_b, x_kde, y_kde, n_grid=4000):
    """Boundary between two fitted Gaussians.

    The boundary is defined as the intersection of the two Gaussians lying
    between their means (the one closest to the midpoint if several are
    present). If no intersection exists in that interval, the position of the
    KDE minimum between the means is used; if no KDE grid point lies in the
    interval, the midpoint between the means is returned.

    Parameters
    ----------
    g_a, g_b : array_like
        Gaussian parameters ``(mu, sigma, h)``.
    x_kde, y_kde : ndarray
        KDE grid and density values.
    n_grid : int, optional
        Number of points used to locate the intersection between the means.

    Returns
    -------
    float
        Position of the boundary.
    """
    lo, hi = sorted((g_a[0], g_b[0]))
    if hi - lo < 1e-12:
        return float(lo)
    xs = np.linspace(lo, hi, n_grid)
    d = gaussian(xs, g_a[2], g_a[0], g_a[1]) - gaussian(xs, g_b[2], g_b[0], g_b[1])
    idx = np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)
    if idx.size > 0:
        roots = xs[idx] - d[idx] * (xs[idx + 1] - xs[idx]) / (d[idx + 1] - d[idx])
        return float(roots[np.argmin(np.abs(roots - 0.5 * (lo + hi)))])
    m = (x_kde >= lo) & (x_kde <= hi)
    if np.any(m):
        return float(x_kde[m][np.argmin(y_kde[m])])
    return 0.5 * (lo + hi)


def assign_population(data_1d, gaussians, x_kde, y_kde, resid_thr=0.5):
    """Assign each sample to a Gaussian population or to the residual one.

    The data range is partitioned by the boundaries between consecutive
    Gaussians (sorted by mean), so that each sample has a single candidate
    Gaussian. A sample is assigned to its candidate if the value of that
    Gaussian at the sample position is at least ``resid_thr`` times the KDE
    density at the same position; otherwise it is labelled as ``RESIDUAL``.

    Parameters
    ----------
    data_1d : ndarray
        One-dimensional input data.
    gaussians : list of tuple
        Gaussian parameters ``(mu, sigma, h)``.
    x_kde, y_kde : ndarray
        KDE grid and density values.
    resid_thr : float, optional
        Minimum ratio between the Gaussian and the KDE density required for
        a sample to be assigned to the Gaussian.

    Returns
    -------
    ndarray of int
        Label of each sample: index of the assigned Gaussian in
        ``gaussians``, or ``RESIDUAL``.
    """
    labels = np.full(data_1d.size, RESIDUAL, dtype=int)
    if len(gaussians) == 0:
        return labels

    g = np.asarray(gaussians, dtype=float)
    order_mu = np.argsort(g[:, 0])
    bounds = np.array([
        _intersection_boundary(g[order_mu[j]], g[order_mu[j + 1]], x_kde, y_kde)
        for j in range(len(g) - 1)
    ])
    cand = order_mu[np.searchsorted(bounds, data_1d, side="right")]
    g_at = gaussian(data_1d, g[cand, 2], g[cand, 0], g[cand, 1])
    covered = g_at >= resid_thr * np.interp(data_1d, x_kde, y_kde)
    labels[covered] = cand[covered]
    return labels

# ==============================================================================
# DISCRETIZATION
# ==============================================================================

def _bin_1d(data_1d: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize a one-dimensional array into uniform bins.

    Parameters
    ----------
    data_1d : ndarray
        One-dimensional input data.
    n_bins : int
        Number of equally spaced bins between the minimum and maximum of the
        data.

    Returns
    -------
    ndarray of int
        Zero-based bin index of each sample.
    """
    edges = np.linspace(data_1d.min(), data_1d.max(), n_bins + 1)
    inds = np.digitize(data_1d, edges[:-1]) - 1
    inds = np.clip(inds, 0, n_bins - 1)
    return inds

# ==============================================================================
# ENTROPY (adapted from the dynsight package)
# ==============================================================================

def shannon(
    data: NDArray[np.float64],
    method: Literal["histo", "kl"],
    base: float = 2.0,
    n_neigh: int = 4,
) -> float:
    """Shannon entropy of a dataset.

    Parameters
    ----------
    data : ndarray
        Input data: discrete labels for ``method="histo"``, continuous values
        for ``method="kl"``.
    method : {"histo", "kl"}
        ``"histo"``: plug-in estimator for discrete data;
        ``"kl"``: Kozachenko-Leonenko k-nearest-neighbour estimator for
        continuous data.
    base : float, optional
        Logarithm base.
    n_neigh : int, optional
        Number of nearest neighbours used by the ``"kl"`` estimator.

    Returns
    -------
    float
        Entropy estimate. For ``method="kl"``, zero is returned when the
        dataset contains two samples or fewer.
    """
    if method not in ("histo", "kl"):
        msg = "method must be 'histo' or 'kl'."
        raise ValueError(msg)
    if method == "histo":
        return im.entropy(data, approach="discrete", base=base)
    min_samples = 2
    if data.shape[0] <= min_samples:
        return 0.0
    return im.entropy(data, approach="metric", k=n_neigh, base=base)


def info_gain(
    data: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    method: Literal["histo", "kl"],
    base: float = 2.0,
    n_neigh: int = 4,
) -> tuple[float, float, float, float]:
    """Information gain associated with a partition of the data.

    The information gain is defined as ``H0 - Hclust``, where ``H0`` is the
    entropy of the whole dataset and ``Hclust`` is the population-weighted
    average of the entropies of the individual clusters. Every distinct
    label, including ``RESIDUAL``, is treated as a cluster.

    Parameters
    ----------
    data : ndarray
        Input data (see ``shannon``).
    labels : ndarray of int
        Cluster label of each sample.
    method : {"histo", "kl"}
        Entropy estimator (see ``shannon``).
    base : float, optional
        Logarithm base.
    n_neigh : int, optional
        Number of nearest neighbours used by the ``"kl"`` estimator.

    Returns
    -------
    tuple of float
        ``(H0 - Hclust, (H0 - Hclust) / H0, H0, Hclust)``.
    """
    if data.shape[0] != labels.shape[0]:
        msg = (
            f"data ({data.shape}) and labels ({labels.shape}) "
            "must have same shape[0]"
        )
        raise RuntimeError(msg)
    if method not in ("histo", "kl"):
        msg = "method must be 'histo' or 'kl'."
        raise ValueError(msg)

    n_clusters = np.unique(labels).size
    frac, entr = np.zeros(n_clusters), np.zeros(n_clusters)

    if method == "histo":
        total_entropy = shannon(data, method="histo", base=base)
        for i, label in enumerate(np.unique(labels)):
            mask = labels == label
            frac[i] = np.sum(mask) / labels.size
            entr[i] = shannon(data[mask], method="histo", base=base)
    else:
        total_entropy = shannon(data, method="kl", base=base, n_neigh=n_neigh)
        for i, label in enumerate(np.unique(labels)):
            mask = labels == label
            frac[i] = np.sum(mask) / labels.size
            entr[i] = shannon(
                data[mask],
                method="kl",
                base=base,
                n_neigh=n_neigh,
            )

    clustered_entropy = np.dot(frac, entr)
    info_gain_val = total_entropy - clustered_entropy

    return (
        info_gain_val,
        info_gain_val / total_entropy,
        total_entropy,
        clustered_entropy,
    )


# ==============================================================================
# MInA
# ==============================================================================

def mina_analysis_area(data, bw_adjust=2, prominence=0.0001,
                       peak_height_threshold=0.05, use_pad=False,
                       return_fractions=False):
    """Area-based MInA.

    Gaussians are added one at a time, in order of decreasing height. At
    step k, the explained area is the area of the union of
    ``min(G_j, KDE)`` over the first k Gaussians, and the unexplained area
    ``dA_k`` is the KDE area minus the explained area.

    Parameters
    ----------
    data : array_like
        Input data; flattened to one dimension.
    bw_adjust, prominence, peak_height_threshold, use_pad
        Passed to ``fit_gaussians``.
    return_fractions : bool, optional
        If True, the explained area at each step is also returned.

    Returns
    -------
    num_peaks : int
        Number of fitted Gaussians.
    Ag : float
        Area under the cumulative explained-fraction curve, integrated over
        the number of Gaussians from 0 to ``num_peaks`` (with value 0 at
        zero Gaussians).
    Ang : float
        Area between the cumulative explained-fraction curve and unity over
        the same interval.
    h_ng : float
        Unexplained area after all Gaussians are included.
    h_g : float
        Explained area after all Gaussians are included.
    Eguf : float
        Sum of the unexplained areas over all steps.
    Egef : float
        Sum of the explained areas over all steps.
    explained : ndarray, optional
        Explained area at each step; returned only if
        ``return_fractions=True``.

    Notes
    -----
    If no Gaussian is fitted, ``num_peaks`` is 0 and all other scalar
    outputs are NaN.
    """
    x_kde, y_kde, gaussians = fit_gaussians(
        data, bw_adjust, prominence, peak_height_threshold, use_pad)
    kde_area = trapezoid(y_kde, x_kde)
    num_peaks = len(gaussians)

    if num_peaks == 0:
        nan = float("nan")
        out = (0, nan, nan, nan, nan, nan, nan)
        return (*out, np.array([])) if return_fractions else out

    union = np.zeros_like(y_kde)
    dA = []
    for mu, sigma, h in gaussians:
        union = np.maximum(union, np.minimum(gaussian(x_kde, h, mu, sigma), y_kde))
        dA.append(float(kde_area - trapezoid(union, x_kde)))
    dA = np.array(dA)
    explained = kde_area - dA

    x_ce = np.arange(0, num_peaks + 1)
    y_ce = np.insert(explained, 0, 0)
    Ag = float(trapezoid(y_ce, x_ce))
    Ang = float(trapezoid(1 - y_ce, x_ce))
    h_ng = float(dA[-1])
    h_g = float(explained[-1])
    Eguf = float(dA.sum())
    Egef = float(explained.sum())

    out = (num_peaks, Ag, Ang, h_ng, h_g, Eguf, Egef)
    return (*out, explained) if return_fractions else out


def mina_analysis_population(data, bw_adjust=2, prominence=0.0001,
                             peak_height_threshold=0.05, use_pad=False,
                             resid_thr=0.5,
                             entropy_method: Literal["histo", "kl"] = "histo",
                             n_bins=50, n_neigh=4, return_fractions=False):
    """Population-based MInA.

    Gaussians are added one at a time, in order of decreasing height, and at
    each step k the samples are assigned to the first k Gaussians or to the
    residual population (see ``assign_population``). The fraction of samples
    assigned to a Gaussian at step k is denoted ``W_k``. The partition
    obtained with all Gaussians is then characterized in terms of population
    weights, Shannon entropies and information gain.

    Parameters
    ----------
    data : array_like
        Input data; flattened to one dimension.
    bw_adjust, prominence, peak_height_threshold, use_pad
        Passed to ``fit_gaussians``.
    resid_thr : float, optional
        Passed to ``assign_population``.
    entropy_method : {"histo", "kl"}, optional
        Entropy estimator. With ``"histo"``, the data are discretized into
        ``n_bins`` uniform bins over the full data range, shared by all
        populations.
    n_bins : int, optional
        Number of bins used with ``entropy_method="histo"``.
    n_neigh : int, optional
        Number of nearest neighbours used with ``entropy_method="kl"``.
    return_fractions : bool, optional
        If True, the array of ``W_k`` is also returned.

    Returns
    -------
    result : dict
        ``n_gaussians``: number of fitted Gaussians;
        ``H0``: entropy of the whole dataset;
        ``mu_i``, ``w_i``, ``H_i``: mean of the i-th Gaussian (1-based, in
        order of decreasing height), fraction of samples assigned to it and
        entropy of its population;
        ``Hclust``: population-weighted average entropy of the partition;
        ``dI``, ``dI_rel``: absolute and relative information gain;
        ``Ew``: sum of ``W_k`` over all steps;
        ``Ewng``: sum of ``1 - W_k`` over all steps;
        ``w_ng``, ``H_ng``: fraction and entropy of the residual population.
    W : ndarray, optional
        ``W_k`` for k = 1, ..., ``n_gaussians``; returned only if
        ``return_fractions=True``.
    """
    if entropy_method not in ("histo", "kl"):
        raise ValueError("entropy_method must be 'histo' or 'kl'.")

    data_1d = np.asarray(data, dtype=float).ravel()
    x_kde, y_kde, gaussians = fit_gaussians(
        data_1d, bw_adjust, prominence, peak_height_threshold, use_pad)
    n_g = len(gaussians)

    # W[k-1]: fraction of samples assigned to one of the first k Gaussians.
    W = []
    labels = np.full(data_1d.size, RESIDUAL, dtype=int)
    for k in range(1, n_g + 1):
        labels = assign_population(data_1d, gaussians[:k], x_kde, y_kde, resid_thr)
        W.append(float(np.mean(labels != RESIDUAL)))
    W = np.array(W)

    # Discretization over the full data range, so that all populations share
    # the same bins.
    x_ent = _bin_1d(data_1d, n_bins) if entropy_method == "histo" else data_1d

    with np.errstate(divide="ignore", invalid="ignore"):
        dI, dI_rel, H0, Hclust = info_gain(
            x_ent, labels, method=entropy_method, base=2.0, n_neigh=n_neigh)
    H0, Hclust, dI = float(H0), float(Hclust), float(dI)
    dI_rel = float(dI_rel) if H0 != 0 else float("nan")

    result = dict(n_gaussians=n_g, H0=H0)
    for i in range(n_g):
        mask = labels == i
        result[f"mu_{i + 1}"] = gaussians[i][0]
        result[f"w_{i + 1}"] = float(mask.mean())
        result[f"H_{i + 1}"] = (
            float(shannon(x_ent[mask], method=entropy_method, base=2.0, n_neigh=n_neigh))
            if mask.any() else float("nan"))

    mask_ng = labels == RESIDUAL
    H_ng = (float(shannon(x_ent[mask_ng], method=entropy_method, base=2.0, n_neigh=n_neigh))
            if mask_ng.any() else float("nan"))

    result.update(
        Hclust=Hclust,
        dI=dI,
        dI_rel=dI_rel,
        Ew=float(W.sum()) if n_g > 0 else float("nan"),
        Ewng=float((1 - W).sum()) if n_g > 0 else float("nan"),
        w_ng=float(mask_ng.mean()),
        H_ng=H_ng,
    )
    return (result, W) if return_fractions else result


def mina_analysis(data, method: Literal["area", "population"] = "area", **kwargs):
    """Run MInA with the selected formulation.

    Parameters
    ----------
    data : array_like
        Input data.
    method : {"area", "population"}, optional
        ``"area"`` calls ``mina_analysis_area``; ``"population"`` calls
        ``mina_analysis_population``.
    **kwargs
        Keyword arguments passed to the selected function.

    Returns
    -------
    Output of ``mina_analysis_area`` or ``mina_analysis_population``.
    """
    if method == "area":
        return mina_analysis_area(data, **kwargs)
    if method == "population":
        return mina_analysis_population(data, **kwargs)
    raise ValueError("method must be 'area' or 'population'.")


 
# ==============================================================================
# PLOT
# ==============================================================================

# Colour used for axes, ticks and labels in all figures.
GREY = "#736f72"

def plot_cumulative_bars(fractions, bar_w=0.45, title=None, save_path=None):
    """Bar chart of the cumulative explained fraction versus the number of
    fitted Gaussians.

    The bar at n = 0 represents the reference value (unity). For n >= 1, the
    grey portion of each bar represents the explained fraction and the orange
    portion the unexplained fraction up to unity. With unit spacing, the area
    above the explained bars equals ``sum_k (1 - fractions_k)``.

    Parameters
    ----------
    fractions : array_like
        Explained fraction at each step: the explained area returned by
        ``mina_analysis(..., method="area", return_fractions=True)`` or
        ``W_k`` returned by
        ``mina_analysis(..., method="population", return_fractions=True)``.
    bar_w : float, optional
        Bar width.
    title : str, optional
        Figure title.
    save_path : str, optional
        If given, the figure is saved to this path at 300 dpi.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    area_above : float
        ``sum_k (1 - fractions_k)``.
    """
    y_bar = np.asarray(fractions, dtype=float)
    n = len(y_bar)
    x_bar = np.arange(1, n + 1)
    x_max_view = max(n, 2)
    area_above = float(np.sum(1 - y_bar))
    fill_alpha = 0.15
 
    fig, ax = plt.subplots(figsize=(6, 5))
 
    # Reference bar (n = 0).
    ax.bar(0, 1, width=bar_w, color='white', edgecolor="#242324", alpha=1, linewidth=2)
 
    # Explained fraction.
    ax.bar(x_bar, y_bar, width=bar_w, color='#736f72', alpha=fill_alpha, edgecolor='none')
    ax.bar(x_bar, y_bar, width=bar_w, color='none', edgecolor='#736f72',
           hatch='////', alpha=1, linewidth=2)
 
    # Unexplained fraction.
    ax.bar(x_bar, 1 - y_bar, bottom=y_bar, width=bar_w,
           color='#ff7f0e', alpha=fill_alpha, edgecolor='none')
    ax.bar(x_bar, 1 - y_bar, bottom=y_bar, width=bar_w,
           color='none', edgecolor='#ff7f0e', hatch='////', alpha=1, linewidth=2)
 
    ax.axhline(y=1, color='#736f72', linestyle='--', linewidth=2)
 
    # Axes formatting.
    ax.set_xlim(-0.6, x_max_view + 0.6)
    ax.set_xlabel("Number of fitted Gaussians (n)", fontsize=12, weight='bold', color='#736f72')
    ax.set_ylabel("Fraction", fontsize=14, weight='bold', color='#736f72')
    ax.set_xticks(np.arange(0, n + 1))
    if title is not None:
        ax.set_title(title, fontsize=12, weight='bold', color='#736f72')
 
    ax.tick_params(axis='x', width=1.5, color="#736f72")
    ax.tick_params(axis='y', width=1.5, color='#736f72')
    for spine in ['left', 'right', 'bottom', 'top']:
        ax.spines[spine].set_linewidth(1)
    ax.spines['left'].set_color('#736f72')
    for lbl in ax.get_xticklabels():
        lbl.set_size(11); lbl.set_color("#736f72"); lbl.set_weight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_size(11); lbl.set_color("#736f72"); lbl.set_weight("bold")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
 
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    plt.show()
    return fig, area_above


def style_ax(ax, labelsize=14, grid_alpha=0.4):
    """Apply the common formatting to a set of axes.

    Ticks, tick labels and the left spine are drawn in ``GREY``, tick labels
    are bold, and a dashed grid is added.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format.
    labelsize : float, optional
        Font size of the tick labels.
    grid_alpha : float, optional
        Opacity of the grid lines.
    """
    ax.tick_params(axis="x", width=1.5, color=GREY)
    ax.tick_params(axis="y", width=1.5, color=GREY)
    for spine in ["left", "right", "bottom", "top"]:
        ax.spines[spine].set_linewidth(1)
    ax.spines["left"].set_color(GREY)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_size(11); lbl.set_color(GREY); lbl.set_weight("bold")
    ax.tick_params(axis="x", labelsize=labelsize)
    ax.tick_params(axis="y", labelsize=labelsize)
    ax.grid(True, linestyle="--", alpha=grid_alpha)


def scatter_plot(x, y, colors, markers, xlabel, ylabel,
                 sizes=80, alphas=1.0, zorders=10,
                 annotations=None, offsets=None,
                 labelsize=14, grid_alpha=0.4, save_path=None):
    """Scatter plot with per-point style and optional text labels.

    Parameters
    ----------
    x, y : array_like
        Coordinates of the points.
    colors : sequence
        Colour of each point.
    markers : sequence of str
        Matplotlib marker of each point.
    xlabel, ylabel : str
        Axis labels.
    sizes, alphas, zorders : float or sequence, optional
        Marker size, opacity and drawing order, either common to all points
        or given per point.
    annotations : sequence of str, optional
        Text label of each point, drawn in the colour of the point.
    offsets : sequence of tuple, optional
        Offset (in points) of each text label from its point; the label is
        centred on the offset position. Defaults to (8, 8) for every label.
    labelsize, grid_alpha : float, optional
        Passed to ``style_ax``.
    save_path : str, optional
        If given, the figure is saved to this path at 300 dpi.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    n = len(x)
    sizes = np.broadcast_to(sizes, n)
    alphas = np.broadcast_to(alphas, n)
    zorders = np.broadcast_to(zorders, n)

    fig, ax = plt.subplots(figsize=(6, 5))
    for xi, yi, c, m, s, a, z in zip(x, y, colors, markers, sizes, alphas, zorders):
        ax.scatter(xi, yi, color=c, s=s, alpha=a, zorder=z, marker=m)

    if annotations is not None:
        if offsets is None:
            offsets = [(8, 8)] * len(annotations)
        for xi, yi, txt, c, off in zip(x, y, annotations, colors, offsets):
            ax.annotate(txt, (xi, yi), xytext=off, textcoords="offset points",
                        ha="center", va="center", fontsize=20, weight="bold",
                        color=c, zorder=11)

    ax.set_xlabel(xlabel, fontsize=14, weight="bold", color=GREY)
    ax.set_ylabel(ylabel, fontsize=14, weight="bold", color=GREY)
    style_ax(ax, labelsize=labelsize, grid_alpha=grid_alpha)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig, ax


def legend_figure(labels, colors, markers, save_path=None):
    """Stand-alone vertical legend, with each label in the colour of its marker.

    Parameters
    ----------
    labels : sequence of str
        Legend entries, from top to bottom.
    colors : sequence
        Colour of each entry.
    markers : sequence of str
        Matplotlib marker of each entry.
    save_path : str, optional
        If given, the figure is saved to this path at 300 dpi.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    handles = [Line2D([], [], linestyle="none", marker=m, markersize=12, color=c)
               for c, m in zip(colors, markers)]
    fig = plt.figure(figsize=(2, 0.45 * len(labels)))
    leg = fig.legend(handles, labels, loc="center", frameon=True, fancybox=True,
                     framealpha=1, edgecolor="#d9d9d9", handlelength=1.0, handletextpad=0.8,
                     labelspacing=0.6, borderpad=0.8, prop=dict(weight="bold", size=14))
    for txt, c in zip(leg.get_texts(), colors):
        txt.set_color(c)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig