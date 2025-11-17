# -*- coding: utf-8 -*-
"""
General Spectral Analysis Toolkit for Climate Mode Diagnostics
-------------------------------------------------------------

The evaluation is performed using both observational and reanalysis datasets, in combination with 18 CMIP6 piControl simulations. All analyses—including the extraction of principal component (PC) time series, the AR(1)-based Monte Carlo significance testing, and the frequency-domain diagnostics—are implemented using a unified workflow. The same procedure is applied consistently across all other climate modes (e.g., ENSO2, IPO, IOD, SDM, AMO, NAO, SAM), ensuring that the methodology remains fully comparable and reproducible among different modes.

It performs:

1. Bulk reading of model EOF/PC time series
2. Automatic variable detection from multiple candidate names
3. AR(1)-based Monte Carlo significance test
4. FFT spectrum plotting (subplots for each model)
5. Heatmap of significant spectral power between 1–20 years

Only two components need to be edited by the user:
    - MODEL_ORDER
    - PATHS

All other functions are fully generic.

Author: Yuxuan Lyu
"""


# ================================================================
# Imports
# ================================================================
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import matplotlib as mpl
from tqdm import tqdm


# ================================================================
# ========== USER CONFIG: Modify this for each climate mode ======
# ================================================================
DECODE_TIMES = False

VAR_CANDIDATES = ("EOFPC1_N", "EOFPC2_N", "PC1", "PC")

MODEL_ORDER = [
    "ERSSTv5", "CanESM5", "HadGEM3-GC31-LL", "EC-Earth3-CC",
    "CMCC-CM2-SR5", "CNRM-CM6-1", "GISS-E2-1-G", "CMCC-ESM2",
    "EC-Earth3", "E3SM1-0", "MIROC6", "MRI-ESM2-0",
    "HadGEM3-GC31-MM", "BCC-CSM2-MR", "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR", "ACCESS-ESM1-5", "ACCESS-CM2", "CESM2",
    "GFDL-CM4", "CIESM", "FGOALS-g3", "SAM0-UNICON", "CNRM-ESM2-1",
]

BASE = "/g/data/jk72/yl3496/Figures_for_sandeep/heat_map/ENSO"

PATHS = {
    name: f"{BASE}/eofpc1_{name}-enso_mave_norm.nc"
    for name in MODEL_ORDER if name != "ERSSTv5"
}
PATHS["ERSSTv5"] = f"{BASE}/eofpc1_ersst-enso_mave_norm.nc"


# ================================================================
# ========== Part 1: Reading PC time series ======================
# ================================================================
def open_nc_select_var(path, var_candidates=VAR_CANDIDATES, decode_times=False):
    """
    Open a netCDF file and return the best-matching variable.

    Parameters
    ----------
    path : str
        Path to the nc file.
    var_candidates : tuple
        Candidate variable names in priority order.
    decode_times : bool
        If False, avoids calendar decoding errors.

    Returns
    -------
    1D numpy array of the selected time series.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    ds = xr.open_dataset(path, decode_times=decode_times)

    # 1) Try candidate names
    for v in var_candidates:
        if v in ds.variables:
            da = ds[v]
            break
    else:
        # 2) If one variable → use it
        if len(ds.data_vars) == 1:
            da = ds[list(ds.data_vars)[0]]
        else:
            # 3) Guess a 1D/time-like variable
            candidates = [
                name for name, var in ds.data_vars.items()
                if var.ndim == 1 or ("time" in var.dims)
            ]
            if not candidates:
                raise KeyError(f"No suitable variable found in {path}")
            da = ds[candidates[0]]

    return np.asarray(da.squeeze().values).ravel()


def load_all_models(paths_map, model_order):
    """
    Load all model time series in the desired order.

    Returns
    -------
    model_dict : dict
        model_name → 1D array
    model_names : list
        list of successfully loaded models
    """
    model_dict = {}
    model_names = []

    for name in model_order:
        path = paths_map.get(name, None)
        if path is None:
            print(f"[WARN] No path configured for: {name}")
            continue

        try:
            arr = open_nc_select_var(path)
            model_dict[name] = arr
            model_names.append(name)
        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")

    return model_dict, model_names


# ================================================================
# ========== Part 2: AR(1) simulation & significance =============
# ================================================================
def generate_ar1_series(alpha, n, size=1):
    """
    Generate AR(1) synthetic series.

    x(t) = alpha * x(t-1) + e(t)

    Returns shape: (size, n)
    """
    e = np.random.normal(0, 1, (size, n))
    x = np.zeros_like(e)
    for t in range(1, n):
        x[:, t] = alpha * x[:, t-1] + e[:, t]
    return x


def estimate_edf_from_simulated_spectra(sim_spectra):
    """
    Estimate effective degrees of freedom (EDF) for each frequency.

    EDF(f) = (mean / std)^2

    Parameters
    ----------
    sim_spectra : array
        Shape: (num_simulations, num_freqs)

    Returns
    -------
    mean_power, std_power, edf_per_freq
    """
    mean_power = np.mean(sim_spectra, axis=0)
    std_power = np.std(sim_spectra, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        edf_per_freq = (mean_power / std_power) ** 2
    return mean_power, std_power, edf_per_freq


# ================================================================
# ========== Part 3: FFT spectrum plotting =======================
# ================================================================
def plot_all_fft(model_dict, model_names, fs=12, num_simulations=1000):
    """
    Plot FFT spectrum + Monte Carlo significance for all models.
    """
    ncols = 6
    nrows = int(np.ceil(len(model_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4*ncols, 3.5*nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, model in enumerate(tqdm(model_names)):
        x = model_dict[model]
        x = (x - np.mean(x)) / np.std(x)
        N = len(x)

        # AR(1) coefficient
        alpha = np.corrcoef(x[:-1], x[1:])[0, 1]

        # Simulate
        sim_data = generate_ar1_series(alpha, N, size=num_simulations)

        # FFT of real series
        freqs, power = periodogram(
            x, fs=fs, scaling="density", window="hann", detrend=False
        )
        valid = freqs > 0
        freqs, power = freqs[valid], power[valid]
        power = power / np.sum(power)
        periods = 1 / freqs

        # Simulated spectra
        sim_power = []
        for s in sim_data:
            f_sim, p = periodogram(s, fs=fs, scaling="density",
                                   window="hann", detrend=False)
            p = p[f_sim > 0]
            p = p / np.sum(p)
            sim_power.append(p)
        sim_power = np.asarray(sim_power)

        # EDF
        mean_power, std_power, edf = estimate_edf_from_simulated_spectra(sim_power)
        se = std_power / np.sqrt(edf)
        threshold = mean_power + se

        # Plot
        ax = axes[i]
        mask = (periods >= 1) & (periods <= 100)

        ax.plot(periods[mask], power[mask], color="black", label="Observed")
        ax.plot(periods[mask], threshold[mask], "--", color="red", label="Threshold")
        ax.plot(periods[mask], mean_power[mask], "--", color="blue", label="Mean AR(1)")

        ax.set_xscale("log")
        ax.set_xlim(1, 20)
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_xlabel("Period (Years)")
        ax.set_ylabel("Normalized Power")
        ax.set_title(model, fontsize=9)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    plt.suptitle("FFT Spectrum with Monte Carlo Significance", fontsize=16)
    plt.show()


# ================================================================
# ========== Part 4: Heatmap of significant power ================
# ================================================================
def compute_significant_heatmap(model_dict, model_names,
                                fs=12, num_simulations=1000):
    """
    Compute fraction of significant power for periods 1–20 years.

    Returns
    -------
    heatmap_data : 2D array (bins × models)
    period_bins  : edges
    """
    period_bins = np.arange(1, 21, 1)   # 1-year bins
    num_bins = len(period_bins) - 1
    num_models = len(model_names)

    heatmap = np.full((num_bins, num_models), np.nan)

    for j, model in enumerate(model_names):
        x = model_dict[model]
        x = (x - np.mean(x)) / np.std(x)
        N = len(x)

        alpha = np.corrcoef(x[:-1], x[1:])[0, 1]
        sim_data = generate_ar1_series(alpha, N, size=num_simulations)

        freqs, power = periodogram(x, fs=fs, scaling="density",
                                   window="hann", detrend=False)
        freqs, power = freqs[freqs > 0], power[freqs > 0]
        power = power / np.sum(power)
        periods = 1 / freqs

        sim_power = []
        for s in sim_data:
            f_sim, p = periodogram(s, fs=fs, scaling="density",
                                   window="hann", detrend=False)
            p = p[f_sim > 0]
            p = p / np.sum(p)
            sim_power.append(p)
        sim_power = np.asarray(sim_power)

        mean_power, std_power, edf = estimate_edf_from_simulated_spectra(sim_power)
        se = std_power / np.sqrt(edf)
        threshold = mean_power + se

        mask = (periods >= 1) & (periods < 20)
        periods_m = periods[mask]
        power_m = power[mask]
        threshold_m = threshold[mask]

        sig_mask = power_m > threshold_m
        if not np.any(sig_mask):
            continue

        sig_periods = periods_m[sig_mask]
        sig_power = power_m[sig_mask]
        total_sig = np.sum(sig_power)

        for i in range(num_bins):
            lo, hi = period_bins[i], period_bins[i+1]
            in_bin = (sig_periods >= lo) & (sig_periods < hi)
            if np.any(in_bin):
                heatmap[i, j] = np.sum(sig_power[in_bin]) / total_sig

    return heatmap, period_bins


def plot_heatmap(heatmap, period_bins, model_names, outpath=None):
    """
    Plot a 1–20 yr frequency-domain heatmap for all models.
    """
    cmap = mpl.cm.get_cmap("Reds").copy()
    cmap.set_bad(alpha=0.0)

    num_models = len(model_names)

    fig, ax = plt.subplots(figsize=(18, 8))

    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=0.5,
        extent=[-0.5, num_models - 0.5, period_bins[0], period_bins[-1]]
    )

    ax.set_ylim(0, 20)
    ax.set_yticks(np.arange(0, 22, 2))
    ax.set_xticks(np.arange(num_models))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)

    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("Period (Years)", fontsize=14)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Fraction of Significant Power", fontsize=14)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.show()


# ================================================================
# ========== Part 5: Main workflow ===============================
# ================================================================
def main():
    print("=== Loading model time series ===")
    model_dict, model_names = load_all_models(PATHS, MODEL_ORDER)

    print("=== Plotting FFT spectra ===")
    plot_all_fft(model_dict, model_names)

    print("=== Computing significant heatmap ===")
    heatmap, bins = compute_significant_heatmap(model_dict, model_names)

    print("=== Plotting heatmap ===")
    plot_heatmap(
        heatmap, bins, model_names,
        outpath="./heatmap.png"
    )


if __name__ == "__main__":
    main()
