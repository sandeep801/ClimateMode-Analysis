# -*- coding: utf-8 -*-
"""
General Taylor Diagram Analysis for Climate Modes
-------------------------------------------------

This script performs a standardized processing pipeline for any
climate mode (e.g., ENSO2, IPO, IOD, SDM, AMO, NAO, SAM). It includes:

1. Loading MMM and model regression fields
2. Applying a common mask across OBS/MMM/models
3. Plotting global maps (OBS + MMM + individual models)
4. Computing Taylor statistics (sdev, crmsd, correlation)
5. Plotting a customized Taylor diagram using SkillMetrics

Only three components need to be edited:
    - FILE_MMM
    - FILE_OBS_MODELS
    - MODEL_NAMES

All functions and plotting steps are generic.

Author: Yuxuan Lyu
"""

# ===============================================================
# Imports
# ===============================================================
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.cm as cm
import skill_metrics as sm


# ===============================================================
# 1. Data loading & masking
# ===============================================================
def load_and_mask_data(file_mmm, file_models, model_names):
    """
    Load MMM and model regression fields, rename dimensions,
    and generate a common non-NaN mask.

    Parameters
    ----------
    file_mmm : str
        Path to MMM regression field (2D)
    file_models : str
        Path to stacked OBS + model regression fields (3D)
    model_names : list
        List of model names including OBS as the first entry

    Returns
    -------
    mmm_masked : xarray.DataArray
        Masked MMM (lat, lon)
    models_masked : xarray.DataArray
        Masked OBS + models field (model, lat, lon)
    """

    # Load datasets
    ds_mmm = xr.open_dataset(file_mmm)
    ds_models = xr.open_dataset(file_models)

    # Extract variables (user should adjust variable names if needed)
    da_mmm = ds_mmm.MMM.rename({'LAT60_121': 'lat', 'LON121_281': 'lon'})
    da_models = ds_models.SLOPE.rename({
        ds_models.SLOPE.dims[0]: 'model',
        ds_models.SLOPE.dims[1]: 'lat',
        ds_models.SLOPE.dims[2]: 'lon'
    })

    # Create masks
    mask_mmm = ~np.isnan(da_mmm)
    mask_models = ~np.isnan(da_models)
    mask_all_models = mask_models.all(dim='model')

    # Combined mask (2D)
    mask_combined = mask_mmm & mask_all_models

    mmm_masked = da_mmm.where(mask_combined)
    models_masked = da_models.where(mask_combined)

    return mmm_masked, models_masked


# ===============================================================
# 2. Plotting global maps
# ===============================================================
def plot_maps(mmm, models, model_names, vmin=-1.0, vmax=1.0):
    """
    Plot global maps for:
        - OBS (first model)
        - MMM
        - All individual models

    Parameters
    ----------
    mmm : xarray.DataArray
        MMM 2D field (lat, lon)
    models : xarray.DataArray
        Stacked model fields (model, lat, lon)
    model_names : list
        List of model names
    """

    mpl.rcParams['font.family'] = 'DejaVu Sans'

    n_models = len(model_names)
    n_panels = n_models + 1
    n_cols = 5
    n_rows = int(np.ceil(n_panels / n_cols))

    fig = plt.figure(figsize=(4.8 * n_cols, 3.0 * n_rows))

    # 1. OBS
    ax = plt.subplot(n_rows, n_cols, 1, projection=ccrs.Robinson(central_longitude=210))
    models.isel(model=0).plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='coolwarm', vmin=vmin, vmax=vmax,
        add_colorbar=False
    )
    ax.set_title(f"OBS: {model_names[0]}", fontsize=10)
    ax.coastlines(); ax.add_feature(cfeature.LAND)

    # 2. MMM
    ax = plt.subplot(n_rows, n_cols, 2, projection=ccrs.Robinson(central_longitude=210))
    mmm.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='coolwarm', vmin=vmin, vmax=vmax,
        add_colorbar=False
    )
    ax.set_title("Multi-Model Mean", fontsize=10)
    ax.coastlines(); ax.add_feature(cfeature.LAND)

    # 3. All models
    for i in range(1, n_models):
        ax = plt.subplot(n_rows, n_cols, i+2, projection=ccrs.Robinson(central_longitude=210))
        models.isel(model=i).plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap='coolwarm', vmin=vmin, vmax=vmax,
            add_colorbar=False
        )
        ax.set_title(model_names[i], fontsize=10)
        ax.coastlines(); ax.add_feature(cfeature.LAND)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(smap, cax=cbar_ax)
    cbar.set_label("SST Anomaly", fontsize=12)

    plt.subplots_adjust(wspace=0.05, hspace=0.25, right=0.91)
    plt.show()


# ===============================================================
# 3. Compute Taylor diagram vectors
# ===============================================================
def compute_taylor_vectors(mmm, models, model_names):
    """
    Flatten masked OBS/MMM/models and compute Taylor statistics.

    Returns
    -------
    sdev : np.ndarray
    crmsd : np.ndarray
    ccoef : np.ndarray
    """

    # Common valid mask for OBS + MMM + models
    valid_mask = np.isfinite(models.isel(model=0)) & np.isfinite(mmm)
    for i in range(1, len(model_names)):
        valid_mask &= np.isfinite(models.isel(model=i))

    # Extract flattened arrays
    obs = models.isel(model=0).where(valid_mask, drop=True).values.flatten()
    mmm1d = mmm.where(valid_mask, drop=True).values.flatten()
    model_dict = {
        name: models.isel(model=i).where(valid_mask, drop=True).values.flatten()
        for i, name in enumerate(model_names[1:], start=1)
    }

    # Remove any leftover NaN
    mask = np.isfinite(obs)
    obs = obs[mask]
    mmm1d = mmm1d[mask]
    for k in model_dict:
        model_dict[k] = model_dict[k][mask]

    # Compute stats
    obs_std = np.std(obs)

    sdev = [1.0]   # OBS
    crmsd = [0.0]
    ccoef = [1.0]

    # MMM
    stats = sm.taylor_statistics(mmm1d, obs, 'data')
    sdev.append(stats['sdev'][1] / obs_std)
    crmsd.append(stats['crmsd'][1] / obs_std)
    ccoef.append(stats['ccoef'][1])

    # Models
    for v in model_dict.values():
        stats = sm.taylor_statistics(v, obs, 'data')
        sdev.append(stats['sdev'][1] / obs_std)
        crmsd.append(stats['crmsd'][1] / obs_std)
        ccoef.append(stats['ccoef'][1])

    return np.array(sdev), np.array(crmsd), np.array(ccoef)


# ===============================================================
# 4. Plot Taylor diagram
# ===============================================================
def plot_taylor_diagram(sdev, crmsd, ccoef, model_names, outpath=None):
    """
    Plot a customized Taylor diagram with SkillMetrics.
    """

    # Coordinate conversion
    def to_cartesian(std, corr):
        x = std * corr
        y = std * np.sqrt(1 - corr**2)
        return x, y

    x_vals, y_vals = zip(*[to_cartesian(s, r) for s, r in zip(sdev, ccoef)])

    # Colors
    cmap = plt.get_cmap("tab20", len(sdev))
    colors_list = [cmap(i) for i in range(len(sdev))]

    # Background diagram
    fig = plt.figure(figsize=(9, 9), dpi=600)
    sm.taylor_diagram(
        sdev, crmsd, ccoef,
        markerSize=1e-5,    # hide default markers
        colCOR="blue", styleCOR=":", widthCOR=1.0,
        titleRMS="off"
    )

    # Points
    plt.plot(x_vals[0], y_vals[0], "ko", markersize=12, label="OBS")
    plt.plot(
        x_vals[1], y_vals[1],
        marker="s", markersize=12,
        markerfacecolor=colors_list[1],
        markeredgecolor="black", markeredgewidth=1.2,
        linestyle="None",
        label="Multi-model Mean"
    )

    for i, (name, color) in enumerate(zip(model_names[1:], colors_list[2:])):
        plt.plot(
            x_vals[i+2], y_vals[i+2],
            "o", color=color, markersize=10,
            linestyle="None",
            label=name
        )

    # Legend
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    # Remove duplicates
    seen = set(); uniq_h = []; uniq_l = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            uniq_h.append(h); uniq_l.append(lab); seen.add(lab)

    N_PER_ROW = 5
    n_items = len(uniq_l)
    n_rows = int(np.ceil(n_items / N_PER_ROW))

    fig.subplots_adjust(bottom=max(0.18 + 0.035*n_rows, 0.26))
    leg = fig.legend(
        uniq_h, uniq_l,
        loc="lower center",
        ncol=N_PER_ROW,
        bbox_to_anchor=(0.5, 0.20),
        fontsize=9, frameon=False
    )

    # Bold fonts
    for t in ax.texts:
        t.set_fontweight("bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for t in leg.get_texts():
        t.set_fontweight("bold")

    # Save
    if outpath:
        plt.savefig(outpath, dpi=600, bbox_inches="tight")

    plt.show()


# ===============================================================
# 5. Main runner
# ===============================================================
def main():
    """
    Main workflow for executing Taylor diagram analysis
    on a single climate mode.
    """

    # User should edit these:
    FILE_MMM = "/g/data/jk72/yl3496/Figures_for_sandeep/data_for_taylor/ENSO1/regress_qtos_pres-enso1_mmm.nc"
    FILE_MODELS = "/g/data/jk72/yl3496/Figures_for_sandeep/data_for_taylor/ENSO1/regress_qtos_pres-enso1_obs-23models.nc"

    MODEL_NAMES = [
        'ERSSTv5', 'CanESM5', 'HadGEM3-GC31-LL', 'EC-Earth3-CC', 'CMCC-CM2-SR5',
        'CNRM-CM6-1', 'GISS-E2-1-G', 'CMCC-ESM2', 'EC-Earth3', 'E3SM1-0',
        'MIROC6', 'MRI-ESM2-0', 'HadGEM3-GC31-MM', 'BCC-CSM2-MR',
        'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'ACCESS-ESM1-5', 'ACCESS-CM2',
        'CESM2', 'GFDL-CM4', 'CIESM', 'FGOALS-g3', 'SAM0-UNICON', 'CNRM-ESM2-1'
    ]

    print("Loading & masking data ...")
    mmm, models = load_and_mask_data(FILE_MMM, FILE_MODELS, MODEL_NAMES)

    print("Plotting maps ...")
    plot_maps(mmm, models, MODEL_NAMES, vmin=-1, vmax=1)

    print("Computing Taylor statistics ...")
    sdev, crmsd, ccoef = compute_taylor_vectors(mmm, models, MODEL_NAMES)

    print("Plotting Taylor diagram ...")
    plot_taylor_diagram(
        sdev, crmsd, ccoef, MODEL_NAMES,
        outpath="./taylor_diagram.png"
    )


if __name__ == "__main__":
    main()
