# -*- coding: utf-8 -*-
"""Figure 4e only: two side-by-side panels for CAM-IAPICE and CAM-HadISST."""
from pathlib import Path
import os
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import Fig4e as panel_e
from Fig4common import (
    fix_time,
    prep_field,
    seasonal_mean,
    detrend_field,
    signflip_test,
    add_horizontal_colorbar,
)

warnings.filterwarnings("ignore")

# -------------------- Output and figure layout --------------------
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("FIG4_OUTPUT_DIR", str(HERE / "Fig4e_two_panel_outputs")))
EXPORT_DPI = int(os.environ.get("FIG4_EXPORT_DPI", "1500"))
SHOW_FIGURE = os.environ.get("FIG4_SHOW", "1").strip().lower() not in {"0", "false", "no"}
FIGURE_SIZE = (183 / 25.4, 105 / 25.4)  # landscape figure, two maps side by side
FIGURE_DPI = 150
MAP_WIDTH = 0.43
MAP_BOTTOM = 0.245
MAP_LEFTS = (0.03, 0.54)
COLORBAR_POSITION = [0.20, 0.105, 0.60, 0.032]

# -------------------- Data and period settings --------------------
TARGET_YEARS = np.arange(1940, 1943)
CLIM_YEARS = np.arange(1971, 2001)
MONTHS = [12, 1]
CROSSYEAR_MONTHS = [1, 2, 3]
DO_DETREND_ALL = False
MAP_EXTENT = [-180, 180, 30, 90]

# Use the same paths and significance settings as Fig4e.py.
PATH_TS_HAD = panel_e.PATH_TS_HAD
PATH_TS_REC = panel_e.PATH_TS_REC
PATH_MEAN_HAD = panel_e.PATH_MEAN_HAD
PATH_MEAN_REC = panel_e.PATH_MEAN_REC
PANEL_E_TICKS = [-0.8, -0.4, 0.4, 0.8]
WIND_REFERENCE = 1.0  # m s^-1, identical arrow reference for both panels


def main():
    # 1. Read the 20-member SAT fields and calculate each experiment's anomaly.
    print("Reading SAT members and calculating Dec-Jan anomalies...", flush=True)
    had_list, rec_list = [], []
    for member in range(20):
        had_path = f"{PATH_TS_HAD}{member:02d}.nc"
        rec_path = f"{PATH_TS_REC}{member:02d}.nc"

        ds = fix_time(xr.open_dataset(had_path))
        if "TS" not in ds:
            ds.close()
            raise KeyError(f"{had_path} 中没有 TS 变量。")
        had_list.append((prep_field(ds["TS"], MAP_EXTENT) - np.float32(273.15)).astype("float32").load())
        ds.close()

        ds = fix_time(xr.open_dataset(rec_path))
        if "TS" not in ds:
            ds.close()
            raise KeyError(f"{rec_path} 中没有 TS 变量。")
        rec_list.append((prep_field(ds["TS"], MAP_EXTENT) - np.float32(273.15)).astype("float32").load())
        ds.close()

    ensemble = xr.DataArray(np.arange(20), dims="ensemble", name="ensemble")
    had_members = xr.concat(had_list, dim=ensemble, join="exact")
    rec_members = xr.concat(rec_list, dim=ensemble, join="exact")
    had_members, rec_members = xr.align(had_members, rec_members, join="inner")

    ts_results = {}
    for exp_name, members in (("HadISST", had_members), ("reconstructed", rec_members)):
        annual = seasonal_mean(members, MONTHS, crossyear_months=CROSSYEAR_MONTHS)
        if DO_DETREND_ALL:
            annual = detrend_field(annual)
        target = annual.where(annual.time.isin(TARGET_YEARS), drop=True).mean("time", skipna=True)
        climate = annual.where(annual.time.isin(CLIM_YEARS), drop=True).mean("time", skipna=True)
        member_response = (target - climate).astype("float32")
        response = member_response.mean("ensemble", skipna=True).astype("float32")
        sig = signflip_test(
            member_response,
            alpha=panel_e.TEMP_MC_ALPHA,
            min_n=panel_e.TEMP_MC_MIN_N,
            n_random=panel_e.MC_N_RANDOM,
            seed=panel_e.MC_RANDOM_SEED,
            chunk_points=panel_e.MC_CHUNK_POINTS,
        )
        ts_results[exp_name] = {"response": response, "sig": sig}

    del had_list, rec_list, had_members, rec_members

    # 2. Read ensemble-mean PSL/U850/V850 and calculate anomalies per experiment.
    print("Reading PSL/U850/V850 ensemble means...", flush=True)
    ds_had = fix_time(xr.open_dataset(PATH_MEAN_HAD))
    ds_rec = fix_time(xr.open_dataset(PATH_MEAN_REC))
    psl_scale_had = 0.01 if float(ds_had["PSL"].mean()) > 5000 else 1.0
    psl_scale_rec = 0.01 if float(ds_rec["PSL"].mean()) > 5000 else 1.0
    surface_results = {"HadISST": {}, "reconstructed": {}}

    for exp_name, ds, psl_scale in (
        ("HadISST", ds_had, psl_scale_had),
        ("reconstructed", ds_rec, psl_scale_rec),
    ):
        for variable, scale in (("PSL", psl_scale), ("U850", 1.0), ("V850", 1.0)):
            field = prep_field(ds[variable], MAP_EXTENT, 2.0) * scale
            annual = seasonal_mean(field, MONTHS, crossyear_months=CROSSYEAR_MONTHS)
            if DO_DETREND_ALL:
                annual = detrend_field(annual)
            target = annual.where(annual.time.isin(TARGET_YEARS), drop=True).mean("time", skipna=True)
            climate = annual.where(annual.time.isin(CLIM_YEARS), drop=True).mean("time", skipna=True)
            surface_results[exp_name][variable] = (target - climate).astype("float32")

    ds_had.close()
    ds_rec.close()

    # 3. Prepare only panel-e data: CAM-IAPICE on the left, CAM-HadISST on the right.
    panel_data = []
    for exp_name, title in (("reconstructed", "CAM-IAPICE"), ("HadISST", "CAM-HadISST")):
        sat = ts_results[exp_name]
        surface = surface_results[exp_name]
        panel_data.append({
            "ts": sat["response"],
            "ts_sig": sat["sig"],
            "psl": surface["PSL"],
            "u850": surface["U850"],
            "v850": surface["V850"],
            "title": title,
        })

    # 4. Draw one figure with two panel-e maps and a shared SAT colorbar.
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor="white")
    map_height = MAP_WIDTH * fig.get_figwidth() / panel_e.SECTOR_MAP_ASPECT / fig.get_figheight()
    axes = [
        fig.add_axes([left, MAP_BOTTOM, MAP_WIDTH, map_height], projection=panel_e.SECTOR_PROJ)
        for left in MAP_LEFTS
    ]

    original_nice_reference = panel_e.nice_reference
    panel_e.nice_reference = lambda value: float(WIND_REFERENCE)
    try:
        image_a, labels_a = panel_e.draw_panel(axes[0], panel_data[0], panel="a", show_slp=True)
        image_b, labels_b = panel_e.draw_panel(axes[1], panel_data[1], panel="b", show_slp=True)
    finally:
        panel_e.nice_reference = original_nice_reference

    panel_e.remove_labels_outside(axes[0], labels_a)
    panel_e.remove_labels_outside(axes[1], labels_b)

    cax = fig.add_axes(COLORBAR_POSITION)
    add_horizontal_colorbar(
        cax,
        panel_e.TEMP_CMAP,
        panel_e.TEMP_NORM,
        panel_e.TEMP_LEVELS,
        "Surface Air Temperature (°C)",
        PANEL_E_TICKS,
        small_size=panel_e.SMALL_SIZE,
        medium_size=panel_e.MEDIUM_SIZE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "Fig4e_CAM-IAPICE_vs_CAM-HadISST_Dec-Jan"
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", facecolor="white", metadata={"Title": stem})
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=EXPORT_DPI, facecolor="white")
    fig.savefig(
        OUTPUT_DIR / f"{stem}.tiff",
        dpi=EXPORT_DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    print(f"Saved one two-panel figure to: {OUTPUT_DIR}", flush=True)

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
