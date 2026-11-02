# -*- coding: utf-8 -*-
from pathlib import Path
import os
import re
import warnings

import cftime
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter

from Fig4common import (
    load_panels, build_figure, add_horizontal_colorbar,
    running_mean_update, mean_arrays, stack_arrays,
    anomaly_from_year_series, false_sig_like, signflip_test,
)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

# ============================== Editable settings ==============================
# CAM成员编号沿用原脚本的对应关系：散点图成员编号 -> CAM文件成员编号（从0开始）
MEMBER2_CAM_INDEX = 1
MEMBER10_18_CAM_INDICES = [9, 17]
MEMBER13_CAM_INDEX = 12

OUTPUT_DIR = Path(os.environ.get("FIG4_MEMBER_OUTPUT_DIR", str(HERE / "Fig4m_outputs")))
EXPORT_DPI = int(os.environ.get("FIG4_EXPORT_DPI", "1200"))
SHOW_FIGURE = os.environ.get("FIG4_SHOW", "1").strip().lower() not in {"0", "false", "no"}

AC_RANGE = (-50.0, 50.0, 10.0)
BD_RANGE = (-5.0, 5.0, 1.0)
E_RANGE = (-5.0, 5.0, 1.0)
COLORBAR_TICKS = [
    [-50, -30, -10, 10, 30, 50],
    [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    [-50, -30, -10, 10, 30, 50],
    [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
]

# 仅用于整幅子图外侧的强调框，不改变地图内部已有的区域框。
HIGHLIGHT_COLOR = "#000000"
HIGHLIGHT_LINEWIDTH = 3.0
# Tight-bbox外再留出边距，使黑框完整包住子图标题、刻度标签和色标。
HIGHLIGHT_PAD = 0.020

# 仅对Member 13的a、c面板降低WAF矢量密度（相对原4、8间隔明显稀疏）。
MEMBER13_WAF_STRIDE_LAT = 6
MEMBER13_WAF_STRIDE_LON = 12

# member2垂直温度剖面设置（沿用上传的独立剖面脚本）
VERTICAL_TARGET_YEARS = np.arange(1940, 1943)
VERTICAL_BASELINE_YEARS = np.arange(1971, 2001)
VERTICAL_MONTHS = (10, 11)
VERTICAL_SECTION_LON_MIN = 120.0
VERTICAL_SECTION_LON_MAX = 240.0
VERTICAL_SECTION_LONGITUDE = 180.0
VERTICAL_LAT_MIN = 66.5
VERTICAL_LAT_MAX = 90.0
VERTICAL_PRESSURE_TOP = 300.0
VERTICAL_PRESSURE_BOTTOM = 1000.0
VERTICAL_PRESSURE_TICKS = [1000, 925, 850, 700, 500, 400, 300]
VERTICAL_LAT_TICKS = [70, 80, 90]
VERTICAL_DETREND = False
VERTICAL_INTERPOLATION_FACTOR = 8
VERTICAL_SECTION_INTERPOLATION_FACTOR = 4
VERTICAL_TEMP_VMIN = -1.8
VERTICAL_TEMP_VMAX = 1.8
VERTICAL_TEMP_INTERVAL = 0.3
VERTICAL_TEMP_ANCHORS = [
    "#21618C", "#75ADD1", "#DCEBF2", "#F7F7F2",
    "#F2D3C4", "#D97A63", "#9E2F3F",
]
VERTICAL_BOX_COLOR = "#D81B60"
VERTICAL_BOX_LINEWIDTH = 3.0
VERTICAL_SMALL_SIZE = 8
VERTICAL_MEDIUM_SIZE = 9
VERTICAL_LARGE_SIZE = 10
VERTICAL_AXIS_LINEWIDTH = 0.6
VERTICAL_TICK_LENGTH = 2.5
VERTICAL_TICK_WIDTH = 0.6
VERTICAL_POLE_LINEWIDTH = 0.45

# ===============================================================================

PANELS = load_panels("fig4_members")
a, b, c, d, e = PANELS
a.REGION_BOX = (120.0, 240.0, 60.0, 75.0)

# Member-difference colorbar ranges.
a.Z500_VMIN, a.Z500_VMAX, a.Z500_STEP = AC_RANGE
a.Z500_LEVELS, a.Z500_CMAP, a.Z500_NORM = a.make_cmap()
c.Z500_VMIN, c.Z500_VMAX, c.Z500_STEP = AC_RANGE
c.Z500_LEVELS, c.Z500_CMAP, c.Z500_NORM = c.make_cmap()
b.U_VMIN, b.U_VMAX, b.U_STEP = BD_RANGE
b.U_LEVELS, b.U_CMAP, b.U_NORM = b.make_cmap()
d.U_VMIN, d.U_VMAX, d.U_STEP = BD_RANGE
d.U_LEVELS, d.U_CMAP, d.U_NORM = d.make_cmap()

e.TEMP_VMIN, e.TEMP_VMAX, e.TEMP_STEP = E_RANGE
e.TEMP_LEVELS = np.round(np.arange(e.TEMP_VMIN, e.TEMP_VMAX + e.TEMP_STEP * .5, e.TEMP_STEP), 10)
e.TEMP_LEVELS = e.TEMP_LEVELS[~np.isclose(e.TEMP_LEVELS, 0.0)]
anchors = ["#21618C", "#4F8FC0", "#75ADD1", "#BFDCEB", "#FFFFFF", "#F2D3C4", "#D97A63", "#B84A4A", "#9E2F3F"]
base = mpl.colors.LinearSegmentedColormap.from_list("sat_base_override", anchors, N=256)
colors = base(np.linspace(0, 1, len(e.TEMP_LEVELS) - 1))
centers = (e.TEMP_LEVELS[:-1] + e.TEMP_LEVELS[1:]) / 2
colors[np.argmin(np.abs(centers)), :3] = mpl.colors.to_rgb("#FFFFFF")
e.TEMP_CMAP = mpl.colors.ListedColormap(colors, name="sat_center_white_override")
e.TEMP_NORM = mpl.colors.BoundaryNorm(e.TEMP_LEVELS, e.TEMP_CMAP.N, clip=False)


def seasonal_anomaly(mod, da, do_detrend=False):
    year = mod.seasonal(da)
    if do_detrend:
        year = mod.detrend(year)
    return anomaly_from_year_series(year, mod.TARGET_YEARS, mod.CLIM_YEARS)


def climatology(mod, da):
    year = mod.seasonal(da)
    climate = np.intersect1d(year.time, mod.CLIM_YEARS)
    if climate.size == 0:
        raise ValueError(f"{mod.__name__}: no valid climatology years")
    return year.sel(time=climate).mean("time", skipna=True).astype("float32")


def member_significance(mod, arrays):
    if len(arrays) == 1:
        return false_sig_like(arrays[0])
    alpha = getattr(mod, "Z500_MC_ALPHA", getattr(mod, "U_MC_ALPHA", getattr(mod, "TEMP_MC_ALPHA", 0.1)))
    return signflip_test(
        stack_arrays(arrays), alpha=alpha, min_n=len(arrays),
        n_random=mod.MC_N_RANDOM, seed=mod.MC_RANDOM_SEED,
        chunk_points=mod.MC_CHUNK_POINTS,
    )


def prepare_z_waf(mod, members, panel_name):
    print(f"Calculating panel {panel_name}: members={members} -> {mod.PERIOD_TITLE} Z500 + WAF", flush=True)
    umean = vmean = None
    for i, member in enumerate(members, 1):
        ds = mod.fix_time(xr.open_dataset(mod.member_path(mod.PATH_UV, "HadISST", member, mod.UV_SUFFIX)))
        u = mod.prep(mod.level(ds, ["U500","uwnd500","u500","U_500","U","u","UWND","uwnd","ua"], 500), margin=5).load()
        v = mod.prep(mod.level(ds, ["V500","vwnd500","v500","V_500","V","v","VWND","vwnd","va"], 500), margin=5).load()
        ds.close()
        umean = running_mean_update(umean, u, i)
        vmean = running_mean_update(vmean, v, i)

    basic_u, basic_v = climatology(mod, umean), climatology(mod, vmean)
    zvars = ["Z500","HGT500","zg500","z500","Z3_500","Z_500","HGT_500","Z3","z3","Z","z","zg","ZG","HGT","hgt","geopotential_height","height","PHI","phi"]
    diffs = []
    for member in members:
        ds = mod.fix_time(xr.open_dataset(mod.member_path(mod.PATH_ZT, "HadISST", member, mod.ZT_SUFFIX)))
        had = mod.prep(mod.level(ds, zvars, 500, height=True), margin=2).load(); ds.close()
        ds = mod.fix_time(xr.open_dataset(mod.member_path(mod.PATH_ZT, "reconstructed", member, mod.ZT_SUFFIX)))
        rec = mod.prep(mod.level(ds, zvars, 500, height=True), margin=2).load(); ds.close()
        had, rec = xr.align(had, rec, join="inner")
        diffs.append((seasonal_anomaly(mod, rec, True) - seasonal_anomaly(mod, had, True)).astype("float32").load())

    z500 = mean_arrays(diffs).astype("float32")
    wx, wy = mod.calculate_waf(z500, basic_u, basic_v)
    return {"z500": z500, "z500_sig": member_significance(mod, diffs),
            "waf_x": wx, "waf_y": wy, "title": mod.PERIOD_TITLE}


def prepare_wind(mod, members, panel_name):
    title = mod.PERIOD_TITLE
    pressure = float(mod.PRESSURE_HPA)
    tag = str(int(round(pressure))) if np.isclose(pressure, round(pressure)) else f"{pressure:g}"
    print(f"Calculating panel {panel_name}: members={members} -> {title} U{tag}", flush=True)
    uvars = [f"U{tag}",f"uwnd{tag}",f"u{tag}",f"U_{tag}","U","u","UWND","uwnd","ua"]
    vvars = [f"V{tag}",f"vwnd{tag}",f"v{tag}",f"V_{tag}","V","v","VWND","vwnd","va"]
    diffs = []
    had_u_mean = rec_u_mean = had_v_mean = None

    for i, member in enumerate(members, 1):
        ds = mod.fix_time(xr.open_dataset(mod.member_path("HadISST", member)))
        had_u = mod.prep(mod.level(ds, uvars, pressure), margin=2).load()
        had_v = mod.prep(mod.level(ds, vvars, pressure), margin=2).load(); ds.close()
        ds = mod.fix_time(xr.open_dataset(mod.member_path("reconstructed", member)))
        rec_u = mod.prep(mod.level(ds, uvars, pressure), margin=2).load(); ds.close()
        had_u, rec_u = xr.align(had_u, rec_u, join="inner")
        had_u_mean = running_mean_update(had_u_mean, had_u, i)
        rec_u_mean = running_mean_update(rec_u_mean, rec_u, i)
        had_v_mean = running_mean_update(had_v_mean, had_v, i)
        diffs.append((seasonal_anomaly(mod, rec_u, mod.DO_DETREND_ALL)
                      - seasonal_anomaly(mod, had_u, mod.DO_DETREND_ALL)).astype("float32").load())

    had_raw, rec_raw, v_raw = mod.seasonal(had_u_mean), mod.seasonal(rec_u_mean), mod.seasonal(had_v_mean)
    climate = np.intersect1d(np.intersect1d(had_raw.time, rec_raw.time), mod.CLIM_YEARS)
    uclim = had_raw.sel(time=climate).mean("time", skipna=True).astype("float32")
    vclim = v_raw.sel(time=climate).mean("time", skipna=True).astype("float32")
    hy, ry = (mod.detrend(had_raw), mod.detrend(rec_raw)) if mod.DO_DETREND_ALL else (had_raw, rec_raw)
    common = np.intersect1d(hy.time, ry.time)
    target, climate = np.intersect1d(common, mod.TARGET_YEARS), np.intersect1d(common, mod.CLIM_YEARS)
    u_field = ((ry.sel(time=target).mean("time") - ry.sel(time=climate).mean("time"))
               - (hy.sel(time=target).mean("time") - hy.sel(time=climate).mean("time"))).astype("float32")
    return {"u": u_field, "u_sig": member_significance(mod, diffs),
            "u_clim": uclim, "v_clim": vclim, "title": title, "pressure_hpa": pressure}


def prepare_e(members):
    print(f"Calculating panel e: members={members} -> Dec-Jan SAT + UV850", flush=True)
    ts_diffs = []
    for member in members:
        ds = e.fix_time(xr.open_dataset(Path(f"{e.PATH_TS_HAD}{member:02d}.nc")))
        had = (e.prep(ds["TS"], e.MAP_EXTENT) - np.float32(273.15)).astype("float32").load(); ds.close()
        ds = e.fix_time(xr.open_dataset(Path(f"{e.PATH_TS_REC}{member:02d}.nc")))
        rec = (e.prep(ds["TS"], e.MAP_EXTENT) - np.float32(273.15)).astype("float32").load(); ds.close()
        had, rec = xr.align(had, rec, join="inner")
        ts_diffs.append((seasonal_anomaly(e, rec, e.DO_DETREND_ALL)
                         - seasonal_anomaly(e, had, e.DO_DETREND_ALL)).astype("float32").load())

    u_diffs, v_diffs = [], []
    uvars = ["U850","uwnd850","u850","U_850","U","u","UWND","uwnd","ua"]
    vvars = ["V850","vwnd850","v850","V_850","V","v","VWND","vwnd","va"]
    for member in members:
        ds = d.fix_time(xr.open_dataset(d.member_path("HadISST", member)))
        hu = d.prep(d.level(ds, uvars, 850), margin=2).load()
        hv = d.prep(d.level(ds, vvars, 850), margin=2).load(); ds.close()
        ds = d.fix_time(xr.open_dataset(d.member_path("reconstructed", member)))
        ru = d.prep(d.level(ds, uvars, 850), margin=2).load()
        rv = d.prep(d.level(ds, vvars, 850), margin=2).load(); ds.close()
        hu, ru = xr.align(hu, ru, join="inner"); hv, rv = xr.align(hv, rv, join="inner")
        u_diffs.append((seasonal_anomaly(e, ru, e.DO_DETREND_ALL) - seasonal_anomaly(e, hu, e.DO_DETREND_ALL)).astype("float32").load())
        v_diffs.append((seasonal_anomaly(e, rv, e.DO_DETREND_ALL) - seasonal_anomaly(e, hv, e.DO_DETREND_ALL)).astype("float32").load())

    return {"ts": mean_arrays(ts_diffs).astype("float32"),
            "ts_sig": member_significance(e, ts_diffs),
            "u850": mean_arrays(u_diffs).astype("float32"),
            "v850": mean_arrays(v_diffs).astype("float32"),
            "title": "Dec-Jan"}



# ===============================================================================
# Member2 vertical-section helpers: adapted from the supplied standalone script
# ===============================================================================
VERTICAL_TEMP_NAMES = ("T", "ta", "temp", "temperature", "air", "air_temperature")


def vertical_add_year_month(ds, label):
    """Use time_bnds/time_bounds starts to identify the model month, as in the source script."""
    time_name = next((x for x in ("time", "Time", "TIME") if x in ds), None)
    if time_name is None:
        raise KeyError(f"{label} has no time coordinate")
    if time_name != "time":
        ds = ds.rename({time_name: "time"})

    bounds_name = ds["time"].attrs.get("bounds")
    if not bounds_name or bounds_name not in ds:
        bounds_name = next((x for x in ("time_bnds", "time_bounds") if x in ds), None)
    source = ds["time"]
    if bounds_name:
        bounds = ds[bounds_name]
        other_dim = next(d for d in bounds.dims if d != "time")
        source = bounds.isel({other_dim: 0})
        print(f"  {Path(str(label)).name}: use {bounds_name} start month")
    else:
        print(f"  {Path(str(label)).name}: no time bounds; use time coordinate")

    values = np.asarray(source.values)
    units = (source.attrs.get("units") or ds["time"].attrs.get("units")
             or ds["time"].encoding.get("units"))
    calendar = (source.attrs.get("calendar") or ds["time"].attrs.get("calendar")
                or ds["time"].encoding.get("calendar") or "standard")

    if np.issubdtype(values.dtype, np.datetime64):
        text_values = values.astype("datetime64[M]").astype(str)
        years = np.array([int(x[:4]) for x in text_values])
        months = np.array([int(x[5:7]) for x in text_values])
    elif values.dtype == object and values.size and hasattr(values.ravel()[0], "year"):
        years = np.array([int(x.year) for x in values.ravel()])
        months = np.array([int(x.month) for x in values.ravel()])
    else:
        if not units:
            raise ValueError(f"{label} numeric time coordinate has no units")
        match = re.match(r"\s*months?\s+since\s+([+-]?\d+)-(\d+)-(\d+)", str(units), re.I)
        if match:
            y0, m0 = int(match.group(1)), int(match.group(2))
            total = y0 * 12 + m0 - 1 + np.rint(values.astype(float).ravel()).astype(int)
            years = total // 12
            months = total % 12 + 1
        else:
            dates = cftime.num2date(
                values.ravel(), units=str(units), calendar=str(calendar),
                only_use_cftime_datetimes=True,
            )
            years = np.array([int(x.year) for x in dates])
            months = np.array([int(x.month) for x in dates])

    if np.any(np.diff(years * 100 + months) < 0):
        raise ValueError(f"{label} months are not in ascending order")
    return ds.assign_coords(year=("time", years), month=("time", months))


def standardize_vertical_temperature(da, label):
    dim_names = {
        "time": ("time", "Time", "TIME"),
        "lat": ("lat", "latitude", "Latitude", "LAT"),
        "lon": ("lon", "longitude", "Longitude", "LON"),
        "plev": ("plev", "lev", "level", "pressure", "pres", "isobaric", "isobaricInhPa"),
    }
    rename = {}
    for target in ("time", "lat", "lon", "plev"):
        old = next((x for x in dim_names[target] if x in da.dims), None)
        if old is None:
            raise KeyError(f"{label} has no {target} dimension")
        if old != target:
            rename[old] = target
    da = da.rename(rename)

    for dim in list(da.dims):
        if dim not in ("time", "lat", "lon", "plev"):
            if da.sizes[dim] == 1:
                da = da.isel({dim: 0}, drop=True)
            else:
                raise ValueError(f"{label} has unexpected dimension {dim}={da.sizes[dim]}")

    da = da.assign_coords(lon=np.mod(da.lon, 360.0)).sortby("lon").sortby("lat")
    _, keep = np.unique(np.round(da.lon.values, 8), return_index=True)
    if len(keep) != da.sizes["lon"]:
        da = da.isel(lon=np.sort(keep))

    pressure = np.asarray(da.plev.values, float)
    if str(da.plev.attrs.get("units", "")).lower() == "pa" or np.nanmax(np.abs(pressure)) > 2000:
        pressure = pressure / 100.0
    da = da.assign_coords(plev=pressure).sortby("plev")
    return da.sel(
        lat=slice(VERTICAL_LAT_MIN, VERTICAL_LAT_MAX),
        plev=slice(VERTICAL_PRESSURE_TOP, VERTICAL_PRESSURE_BOTTOM),
    )


def read_member2_vertical_temperature(filepath, label):
    """Calculate 1940–1942 Oct-Nov minus 1971–2000 Oct-Nov for one CAM member."""
    filepath = Path(filepath)
    print(f"\nRead {label} member2 vertical temperature: {filepath.name}")
    with xr.open_dataset(filepath, decode_times=False, mask_and_scale=True) as ds:
        ds = vertical_add_year_month(ds, filepath)
        var = next((x for x in VERTICAL_TEMP_NAMES if x in ds.data_vars), None)
        if var is None:
            for name, da0 in ds.data_vars.items():
                info = (str(da0.attrs.get("standard_name", "")) + " "
                        + str(da0.attrs.get("long_name", ""))).lower()
                if "temperature" in info:
                    var = name
                    break
        if var is None:
            raise KeyError(f"{filepath.name} has no temperature variable")

        da = standardize_vertical_temperature(ds[var], filepath.name)
        section = da.sel(lon=slice(VERTICAL_SECTION_LON_MIN, VERTICAL_SECTION_LON_MAX)).mean(
            "lon", skipna=True
        ).sortby("lat")
        section_lat = np.asarray(section.lat.values, float)
        section = section.rename({"lat": "section"}).assign_coords(
            section=np.arange(len(section_lat), dtype=int),
            section_lat=("section", section_lat),
            section_lon=("section", np.full(len(section_lat), VERTICAL_SECTION_LONGITUDE)),
        )

        annual_list, used_years = [], []
        for year in np.unique(section.year.values.astype(int)):
            oct_data = section.where(
                (section.year == year) & (section.month == VERTICAL_MONTHS[0]), drop=True
            )
            nov_data = section.where(
                (section.year == year) & (section.month == VERTICAL_MONTHS[1]), drop=True
            )
            if oct_data.sizes.get("time", 0) == 0 or nov_data.sizes.get("time", 0) == 0:
                continue
            annual_mean = (oct_data.mean("time", skipna=True)
                           + nov_data.mean("time", skipna=True)) / 2.0
            annual_list.append(annual_mean.expand_dims(year=[int(year)]))
            used_years.append(int(year))

        if not annual_list:
            raise ValueError(f"{filepath.name} has no complete Oct-Nov records")
        annual = xr.concat(
            annual_list, dim="year", coords="minimal", compat="override", combine_attrs="drop"
        ).sortby("year")

        # 与用户上传的 member2 剖面脚本一致：不去趋势。
        if VERTICAL_DETREND:
            x = xr.DataArray(annual.year.values.astype(float), coords={"year": annual.year}, dims="year")
            valid = np.isfinite(annual)
            n = valid.sum("year")
            x_mean = x.where(valid).sum("year", skipna=True) / n
            y_mean = annual.where(valid).sum("year", skipna=True) / n
            num = ((x - x_mean) * (annual - y_mean)).where(valid).sum("year", skipna=True)
            den = ((x - x_mean) ** 2).where(valid).sum("year", skipna=True)
            slope = (num / den).where((n >= 20) & (den > 0))
            annual = annual - slope * (x - x_mean)

        available = set(annual.year.values.astype(int).tolist())
        needed = np.r_[VERTICAL_TARGET_YEARS, VERTICAL_BASELINE_YEARS]
        missing = [int(x) for x in needed if int(x) not in available]
        if missing:
            raise ValueError(f"{filepath.name} is missing years: {missing}")
        anomaly = (
            annual.sel(year=VERTICAL_TARGET_YEARS).mean("year")
            - annual.sel(year=VERTICAL_BASELINE_YEARS).mean("year")
        ).transpose("plev", "section").load()

    # Retain the original script's logarithmic-pressure and horizontal interpolation.
    p0 = anomaly.plev.values.astype(float)
    x0 = anomaly.section.values.astype(float)
    fine_p = np.exp(np.linspace(
        np.log(p0.min()), np.log(p0.max()),
        (len(p0) - 1) * VERTICAL_INTERPOLATION_FACTOR + 1,
    ))
    fine_x = np.linspace(
        x0.min(), x0.max(),
        (len(x0) - 1) * VERTICAL_SECTION_INTERPOLATION_FACTOR + 1,
    )
    work = anomaly.assign_coords(log_plev=("plev", np.log(p0))).swap_dims({"plev": "log_plev"})
    if "plev" in work.coords:
        work = work.drop_vars("plev")
    result = work.interp(
        log_plev=np.log(fine_p), section=fine_x, method="linear"
    ).rename({"log_plev": "plev"})
    result = result.assign_coords(
        plev=("plev", fine_p),
        section=("section", fine_x),
        section_lat=("section", anomaly.section_lat.interp(section=fine_x).values),
        section_lon=("section", anomaly.section_lon.interp(section=fine_x, method="nearest").values),
    ).transpose("plev", "section")
    print(
        f"{label} member2 temperature anomaly range: "
        f"{float(result.min(skipna=True)):.3f} to {float(result.max(skipna=True)):.3f} °C"
    )
    return result


def load_member2_vertical_sections():
    # Use exactly the same member-file mapping and ZT path as the Fig4m calculations.
    had_file = a.member_path(a.PATH_ZT, "HadISST", MEMBER2_CAM_INDEX, a.ZT_SUFFIX)
    rec_file = a.member_path(a.PATH_ZT, "reconstructed", MEMBER2_CAM_INDEX, a.ZT_SUFFIX)
    rec_temp = read_member2_vertical_temperature(rec_file, "CAM-IAPICE")
    had_temp = read_member2_vertical_temperature(had_file, "CAM-HadISST")
    had_temp = had_temp.interp(plev=rec_temp.plev, section=rec_temp.section, method="linear")
    had_temp = had_temp.assign_coords(section_lat=rec_temp.section_lat, section_lon=rec_temp.section_lon)
    diff_temp = (rec_temp - had_temp).rename("CAM-IAPICE_minus_CAM-HadISST")
    print(
        f"Member2 vertical difference range: {float(diff_temp.min(skipna=True)):.3f} "
        f"to {float(diff_temp.max(skipna=True)):.3f} °C"
    )
    return rec_temp, had_temp, diff_temp


def vertical_temperature_style():
    levels = np.round(
        np.arange(VERTICAL_TEMP_VMIN, VERTICAL_TEMP_VMAX + VERTICAL_TEMP_INTERVAL * 0.5,
                  VERTICAL_TEMP_INTERVAL), 10
    )
    levels = levels[~np.isclose(levels, 0.0)]
    base = LinearSegmentedColormap.from_list("member2_vertical_temperature", VERTICAL_TEMP_ANCHORS, N=256)
    colors = base(np.linspace(0.0, 1.0, len(levels) - 1))
    colors[(levels[:-1] < 0.0) & (levels[1:] > 0.0), :3] = [1.0, 1.0, 1.0]
    cmap = ListedColormap(colors, name="member2_vertical_center_white")
    cmap.set_bad("white")
    return levels, cmap


def draw_member2_vertical_panels(fig, axes, vertical_data, colorbar_ax):
    levels, cmap = vertical_temperature_style()
    panel_data = [
        (axes[0], vertical_data[0], "CAM-IAPICE", "(a)", True),
        (axes[1], vertical_data[1], "CAM-HadISST", "(b)", False),
        (axes[2], vertical_data[2], "CAM-IAPICE − CAM-HadISST", "(c)", False),
    ]
    filled_ref = None
    for ax, temp, experiment, panel_label, draw_box in panel_data:
        x = temp.section.values.astype(float)
        lat = temp.section_lat.values.astype(float)
        pressure = temp.plev.values.astype(float)
        values = np.asarray(temp.values, float)
        tick_pos = [float(x[np.nanargmin(np.abs(lat - value))]) for value in VERTICAL_LAT_TICKS]
        pole_x = float(x[np.nanargmax(lat)])
        x_left, x_right = float(np.nanmin(x)), float(np.nanmax(x))
        filled = ax.contourf(
            x, pressure, np.ma.masked_invalid(values), levels=levels,
            cmap=cmap, extend="both",
        )
        if panel_label == "(a)":
            filled_ref = filled
        ax.set_yscale("log")
        ax.set_ylim(VERTICAL_PRESSURE_BOTTOM, VERTICAL_PRESSURE_TOP)
        ax.set_yticks(VERTICAL_PRESSURE_TICKS)
        ax.set_yticklabels([f"{v:g}" for v in VERTICAL_PRESSURE_TICKS])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(
            axis="y", which="major", direction="out",
            length=VERTICAL_TICK_LENGTH, width=VERTICAL_TICK_WIDTH,
            labelsize=VERTICAL_SMALL_SIZE,
        )
        ax.tick_params(axis="y", which="minor", left=False, right=False, length=0)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f"{v}°N" for v in VERTICAL_LAT_TICKS], fontsize=VERTICAL_SMALL_SIZE)
        ax.tick_params(
            axis="x", which="major", direction="out",
            length=VERTICAL_TICK_LENGTH, width=VERTICAL_TICK_WIDTH,
            labelsize=VERTICAL_SMALL_SIZE,
        )
        ax.set_xlim(x_left, x_right)
        ax.axvline(pole_x, color="black", lw=VERTICAL_POLE_LINEWIDTH, ls="--")

        # Keep the original pink pressure-range box on panel (a).
        if draw_box:
            box_effect = [
                pe.Stroke(linewidth=VERTICAL_BOX_LINEWIDTH + 0.9, foreground="white"),
                pe.Normal(),
            ]
            for xx, yy in (
                ([x_left, x_right], [500.0, 500.0]),
                ([x_left, x_right], [1000.0, 1000.0]),
                ([x_left, x_left], [500.0, 1000.0]),
                ([x_right, x_right], [500.0, 1000.0]),
            ):
                ax.plot(
                    xx, yy, color=VERTICAL_BOX_COLOR,
                    linewidth=VERTICAL_BOX_LINEWIDTH,
                    solid_capstyle="butt", zorder=50, clip_on=True,
                    path_effects=box_effect,
                )

        ax.text(
            0.0, 1.02, panel_label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=VERTICAL_LARGE_SIZE,
            fontweight="bold", clip_on=False,
        )
        ax.text(
            0.985, 1.06, experiment, transform=ax.transAxes,
            ha="right", va="top", fontsize=VERTICAL_MEDIUM_SIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.5),
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(VERTICAL_AXIS_LINEWIDTH)
            spine.set_color("#333333")

    axes[0].set_ylabel("Pressure Level (hPa)", fontsize=VERTICAL_MEDIUM_SIZE)
    axes[1].tick_params(left=False, labelleft=False)
    axes[2].tick_params(left=False, labelleft=False)

    cb = fig.colorbar(
        filled_ref, cax=colorbar_ax, orientation="vertical", ticks=levels, extend="both"
    )
    cb.set_label("Temperature (°C)", fontsize=VERTICAL_MEDIUM_SIZE, labelpad=3)
    cb.ax.minorticks_off()
    cb.ax.tick_params(
        axis="y", which="major", labelsize=VERTICAL_SMALL_SIZE,
        direction="out", length=VERTICAL_TICK_LENGTH,
        width=0.5, pad=1.5,
    )
    cb.outline.set_linewidth(0.5)


def add_highlight_frame(fig, panel_ax, colorbar_ax=None):
    """Frame the complete panel, including title, tick labels and colorbar labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # get_tightbbox includes visible axis decorations (tick labels, tick marks,
    # axis labels and panel-heading text), unlike get_position().
    display_bboxes = [panel_ax.get_tightbbox(renderer)]
    if colorbar_ax is not None:
        display_bboxes.append(colorbar_ax.get_tightbbox(renderer))

    figure_bboxes = [
        bbox.transformed(fig.transFigure.inverted())
        for bbox in display_bboxes if bbox is not None
    ]
    x0 = min(b.x0 for b in figure_bboxes) - HIGHLIGHT_PAD
    y0 = min(b.y0 for b in figure_bboxes) - HIGHLIGHT_PAD
    x1 = max(b.x1 for b in figure_bboxes) + HIGHLIGHT_PAD
    y1 = max(b.y1 for b in figure_bboxes) + HIGHLIGHT_PAD

    frame = Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        transform=fig.transFigure, fill=False,
        edgecolor=HIGHLIGHT_COLOR, linewidth=HIGHLIGHT_LINEWIDTH,
        joinstyle="miter", clip_on=False, zorder=2000,
    )
    fig.add_artist(frame)
    return frame


def build_member2_figure(vertical_data, map_data, waf_style):
    """Output 1: vertical sections a-c above a reduced, left-aligned panel d."""
    fig = plt.figure(figsize=(10.8, 8.2), dpi=180, facecolor="white")
    gs = fig.add_gridspec(
        2, 4, left=0.065, right=0.945, top=0.955, bottom=0.075,
        width_ratios=[1.0, 1.0, 1.0, 0.065],
        height_ratios=[1.0, 1.25], wspace=0.30, hspace=0.02,
    )

    # First row: the three vertical temperature sections and their shared colorbar.
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2], sharey=ax_a)
    axes = [ax_a, ax_b, ax_c]

    section_cax = fig.add_subplot(gs[0, 3])
    p = section_cax.get_position()
    section_cax.set_position([
        p.x0 + (p.width - 0.018) / 2.0,
        p.y0 + 0.17 * p.height,
        0.018,
        0.66 * p.height,
    ])
    draw_member2_vertical_panels(fig, axes, vertical_data, section_cax)

    # Second row: shrink panel d uniformly in both width and height (76%),
    # preserving the map projection aspect ratio and keeping its left/bottom
    # alignment. The colorbar retains the same width-to-panel ratio, thickness,
    # and vertical gap as before.
    ax_d = fig.add_subplot(gs[1, 0:2], projection=a.PROJ)
    ax_d.set_anchor("W")
    original_pos = ax_d.get_position()
    PANEL_D_SCALE = 0.76
    panel_d_width = original_pos.width * PANEL_D_SCALE
    panel_d_height = original_pos.height * PANEL_D_SCALE
    ax_d.set_position([
        original_pos.x0, original_pos.y0,
        panel_d_width, panel_d_height,
    ])
    map_image = a.draw_panel(ax_d, map_data, style=waf_style, panel="d")

    # Keep the original, shorter colorbar length; align its left edge with panel d.
    fig.canvas.draw()
    p = ax_d.get_position()
    map_cax = fig.add_axes([
        p.x0, p.y0 - 0.055,
        p.width, 0.014,
    ])
    add_horizontal_colorbar(
        map_cax, a.Z500_CMAP, a.Z500_NORM, a.Z500_LEVELS,
        "500 hPa Geopotential Height (gpm)", COLORBAR_TICKS[0],
        small_size=a.SMALL_SIZE, medium_size=a.MEDIUM_SIZE,
    )

    # Add black outer frames to panels (a) and (d).
    # Panel (a): include its title and tick labels.
    add_highlight_frame(fig, ax_a)
    # Panel (d): include the map, tick labels, and horizontal colorbar.
    add_highlight_frame(fig, ax_d, map_cax)

    fig.canvas.draw()
    return fig


def build_abc_figure(data_a, data_b, data_c, waf_style):
    """Output 2: only original Fig4 panels a-c, with a black outer frame around c."""
    fig = plt.figure(figsize=(183 / 25.4, 132 / 25.4), dpi=180, facecolor="white")
    gs = fig.add_gridspec(
        2, 2, left=0.045, right=0.975, top=0.96, bottom=0.10,
        wspace=0.28, hspace=0.56,
    )
    ax_a = fig.add_subplot(gs[0, 0], projection=a.PROJ)
    ax_b = fig.add_subplot(gs[0, 1], projection=b.PROJ)
    ax_c = fig.add_subplot(gs[1, 0], projection=c.PROJ)
    ax_a.set_anchor("E")
    ax_b.set_anchor("W")
    ax_c.set_anchor("E")

    a.draw_panel(ax_a, data_a, style=waf_style, panel="a")
    b.draw_panel(ax_b, data_b, panel="b")
    c.draw_panel(ax_c, data_c, style=waf_style, panel="c")

    fig.canvas.draw()
    cbar_axes = []
    specs = [
        (ax_a, a.Z500_CMAP, a.Z500_NORM, a.Z500_LEVELS,
         "500 hPa Geopotential Height (gpm)", COLORBAR_TICKS[0]),
        (ax_b, b.U_CMAP, b.U_NORM, b.U_LEVELS,
         f"{b.PRESSURE_HPA:g} hPa Zonal Wind (m s$^{{-1}}$)", COLORBAR_TICKS[1]),
        (ax_c, c.Z500_CMAP, c.Z500_NORM, c.Z500_LEVELS,
         "500 hPa Geopotential Height (gpm)", COLORBAR_TICKS[2]),
    ]
    for panel_ax, cmap, norm, levels, label, ticks in specs:
        p = panel_ax.get_position()
        # Thicken all three colorbars vertically while preserving the
        # original gap between each panel and the colorbar.
        cbar_height = 0.020
        panel_to_cbar_gap = 0.025  # original gap: 0.037 - 0.012
        cax = fig.add_axes([
            p.x0,
            p.y0 - panel_to_cbar_gap - cbar_height,
            p.width,
            cbar_height,
        ])
        add_horizontal_colorbar(
            cax, cmap, norm, levels, label, ticks,
            small_size=a.SMALL_SIZE, medium_size=a.MEDIUM_SIZE,
        )
        cbar_axes.append(cax)

    add_highlight_frame(fig, ax_c, cbar_axes[2])
    return fig


mpl.rcParams.update({"savefig.facecolor": "white", "savefig.edgecolor": "none"})


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=EXPORT_DPI, facecolor="white")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Member 2: vertical-section a-c + original Fig4 panel-a as panel d
    # ------------------------------------------------------------------
    print("=" * 80, "\nPreparing member2: CAM member index 1", flush=True)
    vertical_data = load_member2_vertical_sections()
    member2_map_d = prepare_z_waf(a, [MEMBER2_CAM_INDEX], "d")

    # ------------------------------------------------------------------
    # 2) Mean of members 10 and 18: only original panels a-c
    # ------------------------------------------------------------------
    print("=" * 80, "\nPreparing mean of members 10 and 18: CAM indices [9, 17]", flush=True)
    mean_a = prepare_z_waf(a, MEMBER10_18_CAM_INDICES, "a")
    mean_b = prepare_wind(b, MEMBER10_18_CAM_INDICES, "b")
    mean_c = prepare_z_waf(c, MEMBER10_18_CAM_INDICES, "c")

    # ------------------------------------------------------------------
    # 3) Member 13: retain the complete original a-e workflow
    # ------------------------------------------------------------------
    print("=" * 80, "\nPreparing member13: CAM member index 12", flush=True)
    member13_a = prepare_z_waf(a, [MEMBER13_CAM_INDEX], "a")
    member13_b = prepare_wind(b, [MEMBER13_CAM_INDEX], "b")
    member13_c = prepare_z_waf(c, [MEMBER13_CAM_INDEX], "c")
    member13_d = prepare_wind(d, [MEMBER13_CAM_INDEX], "d")
    member13_e = prepare_e([MEMBER13_CAM_INDEX])

    # One common WAF style for all Z500+WAF panels, matching the former script logic.
    waf_items = [member2_map_d, mean_a, mean_c, member13_a, member13_c]
    waf_style = a.waf_style(*waf_items)

    figs = []

    # Output 1: member2, vertical profiles a-c + Z500/WAF map d.
    fig_member2 = build_member2_figure(vertical_data, member2_map_d, waf_style)
    save_figure(fig_member2, "Fig4_member2_vertical_abc_plus_d")
    figs.append(fig_member2)

    # Output 2: members 10 and 18 mean, panels a-c only; c has a black outer frame.
    fig_mean = build_abc_figure(mean_a, mean_b, mean_c, waf_style)
    save_figure(fig_mean, "Fig4_member10_18_mean_abc")
    figs.append(fig_mean)

    # Output 3: member13, full a-e; reduce WAF arrow density only for this figure.
    old_waf_sampling = (a.WAF_4, a.WAF_6, c.WAF_4, c.WAF_6)
    a.WAF_4 = c.WAF_4 = MEMBER13_WAF_STRIDE_LAT
    a.WAF_6 = c.WAF_6 = MEMBER13_WAF_STRIDE_LON
    try:
        fig_member13 = build_figure(
            PANELS, member13_a, member13_b, member13_c, member13_d, member13_e,
            waf_style=waf_style, colorbar_ticks=COLORBAR_TICKS, show_slp=False,
        )
    finally:
        a.WAF_4, a.WAF_6, c.WAF_4, c.WAF_6 = old_waf_sampling
    # build_figure creates axes a,b,c,d,e first, followed by their five colorbar axes.
    if len(fig_member13.axes) < 10:
        raise RuntimeError(
            "Unexpected Fig4common.build_figure axes order; cannot identify panel (e) and its colorbar."
        )
    add_highlight_frame(fig_member13, fig_member13.axes[4], fig_member13.axes[9])
    save_figure(fig_member13, "Fig4_member13_full_abcde")
    figs.append(fig_member13)

    if SHOW_FIGURE:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
