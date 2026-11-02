# -*- coding: utf-8 -*-
"""Figure 2a: Oct-Nov sea-ice-area time series."""

import re
import warnings

import cftime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

PATH_SIC_RECON = r"E:\20century\data\sic_reconstructed_1901_2019_all_globe_data.nc"
PATH_SIC_HAD = r"E:\DATA\HadISST_ice.nc\HadISST_ice.nc"
SIC_RECON_VAR = "sic"
SIC_HAD_VAR = "sic"

TARGET_YEARS = range(1940, 1943)
TS_YEARS = range(1901, 2020)
MONTHS = (10, 11)
SIC_THRESHOLD = 0.15

TS_LAT_MIN, TS_LAT_MAX = 60, 85
TS_LON_MIN, TS_LON_MAX = 120, 240

IAPICE_COLOR = "#0072B2"
HADISST_COLOR = "#BDBDBD"
HADISST_SMOOTH_COLOR = "#BDBDBD"
LOWER_FILL_COLOR = "#1F5BFF"
UPPER_FILL_COLOR = "#E41A1C"
SHADE_COLOR = "#D9D9D9"
DIFF_TEXT_COLOR = "#333333"

RAW_LINE_WIDTH = 0.55
SMOOTH_LINE_WIDTH = 1.55
RUNNING_MEAN_WINDOW = 5
SHADE_START, SHADE_END = 1939.5, 1942.5
SHADE_ALPHA = 0.45
DIFF_TEXT_SIZE = 8
DIFF_TEXT_X, DIFF_TEXT_Y = 1943.5, 0.4

SMALL_SIZE = 8
MEDIUM_SIZE = 9
PANEL_SIZE = 10
FIG_DPI = 600
FIG_SIZE = (3.45, 2.75)
FIG_LEFT, FIG_RIGHT = 0.18, 0.98
FIG_TOP, FIG_BOTTOM = 0.94, 0.18
XMIN, XMAX = 1900, 2020
XTICKS = np.arange(1900, 2021, 20)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "bold",
    "axes.unicode_minus": True,
    "font.size": MEDIUM_SIZE,
    "axes.labelsize": MEDIUM_SIZE,
    "axes.titlesize": MEDIUM_SIZE,
    "xtick.labelsize": SMALL_SIZE,
    "ytick.labelsize": SMALL_SIZE,
    "legend.fontsize": MEDIUM_SIZE,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def add_year_month(ds, name, default_units=None, default_calendar="standard"):
    """Read actual year/month, preferring the start of time bounds."""
    if "time" not in ds:
        raise ValueError(f"{name} 没有 time 坐标")

    bounds_name = ds["time"].attrs.get("bounds")
    if bounds_name not in ds:
        bounds_name = next((x for x in ("time_bnds", "time_bounds", "time_bound") if x in ds), None)

    source = ds["time"]
    if bounds_name:
        bounds = ds[bounds_name]
        bound_dim = next(d for d in bounds.dims if d != "time")
        source = bounds.isel({bound_dim: 0})
        print(f"  {name}: 月份使用 {bounds_name} 起点")
    else:
        print(f"  {name}: 无 time_bnds，月份使用原 time")

    values = np.asarray(source.values)
    units = source.attrs.get("units") or ds["time"].attrs.get("units") or default_units
    calendar = source.attrs.get("calendar") or ds["time"].attrs.get("calendar") or default_calendar

    if np.issubdtype(values.dtype, np.datetime64):
        text = values.astype("datetime64[M]").astype(str)
        years = np.array([int(x[:4]) for x in text])
        months = np.array([int(x[5:7]) for x in text])
    elif values.dtype == object and values.size and hasattr(values.ravel()[0], "year"):
        years = np.array([int(x.year) for x in values.ravel()])
        months = np.array([int(x.month) for x in values.ravel()])
    else:
        if not units:
            raise ValueError(f"{name} 时间是数值型，但没有 units")
        match = re.match(r"\s*months?\s+since\s+([+-]?\d+)-(\d+)-(\d+)", str(units), re.I)
        if match:
            y0, m0 = int(match.group(1)), int(match.group(2))
            offsets = np.rint(values.astype(float).ravel()).astype(int)
            total = y0 * 12 + m0 - 1 + offsets
            years, months = total // 12, total % 12 + 1
        else:
            dates = cftime.num2date(
                values.ravel(), units=str(units), calendar=str(calendar),
                only_use_cftime_datetimes=True,
            )
            years = np.array([int(x.year) for x in dates])
            months = np.array([int(x.month) for x in dates])

    if len(years) != ds.sizes["time"]:
        raise ValueError(f"{name} 时间长度与数据不一致")
    keys = years * 100 + months
    if np.any(np.diff(keys) < 0) or len(np.unique(keys)) != len(keys):
        raise ValueError(f"{name} 月份轴非单调或存在重复月份")

    print(f"  {name}: {years[0]:04d}-{months[0]:02d} 至 {years[-1]:04d}-{months[-1]:02d}")
    return ds.assign_coords(year=("time", years), month=("time", months))

def annual_mean(da, years):
    out = []
    for year in years:
        pieces = []
        for month in MONTHS:
            part = da.where((da.year == year) & (da.month == month), drop=True)
            if part.sizes.get("time", 0) != 1:
                raise ValueError(f"缺少或重复月份: {year}-{month:02d}")
            pieces.append(part.isel(time=0, drop=True))
        out.append(xr.concat(pieces, dim="selected_month").mean("selected_month").expand_dims(year=[year]))
    return xr.concat(out, dim="year")

def draw_panel(ax_a):
    print("\n处理方式：不去趋势，直接使用原始年度数据。")
    print("\n读取 IAPICE 和 HadISST 海冰")

    rec_raw = xr.open_dataset(PATH_SIC_RECON, decode_times=False, mask_and_scale=True)
    had_raw = xr.open_dataset(PATH_SIC_HAD, decode_times=False, mask_and_scale=True)
    rec_ds = add_year_month(rec_raw, "IAPICE", "days since 1900-01-01", "360_day")
    had_ds = add_year_month(had_raw, "HadISST")

    rec_sic = rec_ds[SIC_RECON_VAR]
    had_sic = had_ds[SIC_HAD_VAR]
    data = []
    for da in (rec_sic, had_sic):
        rename = {}
        if "latitude" in da.dims:
            rename["latitude"] = "lat"
        if "longitude" in da.dims:
            rename["longitude"] = "lon"
        if rename:
            da = da.rename(rename)
        da = da.sortby("lat")
        da = da.assign_coords(lon=(da.lon + 360) % 360).sortby("lon")
        _, keep = np.unique(np.round(da.lon.values, 8), return_index=True)
        if len(keep) != da.sizes["lon"]:
            da = da.isel(lon=np.sort(keep))
        data.append(da)
    rec_sic, had_sic = data

    if float(rec_sic.max(skipna=True)) > 1.5:
        rec_sic = rec_sic / 100.0
    if float(had_sic.max(skipna=True)) > 1.5:
        had_sic = had_sic / 100.0

    rec_region = rec_sic.sel(lat=slice(TS_LAT_MIN, TS_LAT_MAX), lon=slice(TS_LON_MIN, TS_LON_MAX))
    had_region = had_sic.sel(lat=slice(TS_LAT_MIN, TS_LAT_MAX), lon=slice(TS_LON_MIN, TS_LON_MAX))

    ocean_mask = had_region.notnull().any("time")
    same_grid = (
        rec_region.sizes["lat"] == ocean_mask.sizes["lat"]
        and rec_region.sizes["lon"] == ocean_mask.sizes["lon"]
        and np.allclose(rec_region.lat.values, ocean_mask.lat.values)
        and np.allclose(rec_region.lon.values, ocean_mask.lon.values)
    )
    if same_grid:
        ocean_mask_rec = ocean_mask.assign_coords(lat=rec_region.lat, lon=rec_region.lon)
    else:
        ocean_mask_rec = ocean_mask.astype(float).interp(
            lat=rec_region.lat, lon=rec_region.lon, method="nearest"
        ) >= 0.5

    area = []
    for region in (rec_region, had_region):
        lat = np.asarray(region.lat, float)
        lon = np.asarray(region.lon, float)
        dlat = abs(np.diff(lat).mean())
        dlon = abs(np.diff(lon).mean())
        lat_b = np.r_[lat[0] - dlat / 2, (lat[:-1] + lat[1:]) / 2, lat[-1] + dlat / 2]
        lat_b = np.clip(lat_b, -90, 90)
        band = np.abs(np.diff(np.sin(np.deg2rad(lat_b))))
        one_area = 6371000.0 ** 2 * np.deg2rad(dlon) * band[:, None] * np.ones((len(lat), len(lon)))
        area.append(xr.DataArray(one_area, coords={"lat": lat, "lon": lon}, dims=("lat", "lon")))
    rec_area, had_area = area

    rec_sia_month = (
        rec_region.where(ocean_mask_rec).where(rec_region >= SIC_THRESHOLD) * rec_area
    ).sum(("lat", "lon"), skipna=True) / 1e12
    had_sia_month = (
        had_region.where(ocean_mask).where(had_region >= SIC_THRESHOLD) * had_area
    ).sum(("lat", "lon"), skipna=True) / 1e12

    rec_sia = annual_mean(rec_sia_month, TS_YEARS).values
    had_sia = annual_mean(had_sia_month, TS_YEARS).values
    year_values = np.asarray(list(TS_YEARS), dtype=int)
    target_mask = np.isin(year_values, list(TARGET_YEARS))
    diff_1940_1942 = np.nanmean(rec_sia[target_mask] - had_sia[target_mask])

    rec_smooth = xr.DataArray(rec_sia, dims="year").rolling(
        year=RUNNING_MEAN_WINDOW, center=True, min_periods=RUNNING_MEAN_WINDOW
    ).mean().values
    had_smooth = xr.DataArray(had_sia, dims="year").rolling(
        year=RUNNING_MEAN_WINDOW, center=True, min_periods=RUNNING_MEAN_WINDOW
    ).mean().values
    smooth_valid = np.isfinite(rec_smooth) & np.isfinite(had_smooth)

    ax_a.axvspan(SHADE_START, SHADE_END, color=SHADE_COLOR, alpha=SHADE_ALPHA, ec="none", zorder=0)
    ax_a.plot(year_values, rec_sia, color=IAPICE_COLOR, lw=RAW_LINE_WIDTH, alpha=0.28, zorder=1)
    had_raw_line, = ax_a.plot(year_values, had_sia, color=HADISST_COLOR, lw=RAW_LINE_WIDTH, alpha=0.48, zorder=1)
    had_raw_line.set_gid("hadisst_raw")
    ax_a.plot(year_values, rec_smooth, color=IAPICE_COLOR, lw=SMOOTH_LINE_WIDTH,
              label="IAPICE", zorder=5)
    had_smooth_line, = ax_a.plot(year_values, had_smooth, color=HADISST_SMOOTH_COLOR, lw=SMOOTH_LINE_WIDTH,
                                 label="HadISST", zorder=6)
    had_smooth_line.set_gid("hadisst_smooth")
    ax_a.legend(loc="lower left", ncol=1, handlelength=2.2, handletextpad=0.45,
                labelspacing=0.30, borderaxespad=0.35)

    ax_a.text(
        DIFF_TEXT_X, DIFF_TEXT_Y,
        f"1940–1942: {diff_1940_1942:.2f} ×10⁶ km²",
        transform=ax_a.get_xaxis_transform(), fontsize=DIFF_TEXT_SIZE,
        color=DIFF_TEXT_COLOR, ha="left", va="top", zorder=20,
    )

    ax_a.set_xlim(XMIN, XMAX)
    all_values = np.r_[rec_sia, had_sia]
    span = np.nanmax(all_values) - np.nanmin(all_values)
    if np.isclose(span, 0.0):
        span = max(abs(float(np.nanmax(all_values))), 1.0)
    pad = span * 0.08
    ax_a.set_ylim(np.nanmin(all_values) - pad, np.nanmax(all_values) + pad)
    ax_a.set_xticks(XTICKS)
    ax_a.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel(r"Sea Ice Area ($10^6$ km$^2$)")
    ax_a.tick_params(axis="both", which="major", length=2.5, width=0.6, pad=2)

    for spine in ax_a.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#333333")

    ax_a.text(0.0, 1.02, "(a)", transform=ax_a.transAxes, fontsize=PANEL_SIZE,
              fontweight="bold", ha="left", va="bottom", clip_on=False, zorder=100)

    rec_ds.close()
    had_ds.close()
    rec_raw.close()
    had_raw.close()


if __name__ == "__main__":
    # 独立运行：仅绘制海冰面积时间序列图。
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI, facecolor="white")
    ax = fig.add_axes([FIG_LEFT, FIG_BOTTOM, FIG_RIGHT - FIG_LEFT, FIG_TOP - FIG_BOTTOM])
    draw_panel(ax)
    plt.show()
