# -*- coding: utf-8 -*-
"""Figure 2c: Oct-Nov surface-air-temperature response."""

import os
import re
import warnings

import cftime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle, Polygon

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER

warnings.filterwarnings("ignore")

SAT_HAD_BASE = r"E:\20century\data\sea_ice_exps_HadISST\2deg_TS_HadISST_"
SAT_REC_BASE = r"E:\20century\data\sea_ice_exps_reconstructed\2deg_TS_reconstructed_"
SAT_VAR = "TS"
SAT_EXPECTED_MEMBERS = 20

TARGET_YEARS = range(1940, 1943)
REF_YEARS = range(1971, 2001)
SAT_MAP_LAT_MIN, SAT_MAP_LAT_MAX = 60.0, 90.0

SAT_LEVELS = np.arange(-10.0, 10.001, 2.0)
SAT_LEVELS = SAT_LEVELS[~np.isclose(SAT_LEVELS, 0.0)]
SAT_CBAR_TICKS = (-10.0, -6.0, -2.0, 2.0, 6.0, 10.0)
SAT_CBAR_LABEL = "Surface Air Temperature (°C)"
SAT_NEGATIVE_COLORS = ("#15508D", "#3885BC", "#81BAD8", "#CAE1EE")
SAT_POSITIVE_COLORS = ("#FCD5BF", "#EE9677", "#CB4942", "#930E26")
SAT_CMAP_LIGHTEN = 0.10

SAT_SIG_ALPHA = 0.05
SAT_SIG_N_RANDOM = 10000
SAT_SIG_SEED = 20240618
SAT_SIG_BATCH = 200
SAT_SIG_MIN_MEMBERS = 20
SAT_CONSISTENCY_RATIO = 0.80
SAT_CONSISTENCY_MIN_COUNT = int(np.ceil(SAT_EXPECTED_MEMBERS * SAT_CONSISTENCY_RATIO))
SAT_SIGNIFICANT_CONTOUR_COLOR = "#4B1D6B"
SAT_SIGNIFICANT_CONTOUR_LINEWIDTH = 1.4
SAT_SIGNIFICANT_CONTOUR_OPACITY = 1.0

SMALL_SIZE = 8
MEDIUM_SIZE = 9
PANEL_SIZE = 10
FIG_DPI = 600
FIG_SIZE = (3.25, 3.10)
LAYOUT_HEIGHT_RATIOS = [1.0, 0.075]
LAYOUT_LEFT, LAYOUT_RIGHT = 0.06, 0.97
LAYOUT_TOP, LAYOUT_BOTTOM = 0.94, 0.10
LAYOUT_HSPACE = 0.12
CBAR_VERTICAL_FACTOR = 0.75
CBAR_THICKNESS_INCH = 0.12
CBAR_LABELPAD = 8.0

COAST_COLOR = "0.35"
COAST_LINEWIDTH = 0.45
GRID_COLOR = "0.72"
GRID_LINEWIDTH = 0.30
GRID_ALPHA = 0.70
GRID_ZORDER = 80
MAP_FRAME_LINEWIDTH = 0.6

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

def sat_surface_field(ds, label):
    if SAT_VAR not in ds:
        raise KeyError(f"{label} 中没有 {SAT_VAR}")

    da = ds[SAT_VAR]
    rename = {}
    for old, new in (
        ("latitude", "lat"), ("longitude", "lon"),
        ("Latitude", "lat"), ("Longitude", "lon"),
        ("grid_latitude", "lat"), ("grid_longitude", "lon"),
        ("nav_lat", "lat"), ("nav_lon", "lon"),
        ("y", "lat"), ("x", "lon"),
    ):
        if old in da.dims or old in da.coords:
            rename[old] = new
    if rename:
        da = da.rename(rename)
    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(f"{label} 没有lat/lon坐标，dims={da.dims}")

    da = da.sortby("lat")
    da = da.assign_coords(lon=np.mod(da.lon, 360.0)).sortby("lon")
    _, keep = np.unique(np.round(da.lon.values, 8), return_index=True)
    if len(keep) != da.sizes["lon"]:
        da = da.isel(lon=np.sort(keep))

    sample = da.isel(
        time=slice(0, min(12, da.sizes["time"])),
        lat=slice(0, min(10, da.sizes["lat"])),
        lon=slice(0, min(10, da.sizes["lon"])),
    )
    if float(sample.mean(skipna=True)) > 100:
        da = da - np.float32(273.15)
    return da.astype("float32")

def sat_octnov_series(da, label):
    out, used_years = [], []
    for year in np.unique(da.year.values.astype(int)):
        oct_data = da.where((da.year == year) & (da.month == 10), drop=True)
        nov_data = da.where((da.year == year) & (da.month == 11), drop=True)
        if oct_data.sizes.get("time", 0) != 1 or nov_data.sizes.get("time", 0) != 1:
            continue
        one_year = (
            oct_data.isel(time=0, drop=True) + nov_data.isel(time=0, drop=True)
        ) / 2.0
        out.append(one_year.expand_dims(year=[int(year)]))
        used_years.append(int(year))

    if not out:
        raise ValueError(f"{label}: 没有完整的Oct-Nov季节")
    result = xr.concat(out, dim="year").sortby("year")
    print(f"  {label}: 完整Oct-Nov {used_years[0]}–{used_years[-1]}，共{len(used_years)}年")
    return result

def sat_signflip(member_diff):
    work = member_diff.transpose("member", "lat", "lon")
    values = np.asarray(work.values, np.float32)
    n_member, n_lat, n_lon = values.shape
    flat = values.reshape(n_member, -1)

    valid = np.isfinite(flat)
    n_valid = valid.sum(axis=0)
    filled = np.where(valid, flat, 0.0).astype(np.float32)
    observed = filled.sum(axis=0) / np.where(n_valid > 0, n_valid, 1)
    good = n_valid >= SAT_SIG_MIN_MEMBERS

    count = np.zeros(flat.shape[1], dtype=np.int32)
    rng = np.random.default_rng(SAT_SIG_SEED)
    for start in range(0, SAT_SIG_N_RANDOM, SAT_SIG_BATCH):
        batch = min(SAT_SIG_BATCH, SAT_SIG_N_RANDOM - start)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(batch, n_member))
        random_mean = signs @ filled / np.where(n_valid > 0, n_valid, 1)[None, :]
        count += np.sum(
            (np.abs(random_mean) >= np.abs(observed)[None, :]) & good[None, :],
            axis=0,
        )

    p = (count + 1.0) / (SAT_SIG_N_RANDOM + 1.0)
    p[~good] = np.nan
    sig = (p <= SAT_SIG_ALPHA) & np.isfinite(p)
    return xr.DataArray(
        sig.reshape(n_lat, n_lon),
        coords={"lat": work.lat, "lon": work.lon},
        dims=("lat", "lon"),
    )

def draw_panel(ax_c, cax_c):
    print("\n========== 计算图(c)(c) Surface Air Temperature ==========")

    sat_rec_members, sat_had_members = [], []
    target_years = np.asarray(list(TARGET_YEARS), dtype=int)
    baseline_years = np.asarray(list(REF_YEARS), dtype=int)

    for member in range(SAT_EXPECTED_MEMBERS):
        rec_file = f"{SAT_REC_BASE}{member:02d}.nc"
        had_file = f"{SAT_HAD_BASE}{member:02d}.nc"
        if not os.path.exists(rec_file):
            raise FileNotFoundError(rec_file)
        if not os.path.exists(had_file):
            raise FileNotFoundError(had_file)

        pair = []
        for file, label in ((rec_file, "CAM-IAPICE"), (had_file, "CAM-HadISST")):
            ds = xr.open_dataset(file, decode_times=False, mask_and_scale=True)
            ds = add_year_month(ds, f"{label} member {member:02d}")
            sat = sat_surface_field(ds, f"{label} member {member:02d}")
            sat = sat.sel(lat=slice(SAT_MAP_LAT_MIN, SAT_MAP_LAT_MAX))
            annual = sat_octnov_series(sat, f"{label} member {member:02d} Oct-Nov").load()
            ds.close()
            pair.append(annual)

        rec_octnov, had_octnov = xr.align(pair[0], pair[1], join="inner")
        common_years = np.asarray(rec_octnov.year.values, dtype=int)
        if np.intersect1d(common_years, target_years).size != target_years.size:
            raise ValueError(f"SAT member {member:02d} 缺少完整1940–1942 Oct-Nov")
        if np.intersect1d(common_years, baseline_years).size != baseline_years.size:
            raise ValueError(f"SAT member {member:02d} 缺少完整1971–2000 Oct-Nov")

        sat_rec_members.append(rec_octnov.expand_dims(member=[member]))
        sat_had_members.append(had_octnov.expand_dims(member=[member]))

    sat_rec_members = xr.concat(sat_rec_members, dim="member", join="exact")
    sat_had_members = xr.concat(sat_had_members, dim="member", join="exact")
    sat_rec_members, sat_had_members = xr.align(sat_rec_members, sat_had_members, join="inner")

    rec_clim = sat_rec_members.sel(year=baseline_years).mean("year", skipna=True)
    had_clim = sat_had_members.sel(year=baseline_years).mean("year", skipna=True)
    rec_anom = sat_rec_members.sel(year=target_years).mean("year", skipna=True) - rec_clim
    had_anom = sat_had_members.sel(year=target_years).mean("year", skipna=True) - had_clim

    sat_member_diff = (rec_anom - had_anom).astype("float32")
    sat_diff = sat_member_diff.mean("member", skipna=True)
    sat_diff = sat_diff.assign_coords(lon=(sat_diff.lon + 180) % 360 - 180).sortby("lon")
    sat_member_diff = sat_member_diff.assign_coords(
        lon=(sat_member_diff.lon + 180) % 360 - 180
    ).sortby("lon")

    sat_sig = sat_signflip(sat_member_diff)

    values = np.asarray(sat_member_diff.transpose("member", "lat", "lon").values, np.float32)
    mean_field = np.nanmean(values, axis=0)
    mean_sign = np.sign(mean_field)
    valid = np.isfinite(values)
    valid_count = valid.sum(axis=0)
    same_sign = valid & (
        ((values > 0) & (mean_sign[None, :, :] > 0))
        | ((values < 0) & (mean_sign[None, :, :] < 0))
    )
    agree_count = same_sign.sum(axis=0)
    consistent = (
        (mean_sign != 0)
        & np.isfinite(mean_field)
        & (valid_count >= SAT_SIG_MIN_MEMBERS)
        & (agree_count >= SAT_CONSISTENCY_MIN_COUNT)
    )
    sat_consistent = xr.DataArray(
        consistent, coords={"lat": sat_member_diff.lat, "lon": sat_member_diff.lon},
        dims=("lat", "lon")
    )
    sat_sig_final = sat_sig & sat_consistent

    ax_c.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2 * np.pi, 240)
    circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)
    ax_c.set_boundary(circle, transform=ax_c.transAxes)
    if "geo" in ax_c.spines:
        ax_c.spines["geo"].set_visible(True)
        ax_c.spines["geo"].set_linewidth(MAP_FRAME_LINEWIDTH)
        ax_c.spines["geo"].set_edgecolor("black")
    try:
        ax_c.outline_patch.set_visible(True)
        ax_c.outline_patch.set_linewidth(MAP_FRAME_LINEWIDTH)
        ax_c.outline_patch.set_edgecolor("black")
    except Exception:
        pass
    ax_c.set_facecolor("white")
    ax_c.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=2)
    ax_c.coastlines(resolution="50m", color=COAST_COLOR, lw=COAST_LINEWIDTH, zorder=4)

    gl = ax_c.gridlines(
        crs=ccrs.PlateCarree(), linewidth=GRID_LINEWIDTH, color=GRID_COLOR,
        alpha=GRID_ALPHA, linestyle="-", draw_labels=True,
        x_inline=False, y_inline=False, rotate_labels=False, zorder=8,
    )
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator([])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": SMALL_SIZE, "weight": "normal"}
    gl.xpadding = 2
    gl.top_labels = gl.right_labels = gl.left_labels = False
    gl.xlines = False
    gl.ylines = False

    centers = (SAT_LEVELS[:-1] + SAT_LEVELS[1:]) / 2.0
    step = float(np.min(np.diff(SAT_LEVELS)))
    white = np.abs(centers) <= 0.5 * step + 1e-12
    neg = centers < -0.5 * step - 1e-12
    pos = centers > 0.5 * step + 1e-12
    colors = np.ones((len(centers), 4), dtype=float)
    neg_palette = np.array([plt.matplotlib.colors.to_rgba(c) for c in SAT_NEGATIVE_COLORS])
    pos_palette = np.array([plt.matplotlib.colors.to_rgba(c) for c in SAT_POSITIVE_COLORS])

    for mask, palette in ((neg, neg_palette), (pos, pos_palette)):
        n = int(mask.sum())
        if n > 0:
            old_x = np.linspace(0, 1, palette.shape[0])
            new_x = np.linspace(0, 1, n)
            colors[mask] = np.column_stack([
                np.interp(new_x, old_x, palette[:, i]) for i in range(4)
            ])
    colors[:, :3] = colors[:, :3] * (1.0 - SAT_CMAP_LIGHTEN) + SAT_CMAP_LIGHTEN
    colors[white, :3] = 1.0
    colors[:, 3] = 1.0
    cmap = ListedColormap(colors, name="sat_reference_RdBu")
    norm = BoundaryNorm(SAT_LEVELS, cmap.N, clip=True)

    data_cyc, lon_cyc = add_cyclic_point(
        sat_diff.values, coord=sat_diff.lon.values,
        axis=sat_diff.get_axis_num("lon")
    )
    lon2d, lat2d = np.meshgrid(lon_cyc, sat_diff.lat.values)
    ax_c.pcolormesh(
        lon2d, lat2d, data_cyc, cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), shading="auto", zorder=3,
    )
    ax_c.coastlines(resolution="50m", color=COAST_COLOR, lw=COAST_LINEWIDTH, zorder=4)

    sig_for_contour = sat_sig_final.astype(float)
    south_sig = np.asarray(sat_sig_final.isel(lat=0).values, dtype=bool)
    south_lon = np.asarray(sat_sig_final.lon.values, dtype=float)
    if float(sig_for_contour.lat.values[0]) > SAT_MAP_LAT_MIN + 1e-8:
        south_row = sig_for_contour.isel(lat=0).expand_dims(lat=[SAT_MAP_LAT_MIN])
        sig_for_contour = xr.concat([south_row, sig_for_contour], dim="lat")

    sig_cyc, sig_lon = add_cyclic_point(
        sig_for_contour.values, coord=sig_for_contour.lon.values,
        axis=sig_for_contour.get_axis_num("lon")
    )
    sig_lon2d, sig_lat2d = np.meshgrid(sig_lon, sig_for_contour.lat.values)
    if np.any(sig_cyc >= 0.5):
        contours = ax_c.contour(
            sig_lon2d, sig_lat2d, sig_cyc, levels=[0.5],
            colors=SAT_SIGNIFICANT_CONTOUR_COLOR,
            linewidths=SAT_SIGNIFICANT_CONTOUR_LINEWIDTH,
            linestyles="-", transform=ccrs.PlateCarree(), zorder=70,
            alpha=SAT_SIGNIFICANT_CONTOUR_OPACITY,
        )
        for collection in getattr(contours, "collections", [contours]):
            collection.set_path_effects([
                pe.Stroke(linewidth=SAT_SIGNIFICANT_CONTOUR_LINEWIDTH + 0.9, foreground="white"),
                pe.Normal(),
            ])

    if np.any(south_sig):
        order = np.argsort(south_lon)
        lon_sorted = south_lon[order]
        sig_sorted = south_sig[order]
        lon_unwrap = np.rad2deg(np.unwrap(np.deg2rad(lon_sorted)))
        if lon_unwrap.size > 1:
            mid = 0.5 * (lon_unwrap[:-1] + lon_unwrap[1:])
            first_edge = lon_unwrap[0] - 0.5 * (lon_unwrap[1] - lon_unwrap[0])
            last_edge = lon_unwrap[-1] + 0.5 * (lon_unwrap[-1] - lon_unwrap[-2])
            edges = np.r_[first_edge, mid, last_edge]
        else:
            edges = np.array([lon_unwrap[0] - 0.5, lon_unwrap[0] + 0.5])

        i = 0
        while i < sig_sorted.size:
            if not sig_sorted[i]:
                i += 1
                continue
            j = i
            while j + 1 < sig_sorted.size and sig_sorted[j + 1]:
                j += 1
            arc_lon = np.linspace(edges[i], edges[j + 1], 240)
            arc_lat = np.full_like(arc_lon, SAT_MAP_LAT_MIN)
            ax_c.plot(
                arc_lon, arc_lat, transform=ccrs.PlateCarree(),
                color=SAT_SIGNIFICANT_CONTOUR_COLOR,
                linewidth=SAT_SIGNIFICANT_CONTOUR_LINEWIDTH,
                linestyle="-", solid_capstyle="butt", zorder=95,
                alpha=SAT_SIGNIFICANT_CONTOUR_OPACITY,
                path_effects=[
                    pe.Stroke(linewidth=SAT_SIGNIFICANT_CONTOUR_LINEWIDTH + 0.9, foreground="white"),
                    pe.Normal(),
                ],
            )
            i = j + 1

    outer_lon = np.linspace(-180.0, 180.0, 1441)
    ax_c.plot(
        outer_lon, np.full_like(outer_lon, 60.0),
        transform=ccrs.PlateCarree(), color="black",
        linewidth=0.6, alpha=1.0, zorder=90,
    )

    ax_c.text(0.0, 1.02, "(c)", transform=ax_c.transAxes, fontsize=PANEL_SIZE,
              fontweight="bold", ha="left", va="bottom", clip_on=False, zorder=100)

    cbar_centers = (SAT_LEVELS[:-1] + SAT_LEVELS[1:]) / 2
    white = np.abs(cbar_centers) <= 0.5 * np.min(np.diff(SAT_LEVELS)) + 1e-12
    widths = np.ones(len(cbar_centers))
    widths[white] = 1.0 / max(int(white.sum()), 1)
    cpos = np.r_[0.0, np.cumsum(widths)]
    cbar_colors = np.asarray(cmap.colors[:len(cbar_centers)])
    ext = 0.55

    for i in range(len(cbar_centers)):
        cax_c.add_patch(Rectangle(
            (cpos[i], 0), cpos[i + 1] - cpos[i], 1,
            facecolor=cbar_colors[i], edgecolor="none"
        ))
    cax_c.add_patch(Polygon([(0, 0), (0, 1), (-ext, 0.5)],
                            facecolor=cbar_colors[0], edgecolor="none"))
    cax_c.add_patch(Polygon([(cpos[-1], 0), (cpos[-1], 1), (cpos[-1] + ext, 0.5)],
                            facecolor=cbar_colors[-1], edgecolor="none"))

    tick_values = np.asarray(SAT_CBAR_TICKS, dtype=float)
    divider_indices = [
        i for i in range(len(cpos))
        if i < len(SAT_LEVELS) and np.any(np.isclose(SAT_LEVELS[i], tick_values))
    ]
    for i in divider_indices:
        cax_c.plot([cpos[i], cpos[i]], [0, 1], color="0.25", lw=0.25)

    cax_c.plot([0, cpos[-1]], [0, 0], color="black", lw=0.5, zorder=20)
    cax_c.plot([0, cpos[-1]], [1, 1], color="black", lw=0.5, zorder=20)
    cax_c.plot([0, -ext, 0], [0, 0.5, 1], color="black", lw=0.5, zorder=20)
    cax_c.plot([cpos[-1], cpos[-1] + ext, cpos[-1]], [0, 0.5, 1],
               color="black", lw=0.5, zorder=20)

    tick_mask = [np.any(np.isclose(x, tick_values)) for x in SAT_LEVELS]
    ticks = [cpos[i] for i, keep in enumerate(tick_mask) if keep]
    labels = [f"{SAT_LEVELS[i]:g}".replace("-", "−") for i, keep in enumerate(tick_mask) if keep]
    cax_c.set_xlim(-ext, cpos[-1] + ext)
    cax_c.set_ylim(0, 1)
    cax_c.set_yticks([])
    cax_c.set_xticks(ticks)
    cax_c.set_xticklabels(labels, fontsize=SMALL_SIZE, fontweight="normal")
    cax_c.tick_params(axis="x", width=0.5, length=2.0, pad=1.5)
    cax_c.set_xlabel(SAT_CBAR_LABEL, fontsize=MEDIUM_SIZE, fontweight="normal", labelpad=3)
    cax_c.xaxis.labelpad = CBAR_LABELPAD
    for spine in cax_c.spines.values():
        spine.set_visible(False)

    map_pos = ax_c.get_position()
    cb_pos = cax_c.get_position()
    new_h = CBAR_THICKNESS_INCH / ax_c.figure.get_figheight()
    cax_c.set_position([
        map_pos.x0,
        cb_pos.y0 + (cb_pos.height - new_h) * CBAR_VERTICAL_FACTOR,
        map_pos.width,
        new_h,
    ])


if __name__ == "__main__":
    # 独立运行：仅绘制本文件对应的极区空间分布图。
    fig = plt.figure(figsize=(4.2, 4.0), dpi=FIG_DPI, facecolor="white")
    ax = fig.add_axes([0.08, 0.17, 0.84, 0.76], projection=ccrs.NorthPolarStereo())
    cax = fig.add_axes([0.08, 0.07, 0.84, 0.035])
    draw_panel(ax, cax)
    plt.show()
