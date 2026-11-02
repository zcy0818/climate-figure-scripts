# -*- coding: utf-8 -*-
"""Figure 2d: Oct-Nov turbulent-heat-flux response."""

import os
import re

import cftime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER
from scipy import ndimage

DATA_DIR = r"E:\DATA"
HAD_PREFIX = "HadISST_1deg"
REC_PREFIX = "reconstructed_1deg"
LHFLX_VAR = "LHFLX"
SHFLX_VAR = "SHFLX"
N_MEMBER = 20

TARGET_YEARS = np.arange(1940, 1943)
REF_YEARS = np.arange(1971, 2001)
MONTHS = (10, 11)
LAT_MIN, LAT_MAX = 60.0, 90.0

DISPLAY_SCALE = 100.0
LEVELS = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0])
WHITE_THRESHOLD = 0.2
CBAR_TICKS = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
CBAR_LABEL = r"Turbulent Heat Flux ($10^{2}$ W m$^{-2}$)"

N_RANDOM = 10000
ALPHA = 0.05
SEED = 20260814
BATCH = 250

SIGNIFICANT_CONTOUR_COLOR = "#4B1D6B"
SIGNIFICANT_CONTOUR_LINEWIDTH = 1.4
SIGNIFICANT_CONTOUR_OPACITY = 1.0
SIGNIFICANT_CONTOUR_MIN_RELATIVE_AREA = 0.12
SIGNIFICANT_CONTOUR_MIN_POLAR_AREA = 2.0

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

COAST_COLOR = "0.30"
COAST_LINEWIDTH = 0.45
GRID_COLOR = "0.72"
GRID_LINEWIDTH = 0.30
GRID_ALPHA = 0.75
MAP_FRAME_LINEWIDTH = 0.6

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "bold",
    "axes.unicode_minus": True,
    "font.size": MEDIUM_SIZE,
    "xtick.labelsize": SMALL_SIZE,
    "ytick.labelsize": SMALL_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def add_year_month(ds, label):
    bounds_name = ds["time"].attrs.get("bounds")
    if bounds_name not in ds:
        bounds_name = next((x for x in ("time_bnds", "time_bounds", "time_bound") if x in ds), None)

    if bounds_name is not None:
        bounds = ds[bounds_name]
        bdim = next(d for d in bounds.dims if d != "time")
        source = bounds.isel({bdim: 0})
        print(f"{label}: 月份使用 {bounds_name} 起点")
    else:
        source = ds["time"]
        print(f"{label}: 没有time_bnds，使用time")

    values = np.asarray(source.values)
    units = source.attrs.get("units") or ds["time"].attrs.get("units")
    calendar = source.attrs.get("calendar") or ds["time"].attrs.get("calendar") or "standard"

    if np.issubdtype(values.dtype, np.datetime64):
        text = values.astype("datetime64[M]").astype(str)
        years = np.array([int(x[:4]) for x in text])
        months = np.array([int(x[5:7]) for x in text])
    elif values.dtype == object and values.size > 0 and hasattr(values.ravel()[0], "year"):
        years = np.array([int(x.year) for x in values.ravel()])
        months = np.array([int(x.month) for x in values.ravel()])
    else:
        if units is None:
            raise ValueError(f"{label}: 时间没有units")
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
        raise ValueError(f"{label}: 时间长度与数据长度不一致")

    print(f"    {years[0]:04d}-{months[0]:02d} → {years[-1]:04d}-{months[-1]:02d}")
    return ds.assign_coords(year=("time", years), month=("time", months))

def octnov_mean(da, years, label):
    result = []
    for year in years:
        one = da.where((da.year == year) & da.month.isin(MONTHS), drop=True)
        if one.sizes.get("time", 0) != 2:
            raise ValueError(
                f"{label}: {year}年没有完整的{MONTHS[0]}、{MONTHS[1]}月，"
                f"实际找到 {one.sizes.get('time', 0)} 个月"
            )
        result.append(one.mean("time", skipna=True).expand_dims(year=[year]))
    return xr.concat(result, dim="year")

def signflip_test(member_diff):
    work = member_diff.transpose("member", "lat", "lon")
    values = np.asarray(work.values, dtype=np.float32)
    n_member, nlat, nlon = values.shape
    flat = values.reshape(n_member, -1)

    valid = np.isfinite(flat)
    n_valid = valid.sum(axis=0)
    filled = np.where(valid, flat, 0.0).astype(np.float32)
    observed = filled.sum(axis=0) / np.maximum(n_valid, 1)
    good = n_valid == N_MEMBER

    count = np.zeros(flat.shape[1], dtype=np.int32)
    rng = np.random.default_rng(SEED)
    for start in range(0, N_RANDOM, BATCH):
        n = min(BATCH, N_RANDOM - start)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, n_member))
        random_mean = signs @ filled / np.maximum(n_valid, 1)[None, :]
        count += np.sum(
            (np.abs(random_mean) >= np.abs(observed)[None, :]) & good[None, :],
            axis=0,
        )

    p = (count + 1.0) / (N_RANDOM + 1.0)
    p[~good] = np.nan
    sig = (p <= ALPHA) & np.isfinite(p)
    return xr.DataArray(
        sig.reshape(nlat, nlon),
        coords={"lat": work.lat, "lon": work.lon},
        dims=("lat", "lon"),
    )

def keep_largest_significant_region(mask):
    """Keep the largest 8-connected region, treating the longitude seam as periodic."""
    arr = np.asarray(mask.values, dtype=bool)
    if not np.any(arr):
        return xr.DataArray(np.zeros_like(arr, dtype=bool), coords=mask.coords, dims=mask.dims)

    labeled, n_labels = ndimage.label(arr, structure=np.ones((3, 3), dtype=int))
    parent = np.arange(n_labels + 1, dtype=int)

    for i in range(arr.shape[0]):
        a, b = int(labeled[i, 0]), int(labeled[i, -1])
        if a <= 0 or b <= 0:
            continue
        ra, rb = a, b
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]
        while parent[rb] != rb:
            parent[rb] = parent[parent[rb]]
            rb = parent[rb]
        if ra != rb:
            parent[rb] = ra

    for label in range(1, n_labels + 1):
        root = label
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        parent[label] = root

    rows, cols = np.where(labeled > 0)
    root_sizes = {}
    for r, c in zip(rows, cols):
        root = parent[int(labeled[r, c])]
        root_sizes[root] = root_sizes.get(root, 0) + 1

    keep = np.zeros_like(arr, dtype=bool)
    if root_sizes:
        largest_root = max(root_sizes, key=root_sizes.get)
        for r, c in zip(rows, cols):
            if parent[int(labeled[r, c])] == largest_root:
                keep[r, c] = True

    return xr.DataArray(keep, coords=mask.coords, dims=mask.dims)

def draw_panel(ax_d, cax_d):
    member_diff_list = []
    all_years = np.concatenate([TARGET_YEARS, REF_YEARS])

    for member in range(N_MEMBER):
        print("\n" + "=" * 60)
        print(f"处理THF成员 {member + 1:02d}/{N_MEMBER}")

        if member == 0:
            had_name = f"{HAD_PREFIX}.cam.h0.extract07-fulldims.nc"
            rec_name = f"{REC_PREFIX}.cam.h0.extract07-fulldims.nc"
        else:
            had_name = f"{HAD_PREFIX}_{member}.cam.h0.extract07-fulldims.nc"
            rec_name = f"{REC_PREFIX}_{member}.cam.h0.extract07-fulldims.nc"
        had_file = os.path.join(DATA_DIR, had_name)
        rec_file = os.path.join(DATA_DIR, rec_name)

        if not os.path.exists(had_file):
            raise FileNotFoundError(had_file)
        if not os.path.exists(rec_file):
            raise FileNotFoundError(rec_file)

        pair_anom = []
        for path, label in ((rec_file, "reconstructed"), (had_file, "HadISST")):
            ds = xr.open_dataset(path, decode_times=False, mask_and_scale=True)
            ds = add_year_month(ds, f"{label} member {member}")
            thf = ds[LHFLX_VAR] + ds[SHFLX_VAR]

            rename = {}
            if "latitude" in thf.dims:
                rename["latitude"] = "lat"
            if "longitude" in thf.dims:
                rename["longitude"] = "lon"
            if rename:
                thf = thf.rename(rename)

            thf = thf.sortby("lat")
            thf = thf.assign_coords(lon=np.mod(thf.lon, 360.0)).sortby("lon")
            thf = thf.sel(lat=slice(LAT_MIN, LAT_MAX))
            thf = thf.where(ds.year.isin(all_years) & ds.month.isin(MONTHS), drop=True).load()

            target = octnov_mean(thf, TARGET_YEARS, label).mean("year", skipna=True)
            climatology = octnov_mean(thf, REF_YEARS, label).mean("year", skipna=True)
            pair_anom.append(target - climatology)
            ds.close()

        rec_anom, had_anom = xr.align(pair_anom[0], pair_anom[1], join="exact")
        member_diff_list.append(
            (rec_anom - had_anom).astype("float32").expand_dims(member=[member])
        )

    member_diff = xr.concat(member_diff_list, dim="member", join="exact")
    thf_diff = member_diff.mean("member", skipna=True)
    thf_diff = thf_diff.assign_coords(lon=(thf_diff.lon + 180) % 360 - 180).sortby("lon")
    member_diff = member_diff.assign_coords(lon=(member_diff.lon + 180) % 360 - 180).sortby("lon")

    print("\n进行20成员配对 sign-flip Monte Carlo...")
    sig = signflip_test(member_diff)
    print(f"THF异常差范围：{float(thf_diff.min(skipna=True)):.2f} ~ {float(thf_diff.max(skipna=True)):.2f} W m-2")
    print(f"95%显著格点：{int(sig.sum())}/{sig.size}")

    thf_plot = thf_diff / DISPLAY_SCALE
    sig_main = keep_largest_significant_region(sig)
    print(f"95%显著格点：{int(sig.sum())}/{sig.size}")
    print(f"保留的主要连续显著区格点：{int(sig_main.sum())}/{sig_main.size}")

    ax_d.set_extent([-180, 180, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2 * np.pi, 240)
    circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)
    ax_d.set_boundary(circle, transform=ax_d.transAxes)
    if "geo" in ax_d.spines:
        ax_d.spines["geo"].set_visible(True)
        ax_d.spines["geo"].set_linewidth(MAP_FRAME_LINEWIDTH)
        ax_d.spines["geo"].set_edgecolor("black")
    ax_d.set_facecolor("white")

    NEGATIVE_COLORS = ("#15508D", "#3885BC", "#81BAD8", "#CAE1EE")
    POSITIVE_COLORS = ("#FCD5BF", "#EE9677", "#CB4942", "#930E26")
    CMAP_LIGHTEN = 0.10

    centers = (LEVELS[:-1] + LEVELS[1:]) / 2.0
    step = float(np.min(np.diff(LEVELS)))
    white = np.abs(centers) <= 0.5 * step + 1e-12
    neg = centers < -0.5 * step - 1e-12
    pos = centers > 0.5 * step + 1e-12

    colors = np.ones((len(centers), 4), dtype=float)
    neg_palette = np.array([plt.matplotlib.colors.to_rgba(c) for c in NEGATIVE_COLORS])
    pos_palette = np.array([plt.matplotlib.colors.to_rgba(c) for c in POSITIVE_COLORS])

    for mask, palette in ((neg, neg_palette), (pos, pos_palette)):
        n = int(mask.sum())
        if n > 0:
            old_x = np.linspace(0, 1, palette.shape[0])
            new_x = np.linspace(0, 1, n)
            colors[mask] = np.column_stack([
                np.interp(new_x, old_x, palette[:, i]) for i in range(4)
            ])

    colors[:, :3] = colors[:, :3] * (1.0 - CMAP_LIGHTEN) + CMAP_LIGHTEN
    colors[white, :3] = 1.0
    colors[:, 3] = 1.0

    cmap = ListedColormap(colors, name="shared_blue_white_red")
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    norm = BoundaryNorm(LEVELS, cmap.N)

    data_cyc, lon_cyc = add_cyclic_point(
        thf_plot.values, coord=thf_plot.lon.values,
        axis=thf_plot.get_axis_num("lon")
    )
    lon2d, lat2d = np.meshgrid(lon_cyc, thf_plot.lat.values)
    mesh = ax_d.pcolormesh(
        lon2d, lat2d, data_cyc, cmap=cmap, norm=norm,
        shading="auto", transform=ccrs.PlateCarree(), zorder=2,
    )
    ax_d.coastlines(resolution="50m", color=COAST_COLOR, linewidth=COAST_LINEWIDTH, zorder=10)

    sig_for_contour = sig_main.astype(float)
    south_sig = np.asarray(sig_main.isel(lat=0).values, dtype=bool)
    south_lon = np.asarray(sig_main.lon.values, dtype=float)

    if float(sig_for_contour.lat.values[0]) > LAT_MIN + 1e-8:
        south_row = sig_for_contour.isel(lat=0).expand_dims(lat=[LAT_MIN])
        sig_for_contour = xr.concat([south_row, sig_for_contour], dim="lat")

    sig_cyc, sig_lon_cyc = add_cyclic_point(
        sig_for_contour.values, coord=sig_for_contour.lon.values,
        axis=sig_for_contour.get_axis_num("lon")
    )
    sig_lon2d, sig_lat2d = np.meshgrid(sig_lon_cyc, sig_for_contour.lat.values)

    if np.any(sig_cyc >= 0.5):
        if sig_for_contour.sizes.get("lat", 0) >= 2 and float(sig_for_contour.lat.values[-1]) >= 89.5:
            near_pole = np.asarray(sig_for_contour.isel(lat=-2).values, dtype=float)
            sig_for_contour[{"lat": -1}] = 1.0 if np.nanmean(near_pole) >= 0.5 else 0.0
            sig_cyc, sig_lon_cyc = add_cyclic_point(
                sig_for_contour.values, coord=sig_for_contour.lon.values,
                axis=sig_for_contour.get_axis_num("lon")
            )
            sig_lon2d, sig_lat2d = np.meshgrid(sig_lon_cyc, sig_for_contour.lat.values)

        contour_tmp = ax_d.contour(
            sig_lon2d, sig_lat2d, sig_cyc, levels=[0.5],
            colors="none", linewidths=0.0,
            transform=ccrs.PlateCarree(), zorder=34,
        )
        contour_segments = contour_tmp.allsegs[0] if getattr(contour_tmp, "allsegs", None) else []
        try:
            for collection in contour_tmp.collections:
                collection.remove()
        except Exception:
            pass

        if contour_segments:
            areas = []
            for segment in contour_segments:
                segment = np.asarray(segment, dtype=float)
                if segment.ndim != 2 or segment.shape[0] < 4:
                    areas.append(0.0)
                    continue
                theta_seg = np.deg2rad(segment[:, 0])
                radius_seg = 90.0 - segment[:, 1]
                xx = radius_seg * np.sin(theta_seg)
                yy = radius_seg * np.cos(theta_seg)
                area = 0.5 * abs(
                    np.sum(xx[:-1] * yy[1:] - xx[1:] * yy[:-1])
                    + xx[-1] * yy[0] - xx[0] * yy[-1]
                )
                areas.append(area)

            areas = np.asarray(areas, dtype=float)
            max_area = float(np.nanmax(areas)) if areas.size else 0.0
            area_threshold = max(
                SIGNIFICANT_CONTOUR_MIN_POLAR_AREA,
                SIGNIFICANT_CONTOUR_MIN_RELATIVE_AREA * max_area,
            )

            kept_count = 0
            for segment, area in zip(contour_segments, areas):
                if area < area_threshold:
                    continue
                segment = np.asarray(segment, dtype=float)
                ax_d.plot(
                    segment[:, 0], segment[:, 1], transform=ccrs.PlateCarree(),
                    color=SIGNIFICANT_CONTOUR_COLOR,
                    linewidth=SIGNIFICANT_CONTOUR_LINEWIDTH,
                    linestyle="-", solid_capstyle="round", solid_joinstyle="round",
                    zorder=35, alpha=SIGNIFICANT_CONTOUR_OPACITY,
                    path_effects=[
                        pe.Stroke(linewidth=SIGNIFICANT_CONTOUR_LINEWIDTH + 0.9, foreground="white"),
                        pe.Normal(),
                    ],
                )
                kept_count += 1
            print(
                f"显著性轮廓过滤：原{len(contour_segments)}条，保留{kept_count}条；"
                f"面积阈值={area_threshold:.2f}"
            )

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
            arc_lat = np.full_like(arc_lon, LAT_MIN)
            ax_d.plot(
                arc_lon, arc_lat, transform=ccrs.PlateCarree(),
                color=SIGNIFICANT_CONTOUR_COLOR,
                linewidth=SIGNIFICANT_CONTOUR_LINEWIDTH,
                linestyle="-", solid_capstyle="butt", zorder=95,
                alpha=SIGNIFICANT_CONTOUR_OPACITY,
                path_effects=[
                    pe.Stroke(linewidth=SIGNIFICANT_CONTOUR_LINEWIDTH + 0.9, foreground="white"),
                    pe.Normal(),
                ],
            )
            i = j + 1

    gl = ax_d.gridlines(
        crs=ccrs.PlateCarree(), linewidth=GRID_LINEWIDTH, color=GRID_COLOR,
        alpha=GRID_ALPHA, linestyle="-", draw_labels=True,
        x_inline=False, y_inline=False, rotate_labels=False, zorder=25,
    )
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator([])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": SMALL_SIZE, "weight": "normal"}
    gl.xpadding = 2
    gl.left_labels = gl.right_labels = gl.top_labels = False
    gl.xlines = False
    gl.ylines = False

    outer_lon = np.linspace(-180, 180, 1441)
    ax_d.plot(
        outer_lon, np.full_like(outer_lon, LAT_MIN),
        transform=ccrs.PlateCarree(), color="black", linewidth=0.6, zorder=40,
    )
    ax_d.text(0.0, 1.02, "(d)", transform=ax_d.transAxes, fontsize=PANEL_SIZE,
              fontweight="bold", ha="left", va="bottom", clip_on=False, zorder=100)

    cb = ax_d.figure.colorbar(
        mesh, cax=cax_d, orientation="horizontal", extend="both",
        boundaries=LEVELS, spacing="uniform",
    )
    cb.set_ticks(CBAR_TICKS)

    tick_labels = []
    for x in CBAR_TICKS:
        if np.isclose(x, round(x)):
            tick_labels.append(f"{int(round(x))}".replace("-", "−"))
        else:
            tick_labels.append(f"{x:.1f}".replace("-", "−"))
    cb.ax.set_xticklabels(tick_labels)
    cb.ax.xaxis.set_minor_locator(mticker.NullLocator())
    cb.minorticks_off()
    cb.ax.tick_params(
        axis="x", which="major", labelsize=SMALL_SIZE,
        width=0.45, length=2.0, pad=1.5, bottom=True, top=False,
    )
    cb.set_label(CBAR_LABEL, fontsize=MEDIUM_SIZE, fontweight="normal", labelpad=3)
    cb.outline.set_linewidth(0.5)

    map_pos = ax_d.get_position()
    cb_pos = cax_d.get_position()
    new_h = CBAR_THICKNESS_INCH / ax_d.figure.get_figheight()
    cax_d.set_position([
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
