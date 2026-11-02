
# -*- coding: utf-8 -*-
"""Fig2(a,b): IAPICE和HadISST海冰面积异常空间分布"""

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


# ============================================================
# 1. 所有可调参数
# ============================================================

PATH_SIC_RECON = r"E:\20century\data\sic_reconstructed_1901_2019_all_globe_data.nc"
PATH_SIC_HAD = r"E:\DATA\HadISST_ice.nc\HadISST_ice.nc"

SIC_RECON_VAR = "sic"
SIC_HAD_VAR = "sic"

TARGET_YEARS = range(1940, 1943)
REF_YEARS = range(1971, 2001)
MONTHS = (10, 11)

SIC_THRESHOLD = 0.15

MAP_LAT_MIN = 60
MAP_LAT_MAX = 90

# 研究区方框
TS_LAT_MIN = 60
TS_LON_MIN = 120
TS_LON_MAX = 240
REGION_BOX_LAT_TOP = 75.0

REGION_BOX_COLOR = "#D81B60"
REGION_BOX_LINEWIDTH = 3.0
REGION_BOX_OPACITY = 1.0
REGION_BOX_LINESTYLE = "-"

# ============================================================
# Colorbar设置：-2到2，间隔0.4
# ============================================================

LEVELS_SIA = np.round(np.arange(-2.0, 2.001, 0.4), 1)

SIA_CMAP = "BrBG"
SIA_CMAP_LIGHTEN = 0.12

SIA_CBAR_LABEL = r"Sea Ice Area ($10^3$ km$^2$)"
SIA_CBAR_TICKS = (-2, -1.2, -0.4, 0.4, 1.2, 2)

# 字号
SMALL_SIZE = 8
MEDIUM_SIZE = 9
PANEL_SIZE = 10

# 整体布局
FIG_DPI = 600
FIG_SIZE = (6.83, 3.35)

AX_A_POS = [0.045, 0.20, 0.425, 0.72]
AX_B_POS = [0.530, 0.20, 0.425, 0.72]

# 共用水平色标
CBAR_POS = [0.245, 0.095, 0.510, 0.035]

COAST_COLOR = "0.35"
COAST_LINEWIDTH = 0.45

GRID_COLOR = "0.72"
GRID_LINEWIDTH = 0.30
GRID_ALPHA = 0.70

MAP_FRAME_LINEWIDTH = 0.6

SAVE_FIG = None

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
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 2. 时间处理
# 优先使用time_bnds起点
# ============================================================

def add_year_month(
    ds, name,
    default_units=None,
    default_calendar="standard"
):

    if "time" not in ds:
        raise ValueError(f"{name}没有time坐标")

    bounds_name = ds["time"].attrs.get("bounds")

    if bounds_name not in ds:
        bounds_name = next(
            (x for x in ("time_bnds", "time_bounds", "time_bound")
             if x in ds),
            None
        )

    source = ds["time"]

    if bounds_name:
        bounds = ds[bounds_name]
        bound_dim = next(d for d in bounds.dims if d != "time")
        source = bounds.isel({bound_dim: 0})
        print(f"{name}: 使用{bounds_name}起点")
    else:
        print(f"{name}: 使用原time坐标")

    values = np.asarray(source.values)

    units = (
        source.attrs.get("units")
        or ds["time"].attrs.get("units")
        or default_units
    )

    calendar = (
        source.attrs.get("calendar")
        or ds["time"].attrs.get("calendar")
        or default_calendar
    )

    if np.issubdtype(values.dtype, np.datetime64):

        text = values.astype("datetime64[M]").astype(str)
        years = np.array([int(x[:4]) for x in text])
        months = np.array([int(x[5:7]) for x in text])

    elif (
        values.dtype == object
        and values.size
        and hasattr(values.ravel()[0], "year")
    ):

        years = np.array([int(x.year) for x in values.ravel()])
        months = np.array([int(x.month) for x in values.ravel()])

    else:

        if not units:
            raise ValueError(f"{name}时间缺少units")

        match = re.match(
            r"\s*months?\s+since\s+([+-]?\d+)-(\d+)-(\d+)",
            str(units), re.I
        )

        if match:

            y0 = int(match.group(1))
            m0 = int(match.group(2))

            offsets = np.rint(
                values.astype(float).ravel()
            ).astype(int)

            total = y0 * 12 + m0 - 1 + offsets
            years = total // 12
            months = total % 12 + 1

        else:

            dates = cftime.num2date(
                values.ravel(),
                units=str(units),
                calendar=str(calendar),
                only_use_cftime_datetimes=True
            )

            years = np.array([int(x.year) for x in dates])
            months = np.array([int(x.month) for x in dates])

    if len(years) != ds.sizes["time"]:
        raise ValueError(f"{name}时间长度与数据不一致")

    keys = years * 100 + months

    if (
        np.any(np.diff(keys) < 0)
        or len(np.unique(keys)) != len(keys)
    ):
        raise ValueError(f"{name}月份重复或非单调")

    print(
        f"{name}: "
        f"{years[0]:04d}-{months[0]:02d} 至 "
        f"{years[-1]:04d}-{months[-1]:02d}"
    )

    return ds.assign_coords(
        year=("time", years),
        month=("time", months)
    )


# ============================================================
# 3. 统一经纬度
# ============================================================

def standard_grid(da, lon180=False):

    rename = {}

    if "latitude" in da.dims:
        rename["latitude"] = "lat"

    if "longitude" in da.dims:
        rename["longitude"] = "lon"

    if rename:
        da = da.rename(rename)

    da = da.sortby("lat")

    da = da.assign_coords(
        lon=(da.lon + 360) % 360
    ).sortby("lon")

    _, keep = np.unique(
        np.round(da.lon.values, 8),
        return_index=True
    )

    if len(keep) != da.sizes["lon"]:
        da = da.isel(lon=np.sort(keep))

    if lon180:
        da = da.assign_coords(
            lon=(da.lon + 180) % 360 - 180
        ).sortby("lon")

    return da


# ============================================================
# 4. 指定年份10–11月平均
# ============================================================

def annual_mean(da, years):

    out = []

    for year in years:

        pieces = []

        for month in MONTHS:

            part = da.where(
                (da.year == year) & (da.month == month),
                drop=True
            )

            if part.sizes.get("time", 0) != 1:
                raise ValueError(
                    f"缺少或重复月份: {year}-{month:02d}"
                )

            pieces.append(
                part.isel(time=0, drop=True)
            )

        seasonal = xr.concat(
            pieces, dim="selected_month"
        ).mean("selected_month")

        out.append(
            seasonal.expand_dims(year=[year])
        )

    return xr.concat(out, dim="year")


# ============================================================
# 5. 读取数据并计算海冰面积异常
# ============================================================

print("\n读取IAPICE和HadISST海冰数据...")

rec_raw = xr.open_dataset(
    PATH_SIC_RECON,
    decode_times=False,
    mask_and_scale=True
)

had_raw = xr.open_dataset(
    PATH_SIC_HAD,
    decode_times=False,
    mask_and_scale=True
)

rec_ds = add_year_month(
    rec_raw, "IAPICE",
    "days since 1900-01-01", "360_day"
)

had_ds = add_year_month(
    had_raw, "HadISST"
)

rec_sic = standard_grid(
    rec_ds[SIC_RECON_VAR]
)

had_sic = standard_grid(
    had_ds[SIC_HAD_VAR]
)

# SIC统一为0–1
if float(rec_sic.max(skipna=True)) > 1.5:
    rec_sic = rec_sic / 100.0

if float(had_sic.max(skipna=True)) > 1.5:
    had_sic = had_sic / 100.0

# 北极区域
rec_arctic = rec_sic.sel(
    lat=slice(MAP_LAT_MIN, MAP_LAT_MAX)
)

had_arctic = had_sic.sel(
    lat=slice(MAP_LAT_MIN, MAP_LAT_MAX)
)

# HadISST插值到IAPICE网格
had_arctic = had_arctic.interp(
    lat=rec_arctic.lat,
    lon=rec_arctic.lon,
    method="nearest"
)

# 共用海洋掩膜
ocean_arctic = had_arctic.notnull().any("time")

# ---------- 计算格点面积 ----------
lat = np.asarray(rec_arctic.lat, float)
lon = np.asarray(rec_arctic.lon, float)

dlat = abs(np.diff(lat).mean())
dlon = abs(np.diff(lon).mean())

lat_b = np.r_[
    lat[0] - dlat / 2,
    (lat[:-1] + lat[1:]) / 2,
    lat[-1] + dlat / 2
]

lat_b = np.clip(lat_b, -90, 90)

band = np.abs(
    np.diff(np.sin(np.deg2rad(lat_b)))
)

area = (
    6371000.0 ** 2
    * np.deg2rad(dlon)
    * band[:, None]
    * np.ones((len(lat), len(lon)))
)

area = xr.DataArray(
    area,
    coords={"lat": lat, "lon": lon},
    dims=("lat", "lon")
)

# ---------- 海冰面积贡献 ----------
rec_contribution = xr.where(
    rec_arctic >= SIC_THRESHOLD,
    rec_arctic * area,
    0.0
).where(
    ocean_arctic & rec_arctic.notnull()
)

had_contribution = xr.where(
    had_arctic >= SIC_THRESHOLD,
    had_arctic * area,
    0.0
).where(
    ocean_arctic & had_arctic.notnull()
)

# ---------- IAPICE异常 ----------
rec_target = annual_mean(
    rec_contribution, TARGET_YEARS
).mean("year")

rec_ref = annual_mean(
    rec_contribution, REF_YEARS
).mean("year")

sia_rec = (rec_target - rec_ref) / 1e9

# ---------- HadISST异常 ----------
had_target = annual_mean(
    had_contribution, TARGET_YEARS
).mean("year")

had_ref = annual_mean(
    had_contribution, REF_YEARS
).mean("year")

sia_had = (had_target - had_ref) / 1e9

# 经度转换
sia_rec = standard_grid(sia_rec, lon180=True)
sia_had = standard_grid(sia_had, lon180=True)

print(
    "\nIAPICE异常范围:",
    float(sia_rec.min(skipna=True)),
    float(sia_rec.max(skipna=True))
)

print(
    "HadISST异常范围:",
    float(sia_had.min(skipna=True)),
    float(sia_had.max(skipna=True))
)


# ============================================================
# 6. 配色：BrBG，中心白色
# ============================================================

centers = (LEVELS_SIA[:-1] + LEVELS_SIA[1:]) / 2.0
step = float(np.min(np.diff(LEVELS_SIA)))

white = np.abs(centers) <= step + 1e-12

x = np.linspace(-1.0, 1.0, centers.size)

sample = (
    0.5 * (
        np.sign(x) * np.abs(x) ** 0.88 + 1.0
    )
)

sample = 0.07 + sample * 0.86

colors = plt.get_cmap(SIA_CMAP)(
    np.clip(sample, 0.07, 0.93)
)

colors[:, :3] = (
    colors[:, :3] * (1.0 - SIA_CMAP_LIGHTEN)
    + SIA_CMAP_LIGHTEN
)

colors[white, :3] = 1.0
colors[:, 3] = 1.0

cmap = ListedColormap(colors)

norm = BoundaryNorm(
    LEVELS_SIA, cmap.N, clip=True
)


# ============================================================
# 7. 绘制极区地图
# ============================================================

def draw_panel(ax, sia, label, title):

    ax.set_extent(
        [-180, 180, 60, 90],
        crs=ccrs.PlateCarree()
    )

    theta = np.linspace(0, 2 * np.pi, 240)

    circle = mpath.Path(
        np.vstack([
            np.sin(theta),
            np.cos(theta)
        ]).T * 0.5 + 0.5
    )

    ax.set_boundary(
        circle, transform=ax.transAxes
    )

    if "geo" in ax.spines:
        ax.spines["geo"].set_visible(True)
        ax.spines["geo"].set_linewidth(
            MAP_FRAME_LINEWIDTH
        )
        ax.spines["geo"].set_edgecolor("black")

    ax.set_facecolor("white")

    ax.add_feature(
        cfeature.LAND,
        facecolor="white",
        edgecolor="none",
        zorder=2
    )

    # 海冰异常填色
    data_cyc, lon_cyc = add_cyclic_point(
        sia.values,
        coord=sia.lon.values,
        axis=sia.get_axis_num("lon")
    )

    lon2d, lat2d = np.meshgrid(
        lon_cyc, sia.lat.values
    )

    ax.pcolormesh(
        lon2d, lat2d, data_cyc,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
        zorder=3
    )

    # 海岸线
    ax.coastlines(
        resolution="50m",
        color=COAST_COLOR,
        lw=COAST_LINEWIDTH,
        zorder=4
    )

    # 经纬网
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=GRID_LINEWIDTH,
        color=GRID_COLOR,
        alpha=GRID_ALPHA,
        linestyle="-",
        draw_labels=True,
        x_inline=False,
        y_inline=False,
        rotate_labels=False,
        zorder=8
    )

    gl.xlocator = mticker.FixedLocator(
        np.arange(-180, 181, 60)
    )

    gl.ylocator = mticker.FixedLocator([])
    gl.xformatter = LONGITUDE_FORMATTER

    gl.xlabel_style = {
        "size": SMALL_SIZE,
        "weight": "normal"
    }

    gl.xpadding = 2
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False

    # ---------- 研究区方框 ----------
    box_lon = np.linspace(
        TS_LON_MIN, TS_LON_MAX, 181
    )

    box_lat = np.linspace(
        TS_LAT_MIN, REGION_BOX_LAT_TOP, 121
    )

    box_kw = dict(
        transform=ccrs.PlateCarree(),
        color=REGION_BOX_COLOR,
        linewidth=REGION_BOX_LINEWIDTH,
        linestyle=REGION_BOX_LINESTYLE,
        zorder=100,
        alpha=REGION_BOX_OPACITY,
        path_effects=[
            pe.Stroke(
                linewidth=REGION_BOX_LINEWIDTH + 0.9,
                foreground="white"
            ),
            pe.Normal()
        ]
    )

    ax.plot(
        box_lon,
        np.full_like(box_lon, TS_LAT_MIN),
        **box_kw
    )

    ax.plot(
        box_lon,
        np.full_like(box_lon, REGION_BOX_LAT_TOP),
        **box_kw
    )

    ax.plot(
        np.full_like(box_lat, TS_LON_MIN),
        box_lat,
        **box_kw
    )

    ax.plot(
        np.full_like(box_lat, TS_LON_MAX),
        box_lat,
        **box_kw
    )

    # 子图编号
    ax.text(
        0.0, 1.02, label,
        transform=ax.transAxes,
        fontsize=PANEL_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
        zorder=200
    )

    # 数据名称
    ax.text(
        0.5, 1.02, title,
        transform=ax.transAxes,
        fontsize=MEDIUM_SIZE,
        fontweight="normal",
        ha="center",
        va="bottom",
        clip_on=False,
        zorder=200
    )


# ============================================================
# 8. 共用水平Colorbar
# -2至2，间隔0.4，指定刻度
# ============================================================

def draw_colorbar(cax):

    cbar_centers = (
        LEVELS_SIA[:-1] + LEVELS_SIA[1:]
    ) / 2

    # 中心白色区域压缩
    white = (
        np.abs(cbar_centers)
        <= np.min(np.diff(LEVELS_SIA)) + 1e-12
    )

    widths = np.ones(len(cbar_centers))

    widths[white] = (
        1.0 / max(int(white.sum()), 1)
    )

    pos = np.r_[0.0, np.cumsum(widths)]

    cbar_colors = np.asarray(
        cmap.colors[:len(cbar_centers)]
    )

    ext = 0.55

    # 色块
    for i in range(len(cbar_centers)):

        cax.add_patch(
            Rectangle(
                (pos[i], 0),
                pos[i + 1] - pos[i],
                1,
                facecolor=cbar_colors[i],
                edgecolor="none"
            )
        )

    # 左侧三角形
    cax.add_patch(
        Polygon(
            [(0, 0), (0, 1), (-ext, 0.5)],
            facecolor=cbar_colors[0],
            edgecolor="none"
        )
    )

    # 右侧三角形
    cax.add_patch(
        Polygon(
            [
                (pos[-1], 0),
                (pos[-1], 1),
                (pos[-1] + ext, 0.5)
            ],
            facecolor=cbar_colors[-1],
            edgecolor="none"
        )
    )

    # 分隔线
    for i, value in enumerate(LEVELS_SIA):

        if (
            i < len(pos)
            and not np.isclose(value, 0.0)
        ):

            cax.plot(
                [pos[i], pos[i]],
                [0, 1],
                color="0.25",
                lw=0.25
            )

    # 色标边框
    cax.plot(
        [0, pos[-1]], [0, 0],
        color="black", lw=0.5
    )

    cax.plot(
        [0, pos[-1]], [1, 1],
        color="black", lw=0.5
    )

    cax.plot(
        [0, -ext, 0], [0, 0.5, 1],
        color="black", lw=0.5
    )

    cax.plot(
        [pos[-1], pos[-1] + ext, pos[-1]],
        [0, 0.5, 1],
        color="black", lw=0.5
    )

    # ---------- 指定刻度 ----------
    # 使用np.isclose避免浮点精度问题
    tick_indices = []

    for value in SIA_CBAR_TICKS:

        idx = np.where(
            np.isclose(LEVELS_SIA, value)
        )[0]

        if len(idx) != 1:
            raise ValueError(
                f"色标刻度{value}不在LEVELS_SIA中"
            )

        tick_indices.append(int(idx[0]))

    ticks = [pos[i] for i in tick_indices]

    labels = [
        f"{value:g}".replace("-", "−")
        for value in SIA_CBAR_TICKS
    ]

    cax.set_xlim(-ext, pos[-1] + ext)
    cax.set_ylim(0, 1)

    cax.set_yticks([])
    cax.set_xticks(ticks)

    cax.set_xticklabels(
        labels,
        fontsize=SMALL_SIZE,
        fontweight="normal"
    )

    cax.tick_params(
        axis="x",
        width=0.5,
        length=2.0,
        pad=1.5
    )

    cax.set_xlabel(
        SIA_CBAR_LABEL,
        fontsize=MEDIUM_SIZE,
        fontweight="normal",
        labelpad=3
    )

    for spine in cax.spines.values():
        spine.set_visible(False)


# ============================================================
# 9. 绘制两张图
# ============================================================

fig = plt.figure(
    figsize=FIG_SIZE,
    dpi=FIG_DPI,
    facecolor="white"
)

ax_a = fig.add_axes(
    AX_A_POS,
    projection=ccrs.NorthPolarStereo()
)

ax_b = fig.add_axes(
    AX_B_POS,
    projection=ccrs.NorthPolarStereo()
)

cax = fig.add_axes(CBAR_POS)

draw_panel(
    ax_a,
    sia_rec,
    "(a)",
    "IAPICE"
)

draw_panel(
    ax_b,
    sia_had,
    "(b)",
    "HadISST"
)

draw_colorbar(cax)

if SAVE_FIG:
    fig.savefig(
        SAVE_FIG,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

plt.show()

rec_ds.close()
had_ds.close()
rec_raw.close()
had_raw.close()
