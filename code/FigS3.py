
# -*- coding: utf-8 -*-
"""Four panels: PDO, ENSO, raw climate anomalies, and anomalies
after simultaneously removing PDO and ENSO.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.text import Text

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


# ============================================================
# 1. 文件路径
# ============================================================

PATH_CRU = r"E:\20century\data\cru_ts4.05.1901.2020.tmp.dat.nc"
PATH_SLP = r"E:\20century\data\hadslp2_slp.nc"
PATH_STA = r"E:\all station\GHCNM 80stations.nc"
PATH_PDO = r"E:\20century\data\pdo.timeseries.sstens.nc"
PATH_ENSO = r"E:\20century\data\nino34.long.anom.nc"


# ============================================================
# 2. 时间参数
# ============================================================

EVENT_START, EVENT_END = 1940, 1942
REF_START, REF_END = 1971, 2000
INDEX_START, INDEX_END = 1901, 2000

MIN_REG_YEARS = 30


# ============================================================
# 3. 整体布局参数
# ============================================================

DPI = 600
FIGSIZE = (10.5, 12.2)

# 第一行：a、b缩短宽度、增加高度
# 图a、b整体放大，同时保持左右对称
AX_A_POS = [0.09, 0.76, 0.36, 0.19]
AX_B_POS = [0.55, 0.76, 0.36, 0.19]

# 第二行：c放大
AX_C_POS = [0.075, 0.415, 0.785, 0.29]

# 第三行：d放大，缩小与c之间的间距
AX_D_POS = [0.075, 0.065, 0.785, 0.29]

# c、d共用色标
CAX_POS = [0.89, 0.115, 0.022, 0.54]


# ============================================================
# 4. 图a、b指数柱状图参数
# ============================================================

INDEX_XLIM = (1900, 2000)
INDEX_XTICKS = np.arange(1900, 2001, 20)

INDEX_YLIM = (-3.0, 3.0)
INDEX_YTICKS = np.arange(-3.0, 3.1, 1.0)

INDEX_POS_COLOR = "#B2182B"
INDEX_NEG_COLOR = "#2166AC"

INDEX_BAR_WIDTH = 0.86

ZERO_COLOR = "#A8A8A8"
ZERO_LW = 0.65

SHADE_YEARS = (1940, 1942)
SHADE_COLOR = "#D3D3D3"
SHADE_ALPHA = 0.65


# ============================================================
# 5. 图c、d地图参数
# ============================================================

MAP_EXTENT = [-15, 180, 20, 85]

DATA_LON_RANGE = (-20, 180)
DATA_LAT_RANGE = (15, 90)

BOX = [30, 60, 45, 60]

DATA_CRS = ccrs.PlateCarree()
PROJECTION = ccrs.PlateCarree()

XTICKS = np.arange(0, 181, 30)
YTICKS = np.arange(20, 86, 15)

FS_SMALL = 12
FS_MEDIUM = 13
FS_LARGE = 14

BOX_COLOR = "#D81B60"
BOX_LW = 2.0

COAST_COLOR = "0.25"
COAST_LW = 0.4

SLP_COLOR = "darkgreen"
SLP_LW = 0.75

TEMP_LEVELS = np.array(
    [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    dtype=float
)

TEMP_COLORS = [
    "#21618C",
    "#4F8FC0",
    "#75ADD1",
    "#BFDCEB",
    "#FFFFFF",
    "#F2D3C4",
    "#D97A63",
    "#B84A4A",
    "#9E2F3F"
]

SLP_NEG_LEVELS = np.arange(-10, 0, 1)
SLP_POS_LEVELS = np.arange(1, 11, 1)

STATION_SIZE = 16
STATION_EDGE_WIDTH = 0.35

MOSCOW = (37.6173, 55.7558)
MOSCOW_SIZE = 32


# ============================================================
# 6. 全局字体
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.unicode_minus": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 7. 时间坐标和DJ季节平均
# ============================================================

def fix_time(da):
    if "time" not in da.dims:
        raise ValueError(f"数据缺少time维度: {da.dims}")

    try:
        _ = da.time.dt.year
    except Exception:
        da = da.assign_coords(
            time=("time", pd.to_datetime(da.time.values))
        )

    return da


def seasonal_mean_dj(da):
    """
    DJ(y) = December(y) + January(y+1)
    例如season_year=1940表示1940年12月+1941年1月。
    缺少任意一个月时，该DJ值设为NaN。
    """

    da = fix_time(da).sortby("time")
    da = da.where(da.time.dt.month.isin([12, 1]), drop=True)

    if da.sizes.get("time", 0) == 0:
        raise ValueError("数据中没有12月或1月")

    season_year = (
        da.time.dt.year
        - xr.where(da.time.dt.month == 1, 1, 0)
    )

    da = da.assign_coords(
        season_year=(
            "time",
            np.asarray(season_year.values, dtype=int)
        )
    )

    mean = da.groupby("season_year").mean(
        "time", skipna=True
    )

    count = da.notnull().groupby("season_year").sum("time")

    return mean.where(count == 2).sortby("season_year")


# ============================================================
# 8. 读取PDO与ENSO指数
# ============================================================

def read_indices():

    with xr.open_dataset(PATH_PDO) as ds:
        pdo = fix_time(ds["pdo"]).astype(float).load()

    with xr.open_dataset(PATH_ENSO) as ds:
        enso = fix_time(ds["value"]).astype(float).load()

    # ENSO逐月距平：1971–2000年气候态
    base = enso.sel(
        time=slice("1971-01-01", "2000-12-31")
    )

    if base.sizes.get("time", 0) != 360:
        raise ValueError(
            "ENSO的1971–2000年基准期不完整，应有360个月"
        )

    monthly_clim = base.groupby("time.month").mean(
        "time", skipna=True
    )

    enso = enso.groupby("time.month") - monthly_clim

    # PDO使用原始指数；ENSO使用逐月气候态距平
    pdo_dj = seasonal_mean_dj(pdo)
    enso_dj = seasonal_mean_dj(enso)

    return pdo_dj, enso_dj


# ============================================================
# 9. 读取和预处理格点数据
# ============================================================

def prepare_grid(da, name):

    aliases = {
        "latitude": "lat",
        "longitude": "lon",
        "Latitude": "lat",
        "Longitude": "lon",
        "LATITUDE": "lat",
        "LONGITUDE": "lon",
        "grid_latitude": "lat",
        "grid_longitude": "lon",
        "nav_lat": "lat",
        "nav_lon": "lon",
        "y": "lat",
        "x": "lon"
    }

    rename = {
        old: new
        for old, new in aliases.items()
        if (old in da.dims or old in da.coords)
        and new not in da.dims
        and new not in da.coords
    }

    if rename:
        da = da.rename(rename)

    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(
            f"{name}无法识别经纬度坐标: {list(da.coords)}"
        )

    if da.lat.ndim != 1 or da.lon.ndim != 1:
        raise ValueError(f"{name}的经纬度不是一维规则网格")

    da = fix_time(da)

    # 统一经度到-180至180度
    da = da.assign_coords(
        lon=((da.lon + 180) % 360) - 180
    )

    da = da.sortby("lon").sortby("lat")

    da = da.sel(
        lon=slice(*DATA_LON_RANGE),
        lat=slice(*DATA_LAT_RANGE)
    )

    if da.sizes.get("lon", 0) < 3 or da.sizes.get("lat", 0) < 3:
        raise ValueError(f"{name}裁剪后的网格数量不足")

    da = da.astype("float32").load()

    mean_value = float(da.mean(skipna=True))

    # CRU统一为摄氏度
    if name == "CRU" and mean_value > 100:
        da = da - 273.15

    # SLP统一为hPa
    if name == "SLP" and mean_value > 2000:
        da = da / 100.0

    return da


# ============================================================
# 10. 读取GHCNM站点
# ============================================================

def read_stations():

    with xr.open_dataset(PATH_STA, engine="h5netcdf") as ds:
        temp = fix_time(ds["tavg"]).astype(float).load()

        lon = np.asarray(
            ds["lon"].values, dtype=float
        ).ravel()

        lat = np.asarray(
            ds["lat"].values, dtype=float
        ).ravel()

    temp = temp.where(temp > -9000)

    # 如果为开尔文，转换为摄氏度
    if float(temp.mean(skipna=True)) > 100:
        temp = temp - 273.15

    lon = (lon + 180) % 360 - 180

    station_dim = next(
        d for d in temp.dims if d != "time"
    )

    if (
        temp.sizes[station_dim] != len(lon)
        or len(lon) != len(lat)
    ):
        raise ValueError("站点温度与经纬度数量不一致")

    return temp, lon, lat


# ============================================================
# 11. 同时去除PDO和ENSO的线性影响
# ============================================================

def remove_pdo_enso(y, pdo, enso):
    """
    多元线性回归：
    Y = a + b1*PDO + b2*ENSO + residual

    从Y中扣除PDO和ENSO的线性贡献。
    """

    y, pdo, enso = xr.align(
        y, pdo, enso, join="inner"
    )

    std_pdo = float(
        pdo.std("season_year", skipna=True)
    )

    std_enso = float(
        enso.std("season_year", skipna=True)
    )

    if (
        not np.isfinite(std_pdo)
        or not np.isfinite(std_enso)
        or std_pdo <= 0
        or std_enso <= 0
    ):
        raise ValueError("PDO或ENSO标准差无效")

    # 指数标准化
    x1 = (
        pdo - pdo.mean("season_year")
    ) / std_pdo

    x2 = (
        enso - enso.mean("season_year")
    ) / std_enso

    valid = (
        y.notnull()
        & x1.notnull()
        & x2.notnull()
    )

    n = valid.sum("season_year")

    yy = y.where(valid)
    xx1 = x1.where(valid)
    xx2 = x2.where(valid)

    dy = yy - yy.mean("season_year")
    dx1 = xx1 - xx1.mean("season_year")
    dx2 = xx2 - xx2.mean("season_year")

    a11 = (
        dx1 * dx1
    ).sum("season_year", skipna=True)

    a22 = (
        dx2 * dx2
    ).sum("season_year", skipna=True)

    a12 = (
        dx1 * dx2
    ).sum("season_year", skipna=True)

    b1 = (
        dx1 * dy
    ).sum("season_year", skipna=True)

    b2 = (
        dx2 * dy
    ).sum("season_year", skipna=True)

    det = a11 * a22 - a12 * a12

    good = (
        (n >= MIN_REG_YEARS)
        & (det > 1e-10)
    )

    beta_pdo = (
        (b1 * a22 - b2 * a12) / det
    ).where(good)

    beta_enso = (
        (b2 * a11 - b1 * a12) / det
    ).where(good)

    return (
        y
        - beta_pdo * x1
        - beta_enso * x2
    ).where(good)


# ============================================================
# 12. 事件期减去参考期
# ============================================================

def calculate_anomaly(da):

    event = da.sel(
        season_year=slice(EVENT_START, EVENT_END)
    )

    ref = da.sel(
        season_year=slice(REF_START, REF_END)
    )

    if (
        event.sizes.get("season_year", 0)
        != EVENT_END - EVENT_START + 1
    ):
        raise ValueError("1940–1942事件期年份不完整")

    if (
        ref.sizes.get("season_year", 0)
        != REF_END - REF_START + 1
    ):
        raise ValueError("1971–2000参考期年份不完整")

    return (
        event.mean("season_year", skipna=True)
        - ref.mean("season_year", skipna=True)
    )


# ============================================================
# 13. 图a、b：完全统一风格的指数柱状图
# ============================================================

def draw_index_panel(ax, years, values, label, ylabel):

    ax.set_facecolor("white")

    # 1940–1942年灰色阴影
    ax.axvspan(
        SHADE_YEARS[0],
        SHADE_YEARS[1],
        facecolor=SHADE_COLOR,
        edgecolor="none",
        alpha=SHADE_ALPHA,
        zorder=0
    )

    # 正值红色，负值蓝色
    colors = np.where(
        values >= 0,
        INDEX_POS_COLOR,
        INDEX_NEG_COLOR
    )

    ax.bar(
        years,
        values,
        width=INDEX_BAR_WIDTH,
        color=colors,
        edgecolor="none",
        linewidth=0,
        zorder=3
    )

    # 零线
    ax.axhline(
        0,
        color=ZERO_COLOR,
        linewidth=ZERO_LW,
        zorder=4
    )

    # 统一坐标范围
    ax.set_xlim(*INDEX_XLIM)
    ax.set_ylim(*INDEX_YLIM)

    ax.set_xticks(INDEX_XTICKS)
    ax.set_yticks(INDEX_YTICKS)
    ax.set_xticks([], minor=True)

    # 统一刻度样式
    ax.tick_params(
        axis="both",
        direction="out",
        length=2.4,
        width=0.6,
        pad=1.5,
        labelsize=10
    )

    for tick in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        tick.set_fontname("Arial")
        tick.set_fontweight("normal")
        tick.set_fontsize(10)

    # 坐标轴标签
    ax.set_xlabel(
        "Year",
        fontsize=11,
        fontname="Arial",
        fontweight="normal"
    )

    ax.set_ylabel(
        ylabel,
        fontsize=11,
        fontname="Arial",
        fontweight="normal",
        rotation=90
    )

    ax.xaxis.set_label_coords(0.5, -0.15)
    ax.yaxis.set_label_coords(-0.075, 0.5)

    # 统一边框
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#333333")

    # 子图序号
    ax.text(
        0,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom"
    )


# ============================================================
# 14. 图c、d：地图绘图函数
# ============================================================

def draw_map_panel(
    ax,
    cru_map,
    slp_map,
    sta_map,
    station_lon,
    station_lat,
    panel_label,
    panel_title,
    land,
    cmap,
    norm
):

    ax.set_extent(
        MAP_EXTENT,
        crs=DATA_CRS
    )

    ax.set_facecolor("white")
    ax.set_aspect("auto")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#333333")

    # --------------------------------------------------------
    # CRU温度填色，仅显示陆地
    # --------------------------------------------------------

    cru_plot = cru_map.sel(
        lon=slice(*DATA_LON_RANGE),
        lat=slice(*DATA_LAT_RANGE)
    )

    xx, yy = np.meshgrid(
        cru_plot.lon.values,
        cru_plot.lat.values
    )

    try:
        from shapely import contains_xy
        land_mask = contains_xy(land, xx, yy)

    except ImportError:
        try:
            from shapely.vectorized import contains
            land_mask = contains(land, xx, yy)

        except ImportError:
            prepared = prep(land)

            land_mask = np.array([
                prepared.contains(
                    Point(float(x), float(y))
                )
                or land.touches(
                    Point(float(x), float(y))
                )
                for x, y in zip(
                    xx.ravel(),
                    yy.ravel()
                )
            ]).reshape(xx.shape)

    cru_plot = cru_plot.where(
        xr.DataArray(
            land_mask,
            coords={
                "lat": cru_plot.lat,
                "lon": cru_plot.lon
            },
            dims=("lat", "lon")
        )
    )

    mesh = ax.contourf(
        xx,
        yy,
        cru_plot.values,
        levels=TEMP_LEVELS,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=DATA_CRS,
        zorder=10
    )

    # --------------------------------------------------------
    # 海岸线
    # --------------------------------------------------------

    ax.coastlines(
        resolution="50m",
        color=COAST_COLOR,
        linewidth=COAST_LW,
        zorder=90
    )

    # --------------------------------------------------------
    # SLP等值线
    # --------------------------------------------------------

    slp_plot = slp_map.sel(
        lon=slice(*DATA_LON_RANGE),
        lat=slice(*DATA_LAT_RANGE)
    )

    sx, sy = np.meshgrid(
        slp_plot.lon.values,
        slp_plot.lat.values
    )

    sv = np.asarray(
        slp_plot.values,
        dtype=float
    )

    if np.isfinite(sv).any():

        vmin = np.nanmin(sv)
        vmax = np.nanmax(sv)

        for levels, style in [
            (SLP_NEG_LEVELS, "dashed"),
            (SLP_POS_LEVELS, "solid")
        ]:

            use_levels = [
                v for v in levels
                if vmin < v < vmax
            ]

            if not use_levels:
                continue

            cs = ax.contour(
                sx,
                sy,
                sv,
                levels=use_levels,
                colors=SLP_COLOR,
                linewidths=SLP_LW,
                linestyles=style,
                transform=DATA_CRS,
                zorder=110
            )

            texts = ax.clabel(
                cs,
                fmt="%g",
                fontsize=FS_SMALL,
                inline=False,
                colors="black"
            )

            for txt in texts:
                txt.set_path_effects([
                    pe.withStroke(
                        linewidth=1.0,
                        foreground="white"
                    ),
                    pe.Normal()
                ])

    # --------------------------------------------------------
    # 经纬度刻度
    # --------------------------------------------------------

    ax.set_xticks(
        XTICKS,
        crs=DATA_CRS
    )

    ax.set_yticks(
        YTICKS,
        crs=DATA_CRS
    )

    ax.set_xticklabels([
        "0°" if x == 0 else f"{int(x)}°E"
        for x in XTICKS
    ], fontsize=FS_SMALL)

    ax.set_yticklabels([
        f"{int(y)}°N"
        for y in YTICKS
    ], fontsize=FS_SMALL)

    ax.tick_params(
        axis="both",
        direction="out",
        length=2.4,
        width=0.6,
        pad=1.5
    )

    # --------------------------------------------------------
    # 研究区方框
    # --------------------------------------------------------

    x0, x1, y0, y1 = BOX

    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        transform=DATA_CRS,
        color=BOX_COLOR,
        linewidth=BOX_LW,
        zorder=150,
        path_effects=[
            pe.Stroke(
                linewidth=BOX_LW + 0.8,
                foreground="white"
            ),
            pe.Normal()
        ]
    )

    # --------------------------------------------------------
    # GHCNM站点
    # --------------------------------------------------------

    station_values = np.asarray(
        sta_map.values,
        dtype=float
    ).ravel()

    valid = (
        np.isfinite(station_lon)
        & np.isfinite(station_lat)
        & np.isfinite(station_values)
    )

    ax.scatter(
        station_lon[valid],
        station_lat[valid],
        c=station_values[valid],
        s=STATION_SIZE,
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=STATION_EDGE_WIDTH,
        transform=DATA_CRS,
        zorder=170
    )

    # --------------------------------------------------------
    # 莫斯科星标
    # --------------------------------------------------------

    ax.scatter(
        MOSCOW[0],
        MOSCOW[1],
        s=MOSCOW_SIZE,
        marker="*",
        color="black",
        edgecolors="black",
        linewidths=0.35,
        transform=DATA_CRS,
        zorder=190
    )

    ax.legend(
        handles=[
            Line2D(
                [0], [0],
                marker="*",
                color="none",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=8,
                label="Moscow"
            )
        ],
        loc="lower left",
        bbox_to_anchor=(0.015, 0.03),
        fontsize=FS_SMALL,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#333333"
    )

    # 子图序号与标题
    ax.text(
        0,
        1.02,
        panel_label,
        transform=ax.transAxes,
        fontsize=FS_LARGE,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    if panel_title:
        ax.text(
            0.055,
            1.02,
            panel_title,
            transform=ax.transAxes,
            fontsize=FS_MEDIUM,
            ha="left",
            va="bottom"
        )

    return mesh


# ============================================================
# 15. 读取数据
# ============================================================

print("读取CRU、HadSLP2和GHCNM数据...", flush=True)

with xr.open_dataset(PATH_CRU) as ds:
    cru = prepare_grid(ds["tmp"], "CRU")

with xr.open_dataset(PATH_SLP) as ds:
    slp = prepare_grid(ds["slp"], "SLP")

sta, station_lon, station_lat = read_stations()

print("计算DJ季节平均...", flush=True)

cru_dj = seasonal_mean_dj(cru)
slp_dj = seasonal_mean_dj(slp)
sta_dj = seasonal_mean_dj(sta)

pdo_dj, enso_dj = read_indices()

# ------------------------------------------------------------
# 图a、b：PDO和ENSO指数时间序列
# ------------------------------------------------------------

index_years = np.arange(
    INDEX_START,
    INDEX_END + 1
)

pdo_values = pdo_dj.reindex(
    season_year=index_years
).values.astype(float)

enso_values = enso_dj.reindex(
    season_year=index_years
).values.astype(float)

# ------------------------------------------------------------
# 图c：原始气候异常
# ------------------------------------------------------------

print("计算原始异常（图c）...", flush=True)

cru_raw = calculate_anomaly(cru_dj)
slp_raw = calculate_anomaly(slp_dj)
sta_raw = calculate_anomaly(sta_dj)

# ------------------------------------------------------------
# 图d：同时去除PDO和ENSO
# ------------------------------------------------------------

print(
    "同时回归去除PDO和ENSO（图d）...",
    flush=True
)

cru_res = remove_pdo_enso(
    cru_dj, pdo_dj, enso_dj
)

slp_res = remove_pdo_enso(
    slp_dj, pdo_dj, enso_dj
)

sta_res = remove_pdo_enso(
    sta_dj, pdo_dj, enso_dj
)

cru_removed = calculate_anomaly(cru_res)
slp_removed = calculate_anomaly(slp_res)
sta_removed = calculate_anomaly(sta_res)

# ------------------------------------------------------------
# Natural Earth陆地边界
# ------------------------------------------------------------

print("读取Natural Earth陆地边界...", flush=True)

land = unary_union(list(
    shpreader.Reader(
        shpreader.natural_earth(
            resolution="110m",
            category="physical",
            name="land"
        )
    ).geometries()
))


# ============================================================
# 16. 绘制四面板
# ============================================================

fig = plt.figure(
    figsize=FIGSIZE,
    dpi=DPI,
    facecolor="white"
)

ax_a = fig.add_axes(AX_A_POS)
ax_b = fig.add_axes(AX_B_POS)

ax_c = fig.add_axes(
    AX_C_POS,
    projection=PROJECTION
)

ax_d = fig.add_axes(
    AX_D_POS,
    projection=PROJECTION
)

cax = fig.add_axes(CAX_POS)


# ------------------------------------------------------------
# 第一行：a、b风格完全一致
# ------------------------------------------------------------

draw_index_panel(
    ax_a,
    index_years,
    pdo_values,
    "(a)",
    "PDO Index"
)

draw_index_panel(
    ax_b,
    index_years,
    enso_values,
    "(b)",
    "Niño 3.4 Index"
)


# ------------------------------------------------------------
# 第二、三行：c、d共用温度色阶
# ------------------------------------------------------------

cmap = ListedColormap(
    TEMP_COLORS,
    name="sat_center_white"
)

norm = BoundaryNorm(
    TEMP_LEVELS,
    cmap.N,
    clip=False
)

# 图c：原始异常
mesh = draw_map_panel(
    ax_c,
    cru_raw,
    slp_raw,
    sta_raw,
    station_lon,
    station_lat,
    "(c)",
    "",
    land,
    cmap,
    norm
)

# 图d：同时去除PDO和ENSO后的异常
draw_map_panel(
    ax_d,
    cru_removed,
    slp_removed,
    sta_removed,
    station_lon,
    station_lat,
    "(d)",
    "removing PDO and ENSO",
    land,
    cmap,
    norm
)


# ============================================================
# 17. 共用色标
# ============================================================

cb = fig.colorbar(
    mesh,
    cax=cax,
    orientation="vertical",
    extend="both",
    boundaries=TEMP_LEVELS,
    ticks=TEMP_LEVELS,
    spacing="uniform",
    drawedges=True
)

cb.ax.set_yticklabels([
    str(int(v)).replace("-", "−")
    for v in TEMP_LEVELS
])

cb.ax.tick_params(
    labelsize=FS_SMALL,
    length=2.0,
    width=0.5,
    pad=1.5
)

cb.outline.set_linewidth(0.5)
cb.outline.set_edgecolor("#333333")

cb.set_label(
    "Surface Air Temperature (°C)",
    fontsize=FS_MEDIUM
)


# ============================================================
# 18. 纬度刻度描边与显示
# ============================================================

fig.canvas.draw()

for txt in fig.findobj(match=Text):
    if "°N" in txt.get_text():
        txt.set_path_effects([
            pe.withStroke(
                linewidth=1.0,
                foreground="white"
            ),
            pe.Normal()
        ])

plt.show()

print(
    "完成："
    "a=PDO DJ指数；"
    "b=Niño 3.4 DJ指数；"
    "c=原始气候异常；"
    "d=同时去除PDO和ENSO后的气候异常。"
)
