
# -*- coding: utf-8 -*-
"""Fig1b: DJ温度与SLP异常，同时去除PDO和ENSO"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.text import Text

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


# ============================================================
# 1. 参数
# ============================================================

PATH_CRU = r"E:\20century\data\cru_ts4.05.1901.2020.tmp.dat.nc"
PATH_STATION = r"E:\all station\GHCNM 80stations.nc"
PATH_PDO = r"E:\20century\data\pdo.timeseries.sstens.nc"
PATH_ENSO = r"E:\20century\data\nino34.long.anom.nc"
PATH_SLP = r"E:\20century\data\hadslp2_slp.nc"

EVENT = (1940, 1942)
REF = (1971, 2000)

BOX = [30, 60, 45, 60]
MAP_EXTENT = [-30, 90, 30, 85]
LON_RANGE = (-35, 95)
LAT_RANGE = (25, 90)

FIGSIZE = (3.75, 2.70)
AX_POS = [0.13, 0.18, 0.68, 0.72]
CAX_POS = [0.845, 0.25, 0.032, 0.58]
DPI = 600
SAVE_FIG = None

FS_SMALL, FS_MEDIUM, FS_LARGE = 8, 9, 10
TICK_LENGTH, TICK_WIDTH, TICK_PAD = 2.4, 0.6, 1.5

BOX_COLOR, BOX_LW = "#D81B60", 3.0
COAST_COLOR, COAST_LW = "0.25", 0.45
SLP_COLOR, SLP_LW = "darkgreen", 1.0

TEMP_LEVELS = np.array(
    [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], dtype=float
)
TEMP_COLORS = [
    "#21618C", "#4F8FC0", "#75ADD1", "#BFDCEB", "#FFFFFF",
    "#F2D3C4", "#D97A63", "#B84A4A", "#9E2F3F"
]

SLP_NEG = np.arange(-10, 0, 1)
SLP_POS = np.arange(1, 10.5, 1)
XTICKS = np.arange(-30, 91, 30)
YTICKS = np.arange(30, 86, 15)

MOSCOW = (37.6173, 55.7558)
DATA_CRS = PROJECTION = ccrs.PlateCarree()

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
    "axes.labelsize": FS_MEDIUM,
    "axes.titlesize": FS_LARGE,
    "xtick.labelsize": FS_SMALL,
    "ytick.labelsize": FS_SMALL,
    "legend.fontsize": FS_MEDIUM,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.unicode_minus": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 2. 数据处理与绘图：保留一个接口
# ============================================================

def draw_panel(ax_b, cax):

    fig = ax_b.figure

    # ---------- 读取数据 ----------
    land = unary_union(list(shpreader.Reader(
        shpreader.natural_earth(
            resolution="110m", category="physical", name="land"
        )
    ).geometries()))

    with xr.open_dataset(PATH_CRU) as ds:
        cru = ds.tmp.load()
    with xr.open_dataset(PATH_SLP) as ds:
        slp = ds.slp.load()
    with xr.open_dataset(PATH_PDO) as ds:
        pdo = ds.pdo.astype(float).load()
    with xr.open_dataset(PATH_ENSO) as ds:
        enso = ds["value"].astype(float).load()

    station_ds = xr.open_dataset(PATH_STATION)
    tavg = station_ds.tavg.astype(float).where(
        station_ds.tavg > -9000
    ).load()

    station_dim = next(d for d in tavg.dims if d != "time")
    station_lat = np.asarray(station_ds.lat, dtype=float)
    station_lon = np.asarray(station_ds.lon, dtype=float)

    if np.nanmax(station_lon) > 180:
        station_lon = (station_lon + 180) % 360 - 180

    if float(tavg.mean(skipna=True)) > 100:
        tavg = tavg - 273.15

    # ---------- 统一经纬度与温度单位 ----------
    aliases = {
        "latitude": "lat", "longitude": "lon",
        "Latitude": "lat", "Longitude": "lon",
        "grid_latitude": "lat", "grid_longitude": "lon",
        "nav_lat": "lat", "nav_lon": "lon",
        "y": "lat", "x": "lon"
    }

    for name, da in [("cru", cru), ("slp", slp)]:
        rename = {
            old: new for old, new in aliases.items()
            if (old in da.dims or old in da.coords)
            and new not in da.dims and new not in da.coords
        }
        if rename:
            da = da.rename(rename)

        da = da.sortby("lat").assign_coords(
            lon=(da.lon + 180) % 360 - 180
        ).sortby("lon").astype("float32")

        if name == "cru" and float(da.mean(skipna=True)) > 100:
            da = da - np.float32(273.15)

        if name == "cru":
            cru = da
        else:
            slp = da

    # ---------- 统一时间坐标 ----------
    data = {
        "cru": cru, "slp": slp, "station": tavg,
        "pdo": pdo, "enso": enso
    }

    for name, da in data.items():
        try:
            _ = da.time.dt.year
        except Exception:
            da = da.assign_coords(
                time=("time", pd.to_datetime(da.time.values))
            )
        data[name] = da.sortby("time")

    # ENSO逐月距平：1971–2000气候态
    enso_base = data["enso"].sel(
        time=slice("1971-01-01", "2000-12-31")
    )
    if enso_base.sizes.get("time", 0) != 360:
        raise ValueError("ENSO 1971–2000基准期不完整，应有360个月")

    enso_clim = enso_base.groupby("time.month").mean(
        "time", skipna=True
    )
    data["enso"] = data["enso"].groupby("time.month") - enso_clim

    # ---------- DJ平均：12月(y) + 1月(y+1) ----------
    seasonal = {}

    for name, da in data.items():
        da = da.where(da.time.dt.month.isin([12, 1]), drop=True)

        season_year = (
            da.time.dt.year
            - xr.where(da.time.dt.month == 1, 1, 0)
        )
        da = da.assign_coords(
            season_year=("time", np.asarray(season_year.values, dtype=int))
        )

        dj = da.groupby("season_year").mean("time", skipna=True)
        count = da.notnull().groupby("season_year").sum("time")
        seasonal[name] = dj.where(count == 2).sortby("season_year")

    # ---------- 同时去除PDO和ENSO ----------
    residual = {}

    for name in ("cru", "slp", "station"):
        y, p, e = xr.align(
            seasonal[name], seasonal["pdo"], seasonal["enso"],
            join="inner"
        )

        # 两个指数标准化
        p = (p - p.mean("season_year")) / p.std(
            "season_year", skipna=True
        )
        e = (e - e.mean("season_year")) / e.std(
            "season_year", skipna=True
        )

        valid = y.notnull() & p.notnull() & e.notnull()
        n = valid.sum("season_year")

        yy, x1, x2 = y.where(valid), p.where(valid), e.where(valid)
        dy = yy - yy.mean("season_year")
        dx1 = x1 - x1.mean("season_year")
        dx2 = x2 - x2.mean("season_year")

        a11 = (dx1 * dx1).sum("season_year", skipna=True)
        a22 = (dx2 * dx2).sum("season_year", skipna=True)
        a12 = (dx1 * dx2).sum("season_year", skipna=True)
        b1 = (dx1 * dy).sum("season_year", skipna=True)
        b2 = (dx2 * dy).sum("season_year", skipna=True)

        det = a11 * a22 - a12 * a12
        good = (n >= 30) & (det > 1e-10)

        beta_pdo = ((b1 * a22 - b2 * a12) / det).where(good)
        beta_enso = ((b2 * a11 - b1 * a12) / det).where(good)

        residual[name] = (
            y - beta_pdo * p - beta_enso * e
        ).where(good)

    # ---------- 事件期减参考期 ----------
    maps = {}
    for name, da in residual.items():
        maps[name] = (
            da.sel(season_year=slice(*EVENT)).mean(
                "season_year", skipna=True
            )
            - da.sel(season_year=slice(*REF)).mean(
                "season_year", skipna=True
            )
        )

    cru_map, slp_map, station_map = (
        maps["cru"], maps["slp"], maps["station"]
    )

    station_value = np.asarray(station_map, dtype=float)
    valid_sta = (
        np.isfinite(station_lon)
        & np.isfinite(station_lat)
        & np.isfinite(station_value)
    )
    station_lon = station_lon[valid_sta]
    station_lat = station_lat[valid_sta]
    station_value = station_value[valid_sta]

    # ========================================================
    # 3. 绘图：沿用原Fig1b样式
    # ========================================================

    cmap = ListedColormap(TEMP_COLORS, name="sat_center_white")
    norm = BoundaryNorm(TEMP_LEVELS, cmap.N, clip=False)

    ax_b.set_extent(MAP_EXTENT, crs=DATA_CRS)
    ax_b.set_facecolor("white")
    ax_b.set_aspect("auto")

    for spine in ax_b.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_edgecolor("black")

    clip = ax_b.patch

    # 陆地温度填色
    cru_plot = cru_map.sel(
        lon=slice(*LON_RANGE), lat=slice(*LAT_RANGE)
    )
    lon2d, lat2d = np.meshgrid(cru_plot.lon, cru_plot.lat)

    try:
        from shapely import contains_xy
        land_mask = contains_xy(land, lon2d, lat2d)
    except ImportError:
        try:
            from shapely.vectorized import contains
            land_mask = contains(land, lon2d, lat2d)
        except ImportError:
            prepared = prep(land)
            land_mask = np.array([
                prepared.contains(Point(float(x), float(y)))
                or land.touches(Point(float(x), float(y)))
                for x, y in zip(lon2d.ravel(), lat2d.ravel())
            ]).reshape(lon2d.shape)

    cru_plot = cru_plot.where(xr.DataArray(
        land_mask,
        coords={"lat": cru_plot.lat, "lon": cru_plot.lon},
        dims=("lat", "lon")
    ))

    mesh = ax_b.contourf(
        lon2d, lat2d, cru_plot.values,
        levels=TEMP_LEVELS, cmap=cmap, norm=norm, extend="both",
        transform=DATA_CRS, zorder=10, antialiased=True,
        corner_mask=True
    )
    for item in getattr(mesh, "collections", []):
        item.set_clip_path(clip)
        item.set_rasterized(True)

    coast = ax_b.coastlines(
        resolution="50m", color=COAST_COLOR,
        lw=COAST_LW, zorder=95
    )
    try:
        coast.set_clip_path(clip)
    except Exception:
        pass

    # SLP等值线
    slp_plot = slp_map.sel(
        lon=slice(*LON_RANGE), lat=slice(*LAT_RANGE)
    )
    slon, slat = np.meshgrid(slp_plot.lon, slp_plot.lat)
    labels = []

    for levels, style in [(SLP_NEG, "dashed"), (SLP_POS, "solid")]:
        cs = ax_b.contour(
            slon, slat, slp_plot, levels=levels,
            colors=SLP_COLOR, linewidths=SLP_LW,
            linestyles=style, transform=DATA_CRS, zorder=118
        )
        for item in getattr(cs, "collections", []):
            item.set_clip_path(clip)

        labels += ax_b.clabel(
            cs,
            fmt=lambda v: f"{v:.1f}".rstrip("0").rstrip(".").replace("-", "−"),
            inline=False, inline_spacing=2,
            fontsize=FS_SMALL, colors="black"
        )

    for txt in labels:
        txt.set_fontweight("normal")
        txt.set_color("black")
        txt.set_bbox(None)
        txt.set_path_effects([
            pe.withStroke(linewidth=1.2, foreground="white"),
            pe.Normal()
        ])

    # 经纬度刻度
    ax_b.set_xticks(XTICKS, crs=DATA_CRS)
    ax_b.set_yticks(YTICKS, crs=DATA_CRS)
    ax_b.set_xticklabels([
        f"{abs(int(x))}°W" if x < 0
        else f"{int(x)}°E" if x > 0 else "0°"
        for x in XTICKS
    ], fontsize=FS_SMALL)

    ax_b.set_yticklabels([
        "" if y == 30 else f"{int(y)}°N"
        for y in YTICKS
    ], fontsize=FS_SMALL)

    ax_b.tick_params(
        axis="both", which="major", direction="out",
        length=TICK_LENGTH, width=TICK_WIDTH,
        pad=TICK_PAD, labelsize=FS_SMALL
    )
    ax_b.grid(False)

    # 研究区方框
    box_lines = [
        (np.linspace(BOX[0], BOX[1], 250), np.full(250, BOX[2])),
        (np.full(160, BOX[1]), np.linspace(BOX[2], BOX[3], 160)),
        (np.linspace(BOX[1], BOX[0], 250), np.full(250, BOX[3])),
        (np.full(160, BOX[0]), np.linspace(BOX[3], BOX[2], 160))
    ]

    for x, y in box_lines:
        line, = ax_b.plot(
            x, y, transform=DATA_CRS, color=BOX_COLOR,
            lw=BOX_LW, zorder=155,
            path_effects=[
                pe.Stroke(linewidth=BOX_LW + 0.9, foreground="white"),
                pe.Normal()
            ]
        )
        line.set_clip_path(clip)

    # 站点与莫斯科
    print("图(b)实际绘制站点数量：", len(station_value), flush=True)

    pts = ax_b.scatter(
        station_lon, station_lat, c=station_value,
        s=16, cmap=cmap, norm=norm,
        edgecolors="black", linewidths=0.35,
        transform=DATA_CRS, zorder=175
    )
    pts.set_clip_path(clip)

    star = ax_b.scatter(
        *MOSCOW, s=32, marker="*", c="black",
        edgecolors="black", linewidths=0.35,
        transform=DATA_CRS, zorder=190
    )
    star.set_clip_path(clip)

    legend = ax_b.legend(
        handles=[Line2D(
            [0], [0], marker="*", color="none",
            markerfacecolor="black", markeredgecolor="black",
            markersize=8, label="Moscow"
        )],
        loc="lower left", bbox_to_anchor=(0.055, 0.055),
        borderaxespad=0.0, frameon=True, fancybox=False,
        framealpha=0.90, facecolor="white", edgecolor="#333333",
        fontsize=FS_SMALL, handlelength=0.8,
        handletextpad=0.35, borderpad=0.25, labelspacing=0.25
    )
    legend.set_zorder(206)

    ax_b.text(
        0, 1.020, "(b)", transform=ax_b.transAxes,
        fontsize=FS_LARGE, fontweight="bold",
        ha="left", va="bottom", clip_on=False, zorder=400
    )

    # 色标
    cb = fig.colorbar(
        mesh, cax=cax, orientation="vertical",
        extend="both", boundaries=TEMP_LEVELS,
        ticks=TEMP_LEVELS, spacing="uniform", drawedges=True
    )
    cb.ax.set_yticklabels([
        str(int(v)).replace("-", "−") for v in TEMP_LEVELS
    ])
    cb.ax.tick_params(
        labelsize=FS_SMALL, length=2.0, width=0.5, pad=1.5
    )
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#333333")

    for spine in cb.ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#333333")

    try:
        cb.dividers.set_color("#333333")
        cb.dividers.set_linewidth(0.25)
    except Exception:
        pass

    cb.set_label(
        "Surface Air Temperature (°C)",
        fontsize=FS_MEDIUM, fontweight="normal", labelpad=4.0
    )

    station_ds.close()
    return ax_b


# 单独运行Fig1b时使用；合并运行时由Fig1.py调用draw_panel()
if __name__ == "__main__":
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes(AX_POS, projection=PROJECTION)
    cax = fig.add_axes(CAX_POS)
    draw_panel(ax, cax)

    fig.canvas.draw()
    for txt in fig.findobj(match=Text):
        if "°N" in txt.get_text():
            txt.set_path_effects([
                pe.withStroke(linewidth=1.0, foreground="white"),
                pe.Normal()
            ])

    if SAVE_FIG:
        fig.savefig(SAVE_FIG, dpi=DPI, bbox_inches="tight")

    plt.show()
