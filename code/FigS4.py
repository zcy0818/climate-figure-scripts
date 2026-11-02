# -*- coding: utf-8 -*-
"""FigS4: four observational SAT datasets with GHCN-M stations and Moscow; simultaneous PDO/ENSO regression."""

import os
import hashlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from shapely.geometry import Point
from shapely.prepared import prep
from shapely.ops import unary_union


# ============================================================
# 1. 可编辑参数
# ============================================================

# ---------- 数据路径 ----------
PATH_GIS = r"E:\20century\data\gistemp1200 temanomaly1951-1980_GHCNv4_.nc"
PATH_CRU = r"E:\20century\data\cru_ts4.05.1901.2020.tmp.dat.nc"
PATH_BERKELEY = r"E:\20century\data\berkeley.nc"
PATH_NANSEN = r"E:\20century\data\NANSENSAT1900.nc"
PATH_STATION = r"E:\all station\GHCNM 80stations.nc"
PATH_PDO = r"E:\20century\data\pdo.timeseries.sstens.nc"
PATH_ENSO = r"E:\20century\data\nino34.long.anom.nc"

# ---------- 时段 ----------
TARGET_YEAR_START = 1940
TARGET_YEAR_END = 1942
REF_YEAR_START = 1971
REF_YEAR_END = 2000
TARGET_MONTHS = [12, 1]
MIN_REG_YEARS = 30
NANSEN_TIME_START = "1900-01-01"

# ---------- 地图 ----------
MAP_EXTENT = [-30, 90, 30, 85]
MAP_LON_MIN, MAP_LON_MAX, MAP_LAT_MIN, MAP_LAT_MAX = MAP_EXTENT
PROJ = ccrs.PlateCarree()
DATA_CRS = ccrs.PlateCarree()
MOSCOW = (37.6173, 55.7558)  # longitude, latitude; same as Fig1b

XTICKS = np.arange(-30, 91, 30)
YTICKS = np.arange(30, 86, 15)

# ---------- 画布：每个地图面板的物理尺寸与 Fig1b 一致 ----------
# Fig1b: 3.75×2.70 in, axes 0.68×0.72 -> 2.55×1.944 in per map.
# FigS4: 7.50×5.40 in, each axes 0.34×0.36 -> the same 2.55×1.944 in.
FIGSIZE = (7.50, 5.40)
DPI = 600
FIG_FACECOLOR = "white"

AX_POSITIONS = [
    [0.0650, 0.5900, 0.3400, 0.3600],
    [0.5650, 0.5900, 0.3400, 0.3600],
    [0.0650, 0.0900, 0.3400, 0.3600],
    [0.5650, 0.0900, 0.3400, 0.3600],
]
CBAR_POSITIONS = [
    [0.4225, 0.6250, 0.0160, 0.2900],
    [0.9225, 0.6250, 0.0160, 0.2900],
    [0.4225, 0.1250, 0.0160, 0.2900],
    [0.9225, 0.1250, 0.0160, 0.2900],
]

# ---------- 字号 ----------
FS_SMALL = 8
FS_MEDIUM = 9
FS_LARGE = 10
TICK_LENGTH = 2.4
TICK_WIDTH = 0.6
TICK_PAD = 1.5

# ---------- SAT 色阶 ----------
TEMP_LEVELS = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], dtype=float)
TEMP_COLORS = [
    "#21618C",
    "#4F8FC0",
    "#75ADD1",
    "#BFDCEB",
    "#FFFFFF",
    "#F2D3C4",
    "#D97A63",
    "#B84A4A",
    "#9E2F3F",
]
TEMP_CMAP = ListedColormap(TEMP_COLORS, name="sat_center_white")
TEMP_NORM = BoundaryNorm(TEMP_LEVELS, TEMP_CMAP.N, clip=False)

# ---------- 陆地 mask ----------
LAND_MASK_SCALE = "110m"
LAND_MASK_CACHE_DIR = r"E:\20century\data\land_mask_cache_110m"

# ---------- 研究区 BOX ----------
BOX_LON1, BOX_LON2 = 30, 60
BOX_LAT1, BOX_LAT2 = 45, 60
BOX_COLOR = "#D81B60"
BOX_LW = 3.0
BOX_OPACITY = 1.0

# ---------- 海岸线 ----------
COAST_COLOR = "0.25"
COAST_LW = 0.45

# ---------- 保存 ----------
SAVE_FIGURE = False
SHOW_FIGURE = True
OUT_PNG = "FigS4.png"
OUT_PDF = "FigS4.pdf"


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
# 2. 少量必要工具
# ============================================================

def ensure_time(ds):
    """尽量把 time 解码为可用的 datetime/cftime。"""
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("Dataset has no time coordinate or time dimension.")

    try:
        if np.issubdtype(ds["time"].dtype, np.datetime64):
            return ds
    except Exception:
        pass

    try:
        return xr.decode_cf(ds)
    except Exception:
        return ds


def prep_grid(da, label):
    """统一 lat/lon 名称、纬度升序和 -180~180° 经度。"""
    aliases = {
        "latitude": "lat",
        "longitude": "lon",
        "Latitude": "lat",
        "Longitude": "lon",
        "grid_latitude": "lat",
        "grid_longitude": "lon",
        "nav_lat": "lat",
        "nav_lon": "lon",
        "y": "lat",
        "x": "lon",
    }

    rename = {
        old: new
        for old, new in aliases.items()
        if old in da.dims or old in da.coords
    }
    if rename:
        da = da.rename(rename)

    if "lat" not in da.dims and "lat" not in da.coords:
        raise ValueError(f"{label} 中未找到纬度坐标，dims={da.dims}")
    if "lon" not in da.dims and "lon" not in da.coords:
        raise ValueError(f"{label} 中未找到经度坐标，dims={da.dims}")

    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")

    da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    return da.astype(float)


def seasonal_mean(da, months):
    """1–3月归入前一个 season_year，与原代码一致。"""
    da = da.sortby("time")
    da = da.where(da.time.dt.month.isin(months), drop=True)

    if da.time.size == 0:
        raise ValueError(f"没有筛选到月份 {months} 的数据。")

    years = da.time.dt.year.values.astype(int)
    mons = da.time.dt.month.values.astype(int)
    season_year = np.where(np.isin(mons, [1, 2, 3]), years - 1, years)

    da = da.assign_coords(season_year=("time", season_year))
    return da.groupby("season_year").mean("time", skipna=True).sortby("season_year")


def regression_anomaly(
    da, months, name, pdo_yearly, enso_yearly,
    require_complete_months=False,
):
    """先按DJ合成年份，再同时回归去除PDO和ENSO，最后计算目标期减基准期。

    require_complete_months=True时，要求每个站点的DJ季节同时具有12月和次年1月，
    与Fig1b的站点处理保持一致；网格资料仍沿用原有季节平均方式。
    """
    if require_complete_months:
        da = da.sortby("time")
        da = da.where(da.time.dt.month.isin(months), drop=True)
        years = da.time.dt.year.values.astype(int)
        mons = da.time.dt.month.values.astype(int)
        season_year = np.where(np.isin(mons, [1, 2, 3]), years - 1, years)
        da = da.assign_coords(season_year=("time", season_year))
        field = da.groupby("season_year").mean("time", skipna=True)
        month_count = da.notnull().groupby("season_year").sum("time")
        field = field.where(month_count == len(months)).sortby("season_year")
    else:
        field = seasonal_mean(da, months)

    field, pdo, enso = xr.align(field, pdo_yearly, enso_yearly, join="inner")

    if field.sizes.get("season_year", 0) < MIN_REG_YEARS:
        raise ValueError(f"{name}: 与PDO/ENSO共同有效年份不足{MIN_REG_YEARS}年")

    pdo = (pdo - pdo.mean("season_year", skipna=True)) / pdo.std("season_year", skipna=True)
    enso = (enso - enso.mean("season_year", skipna=True)) / enso.std("season_year", skipna=True)

    valid = field.notnull() & pdo.notnull() & enso.notnull()
    n = valid.sum("season_year")
    y_mean = field.where(valid).mean("season_year", skipna=True)
    p_mean = pdo.where(valid).mean("season_year", skipna=True)
    e_mean = enso.where(valid).mean("season_year", skipna=True)

    dy = (field - y_mean).where(valid)
    dp = (pdo - p_mean).where(valid)
    de = (enso - e_mean).where(valid)

    a11 = (dp * dp).sum("season_year", skipna=True)
    a22 = (de * de).sum("season_year", skipna=True)
    a12 = (dp * de).sum("season_year", skipna=True)
    b1 = (dp * dy).sum("season_year", skipna=True)
    b2 = (de * dy).sum("season_year", skipna=True)
    det = a11 * a22 - a12 * a12

    good = (n >= MIN_REG_YEARS) & (det > 1e-10)
    beta_pdo = ((b1 * a22 - b2 * a12) / det).where(good)
    beta_enso = ((b2 * a11 - b1 * a12) / det).where(good)

    residual = (
        field
        - beta_pdo * (pdo - p_mean)
        - beta_enso * (enso - e_mean)
    ).where(good)

    target = residual.sel(season_year=slice(TARGET_YEAR_START, TARGET_YEAR_END))
    ref = residual.sel(season_year=slice(REF_YEAR_START, REF_YEAR_END))
    if target.sizes.get("season_year", 0) != TARGET_YEAR_END - TARGET_YEAR_START + 1:
        raise ValueError(f"{name}: 目标时段年份不完整")
    if ref.sizes.get("season_year", 0) != REF_YEAR_END - REF_YEAR_START + 1:
        raise ValueError(f"{name}: 基准期年份不完整")

    print(f"   {name}: 已同时回归去除PDO和ENSO。")
    return target.mean("season_year", skipna=True) - ref.mean("season_year", skipna=True)


def land_mask_on_grid(da, land_union, land_prepared):
    """Natural Earth 110m 陆地mask；每套网格独立缓存。"""
    lon = np.asarray(da.lon.values, dtype=float)
    lat = np.asarray(da.lat.values, dtype=float)
    key = hashlib.md5(lon.tobytes() + lat.tobytes()).hexdigest()[:12]
    cache = os.path.join(
        LAND_MASK_CACHE_DIR,
        f"land_mask_{LAND_MASK_SCALE}_{key}.nc",
    )

    if os.path.exists(cache):
        try:
            ds_cache = xr.open_dataset(cache)
            ok = (
                "land_mask" in ds_cache
                and ds_cache.sizes.get("lat") == len(lat)
                and ds_cache.sizes.get("lon") == len(lon)
                and np.allclose(ds_cache.lat.values, lat)
                and np.allclose(ds_cache.lon.values, lon)
            )
            if ok:
                print("   直接读取陆地 mask 缓存:", cache)
                mask = ds_cache["land_mask"].astype(bool).load()
                ds_cache.close()
                return mask
            ds_cache.close()
        except Exception as e:
            print("   读取陆地 mask 缓存失败，重新计算:", repr(e))

    lon2d, lat2d = np.meshgrid(lon, lat)
    mask = None

    try:
        from shapely import contains_xy
        print("   使用 shapely.contains_xy 计算陆地 mask...")
        mask = contains_xy(land_union, lon2d, lat2d)
    except Exception:
        pass

    if mask is None:
        try:
            from shapely.vectorized import contains
            print("   使用 shapely.vectorized.contains 计算陆地 mask...")
            mask = contains(land_union, lon2d, lat2d)
        except Exception:
            pass

    if mask is None:
        print("   回退到逐格点陆地判断；第一次会较慢。")
        mask = np.zeros(lon2d.shape, dtype=bool)
        for i in range(lat2d.shape[0]):
            for j in range(lat2d.shape[1]):
                point = Point(float(lon2d[i, j]), float(lat2d[i, j]))
                if land_prepared.contains(point) or land_union.touches(point):
                    mask[i, j] = True

    mask = xr.DataArray(
        np.asarray(mask, dtype=bool),
        coords={"lat": da.lat, "lon": da.lon},
        dims=("lat", "lon"),
        name="land_mask",
    )

    try:
        os.makedirs(LAND_MASK_CACHE_DIR, exist_ok=True)
        mask.to_dataset().to_netcdf(cache)
        print("   已保存陆地 mask 缓存:", cache)
    except Exception as e:
        print("   保存陆地 mask 缓存失败，不影响绘图:", repr(e))

    return mask


def draw_panel(
    ax, temp_anom, name, panel, land_union, land_prepared,
    station_lon, station_lat, station_value,
):
    """四个面板共用Fig1b样式，并叠加GHCN-M站点异常和莫斯科位置。"""

    # 地图范围外多保留5°，与原代码一致。
    temp = temp_anom.sel(
        lon=slice(MAP_LON_MIN - 5, MAP_LON_MAX + 5),
        lat=slice(MAP_LAT_MIN - 5, MAP_LAT_MAX + 5),
    )
    # 温度只保留陆地。
    temp = temp.where(land_mask_on_grid(temp, land_union, land_prepared))

    temp_lon, temp_lat = np.meshgrid(temp.lon.values, temp.lat.values)

    ax.set_extent(MAP_EXTENT, crs=DATA_CRS)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_edgecolor("black")
    ax.set_aspect("auto")
    clip_patch = ax.patch

    # SAT
    im = ax.contourf(
        temp_lon, temp_lat, np.asarray(temp),
        levels=TEMP_LEVELS,
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        extend="both",
        transform=DATA_CRS,
        zorder=10,
        antialiased=True,
        corner_mask=True,
    )
    for collection in getattr(im, "collections", []):
        collection.set_clip_path(clip_patch)
        collection.set_rasterized(True)

    # 50m海岸线
    coast = ax.coastlines(
        resolution="50m",
        color=COAST_COLOR,
        lw=COAST_LW,
        zorder=95,
    )
    try:
        coast.set_clip_path(clip_patch)
    except Exception:
        pass

    # 经纬度
    ax.set_xticks(XTICKS, crs=DATA_CRS)
    ax.set_yticks(YTICKS, crs=DATA_CRS)
    ax.set_xticklabels([
        f"{abs(int(lon))}°W" if lon < 0
        else f"{int(lon)}°E" if lon > 0
        else "0°"
        for lon in XTICKS
    ], fontsize=FS_SMALL, fontweight="normal")
    ax.set_yticklabels([
        "" if np.isclose(lat, 30) else f"{int(lat)}°N"
        for lat in YTICKS
    ], fontsize=FS_SMALL, fontweight="normal")
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        pad=TICK_PAD,
        labelsize=FS_SMALL,
    )
    ax.grid(False)

    # 玫红研究区框
    segments = [
        (np.linspace(BOX_LON1, BOX_LON2, 250), np.full(250, BOX_LAT1)),
        (np.full(160, BOX_LON2), np.linspace(BOX_LAT1, BOX_LAT2, 160)),
        (np.linspace(BOX_LON2, BOX_LON1, 250), np.full(250, BOX_LAT2)),
        (np.full(160, BOX_LON1), np.linspace(BOX_LAT2, BOX_LAT1, 160)),
    ]
    for xx, yy in segments:
        line, = ax.plot(
            xx, yy,
            transform=DATA_CRS,
            color=BOX_COLOR,
            lw=BOX_LW,
            alpha=BOX_OPACITY,
            zorder=155,
            path_effects=[
                pe.Stroke(linewidth=BOX_LW + 0.9, foreground="white"),
                pe.Normal(),
            ],
        )
        line.set_clip_path(clip_patch)

    # GHCN-M station anomalies: same marker style and color scale as Fig1b.
    valid_station = (
        np.isfinite(station_lon)
        & np.isfinite(station_lat)
        & np.isfinite(station_value)
    )
    print(
        f"{name} 面板实际绘制 GHCN-M 站点数量：{int(valid_station.sum())}",
        flush=True,
    )
    station_points = ax.scatter(
        station_lon[valid_station],
        station_lat[valid_station],
        c=station_value[valid_station],
        s=16,
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        edgecolors="black",
        linewidths=0.35,
        transform=DATA_CRS,
        zorder=175,
    )
    station_points.set_clip_path(clip_patch)

    # Moscow: black star, using the exact coordinates in Fig1b.
    moscow_star = ax.scatter(
        *MOSCOW,
        s=32,
        marker="*",
        c="black",
        edgecolors="black",
        linewidths=0.35,
        transform=DATA_CRS,
        zorder=190,
    )
    moscow_star.set_clip_path(clip_patch)

    # Legend follows Fig1b.
    legend = ax.legend(
        handles=[Line2D(
            [0], [0], marker="*", color="none",
            markerfacecolor="black", markeredgecolor="black",
            markersize=8, label="Moscow",
        )],
        loc="lower left",
        bbox_to_anchor=(0.055, 0.055),
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
        framealpha=0.90,
        facecolor="white",
        edgecolor="#333333",
        fontsize=FS_SMALL,
        handlelength=0.8,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.25,
    )
    legend.set_zorder(206)

    ax.text(
        0.000, 1.020, panel,
        transform=ax.transAxes,
        fontsize=FS_LARGE,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
        zorder=400,
    )
    ax.text(
        0.99, 1.020, name,
        transform=ax.transAxes,
        fontsize=FS_LARGE,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
        zorder=400,
    )

    return im


# ============================================================
# 3. Natural Earth 陆地
# ============================================================

print("2. 读取 Natural Earth 110m land mask...")
land_shp = shpreader.natural_earth(
    resolution=LAND_MASK_SCALE,
    category="physical",
    name="land",
)
land_union = unary_union(list(shpreader.Reader(land_shp).geometries()))
land_prepared = prep(land_union)


# ============================================================
# 4. 读取PDO和ENSO指数（不读取SLP）
# ============================================================

print("2. 读取PDO和ENSO指数...")
with xr.open_dataset(PATH_PDO) as ds:
    ds = ensure_time(ds)
    pdo_monthly = ds["pdo"].astype(float).load()

with xr.open_dataset(PATH_ENSO) as ds:
    ds = ensure_time(ds)
    enso_monthly = ds["value"].astype(float).load()

# ENSO先减去1971–2000逐月气候态；PDO保持原指数。
enso_base = enso_monthly.sel(time=slice("1971-01-01", "2000-12-31"))
if enso_base.sizes.get("time", 0) != 360:
    raise ValueError("ENSO的1971–2000基准期不完整，应包含360个月。")
enso_clim = enso_base.groupby("time.month").mean("time", skipna=True)
enso_monthly = enso_monthly.groupby("time.month") - enso_clim

pdo_yearly = seasonal_mean(pdo_monthly, TARGET_MONTHS)
enso_yearly = seasonal_mean(enso_monthly, TARGET_MONTHS)
print("   PDO和ENSO均按12月+次年1月合成为DJ指数。")


# ============================================================
# 5. GHCN-M 站点资料：沿用Fig1b的处理方法
# ============================================================

print("3. 读取GHCN-M 80站点资料，并同时回归去除PDO和ENSO...")
with xr.open_dataset(PATH_STATION) as ds_station:
    if "tavg" not in ds_station:
        raise ValueError(
            f"GHCN-M文件中未找到tavg变量，变量有: {list(ds_station.data_vars)}"
        )
    if "lat" not in ds_station or "lon" not in ds_station:
        raise ValueError("GHCN-M文件中未找到lat/lon站点坐标。")

    station_tavg = ds_station["tavg"].astype(float)
    station_tavg = station_tavg.where(station_tavg > -9000).load()

    # Match Fig1b: convert Kelvin to Celsius only when values indicate Kelvin.
    if float(station_tavg.mean(skipna=True)) > 100:
        station_tavg = station_tavg - 273.15

    station_dim = next(d for d in station_tavg.dims if d != "time")
    station_lat = np.asarray(ds_station["lat"], dtype=float).squeeze()
    station_lon = np.asarray(ds_station["lon"], dtype=float).squeeze()

if np.nanmax(station_lon) > 180:
    station_lon = (station_lon + 180) % 360 - 180

if station_lat.ndim != 1 or station_lon.ndim != 1:
    raise ValueError(
        f"GHCN-M站点经纬度应为一维数组，当前 lat={station_lat.shape}, lon={station_lon.shape}"
    )
if len(station_lat) != station_tavg.sizes[station_dim] or len(station_lon) != station_tavg.sizes[station_dim]:
    raise ValueError(
        "GHCN-M站点经纬度数量与tavg站点维度不一致："
        f"tavg={station_tavg.sizes[station_dim]}, "
        f"lat={len(station_lat)}, lon={len(station_lon)}"
    )

station_anom = regression_anomaly(
    station_tavg,
    TARGET_MONTHS,
    "GHCN-M stations",
    pdo_yearly,
    enso_yearly,
    require_complete_months=True,
)
station_value = np.asarray(station_anom, dtype=float)
print(
    "   GHCN-M有效站点数：",
    int(np.isfinite(station_value).sum()),
    "/",
    len(station_value),
)


# ============================================================
# 6. 四套温度资料
# ============================================================

datasets = []

# GISTEMP
print("5. 读取 GISTEMP...")
ds_gis = ensure_time(xr.open_dataset(PATH_GIS))
if "tempanomaly" not in ds_gis:
    raise ValueError(f"GISTEMP 文件中未找到 tempanomaly，变量有: {list(ds_gis.data_vars)}")
gis = prep_grid(ds_gis["tempanomaly"], "GISTEMP")
gis_anom = regression_anomaly(gis, TARGET_MONTHS, "GISTEMP", pdo_yearly, enso_yearly)
datasets.append((gis_anom, "GISTEMP", "(a)"))

# CRU
print("6. 读取 CRU...")
ds_cru = ensure_time(xr.open_dataset(PATH_CRU))
if "tmp" not in ds_cru:
    raise ValueError(f"CRU 文件中未找到 tmp，变量有: {list(ds_cru.data_vars)}")
cru = prep_grid(ds_cru["tmp"], "CRU")
cru_anom = regression_anomaly(cru, TARGET_MONTHS, "CRU", pdo_yearly, enso_yearly)
datasets.append((cru_anom, "CRU", "(b)"))

# Berkeley
print("7. 读取 Berkeley...")
ds_berkeley = xr.open_dataset(PATH_BERKELEY)

if "time" in ds_berkeley.coords:
    try:
        decimal_time = not np.issubdtype(ds_berkeley.time.dtype, np.datetime64)
    except Exception:
        decimal_time = True

    if decimal_time:
        values = np.asarray(ds_berkeley.time.values, dtype=float)
        years = np.floor(values).astype(int)
        months = np.floor((values - years) * 12.0 + 1.0001).astype(int)
        months = np.clip(months, 1, 12)
        dates = pd.to_datetime({
            "year": years,
            "month": months,
            "day": np.ones_like(months),
        })
        ds_berkeley = ds_berkeley.assign_coords(time=dates)

ds_berkeley = ensure_time(ds_berkeley)
if "temperature" not in ds_berkeley:
    raise ValueError(
        f"Berkeley 文件中未找到 temperature，变量有: {list(ds_berkeley.data_vars)}"
    )
berkeley = prep_grid(ds_berkeley["temperature"], "Berkeley")
berkeley_anom = regression_anomaly(
    berkeley, TARGET_MONTHS, "Berkeley", pdo_yearly, enso_yearly
)
datasets.append((berkeley_anom, "Berkeley", "(c)"))

# NANSEN
print("8. 读取 NANSEN...")
ds_nansen = xr.open_dataset(PATH_NANSEN)

if "time" not in ds_nansen.dims:
    raise ValueError("NANSEN 数据中没有 time 维度。")

if "time" not in ds_nansen.coords:
    ds_nansen = ds_nansen.assign_coords(
        time=pd.date_range(
            start=NANSEN_TIME_START,
            periods=ds_nansen.sizes["time"],
            freq="MS",
        )
    )
else:
    try:
        nansen_datetime = np.issubdtype(ds_nansen.time.dtype, np.datetime64)
    except Exception:
        nansen_datetime = False

    if not nansen_datetime and np.issubdtype(ds_nansen.time.values.dtype, np.number):
        ds_nansen = ds_nansen.assign_coords(
            time=pd.date_range(
                start=NANSEN_TIME_START,
                periods=ds_nansen.sizes["time"],
                freq="MS",
            )
        )

if "SAT" not in ds_nansen:
    raise ValueError(f"NANSEN 文件中未找到 SAT，变量有: {list(ds_nansen.data_vars)}")

nansen = prep_grid(ds_nansen["SAT"], "NANSEN")
nansen_anom = regression_anomaly(
    nansen, TARGET_MONTHS, "NANSEN", pdo_yearly, enso_yearly
)
datasets.append((nansen_anom, "NANSEN", "(d)"))


# ============================================================
# 7. 绘图
# ============================================================

print("8. 开始绘图...")
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=FIG_FACECOLOR)

axes = [
    fig.add_axes(pos, projection=PROJ)
    for pos in AX_POSITIONS
]
caxes = [
    fig.add_axes(pos)
    for pos in CBAR_POSITIONS
]

for ax, cax, (temp_anom, name, panel) in zip(
    axes, caxes, datasets
):
    im = draw_panel(
        ax,
        temp_anom,
        name,
        panel,
        land_union,
        land_prepared,
        station_lon,
        station_lat,
        station_value,
    )

    cb = fig.colorbar(
        im,
        cax=cax,
        orientation="vertical",
        extend="both",
        boundaries=TEMP_LEVELS,
        ticks=TEMP_LEVELS,
        spacing="uniform",
        drawedges=True,
    )
    cb.ax.set_yticklabels([
        f"{int(v)}".replace("-", "−")
        for v in TEMP_LEVELS
    ])
    cb.ax.tick_params(
        labelsize=FS_SMALL,
        length=2.0,
        width=0.5,
        pad=1.5,
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
        fontsize=FS_MEDIUM,
        fontweight="normal",
        labelpad=4.0,
    )


# ============================================================
# 8. 保存与关闭
# ============================================================

if SAVE_FIGURE:
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", facecolor=FIG_FACECOLOR)
    fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", facecolor=FIG_FACECOLOR)
    print("PNG已保存:", OUT_PNG)
    print("PDF已保存:", OUT_PDF)

if SHOW_FIGURE:
    plt.show()

for ds in [ds_gis, ds_cru, ds_berkeley, ds_nansen]:
    try:
        ds.close()
    except Exception:
        pass

print("完成。")
