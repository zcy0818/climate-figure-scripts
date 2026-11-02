
# -*- coding: utf-8 -*-
"""Fig1(a)：五套温度资料DJ时间序列，同时去除ENSO和PDO"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.geometry import Point
from shapely.prepared import prep


# ============================================================
# 1. 参数
# ============================================================

PATH_GIS = r"E:\20century\data\gistemp1200 temanomaly1951-1980_GHCNv4_.nc"
PATH_CRU = r"E:\20century\data\cru_ts4.05.1901.2020.tmp.dat.nc"
PATH_BERKELEY = r"E:\20century\data\berkeley.nc"
PATH_NANSEN = r"E:\20century\data\NANSENSAT1900.nc"
PATH_STA = r"E:\all station\GHCNM 80stations.nc"

PATH_PDO = r"E:\20century\data\pdo.timeseries.sstens.nc"
PATH_ENSO = r"E:\20century\data\nino34.long.anom.nc"

NANSEN_START_DATE = "1900-01-01"

TS_START_YEAR, TS_END_YEAR = 1900, 2020
CLIM_START, CLIM_END = 1971, 2000

TS_MONTHS = [12, 1]
CROSSYEAR_MONTHS = [1, 2, 3]

BOX_LAT_MIN, BOX_LAT_MAX = 45, 60
BOX_LON_MIN, BOX_LON_MAX = 30, 60

LAND_RESOLUTION = "50m"
LAND_SUBSAMPLE = 7
LAND_THRESHOLD = 0.5

XMIN, XMAX = 1899, 2021
XTICKS = np.arange(1900, 2021, 20)
YMIN, YMAX = -8.0, 7.1
YTICKS = np.arange(-6, 7.1, 2)

FIGSIZE = (3.75, 2.70)
AX_POS = [0.13, 0.18, 0.68, 0.72]
DPI = 600
SAVE_FIG = None

FS_SMALL = 8
FS_MEDIUM = 9
FS_LARGE = 10
LEGEND_FONTSIZE = 7

TICK_LENGTH = 2.4
TICK_WIDTH = 0.6
TICK_PAD = 1.5

ANNUAL_ALPHA = 0.55
ANNUAL_LW = 0.65
STATION_ANNUAL_LW = 0.60
SMOOTH_LW = 0.90
STATION_SMOOTH_LW = 0.85

RUNNING_WINDOW = 5
RUNNING_CENTER = True
RUNNING_MIN_PERIODS = 5

STATION_DASH = (0, (5.0, 2.4))

COLOR_GISTEMP = "#C9A227"
COLOR_CRU = "#0072B2"
COLOR_BERKELEY = "#E69F00"
COLOR_NANSEN = "#D55E00"
COLOR_GHCNM = "#707070"

ZERO_LINE_COLOR = "#A8A8A8"
SHADE_COLOR = "#D3D3D3"
SHADE_ALPHA = 0.65
SPINE_COLOR = "#333333"

SHADE_START, SHADE_END = 1940, 1942
MEAN_TEXT_X, MEAN_TEXT_Y = 1944.0, -6.35

Y_LABEL_TEXT = "Surface Air Temperature (°C)"
X_LABEL_TEXT = "Year"

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
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 2. DJ季节平均
# 1940年 = 1940年12月 + 1941年1月
# ============================================================

def seasonal_mean(da, start_year, end_year):

    da = da.sortby("time").load()

    empty = da.isel(time=0, drop=True).copy(deep=True).astype(float)
    empty.data = np.full(empty.shape, np.nan, dtype=float)

    result = []

    for year in range(start_year, end_year + 1):

        monthly = []

        for month in TS_MONTHS:

            real_year = (
                year + 1 if month in CROSSYEAR_MONTHS else year
            )

            one = da.where(
                (da.time.dt.year == real_year)
                & (da.time.dt.month == month),
                drop=True
            )

            if one.sizes.get("time", 0) == 0:
                monthly = []
                break

            monthly.append(
                one.mean("time", skipna=True)
            )

        if monthly:
            season = xr.concat(
                monthly, dim="month_tmp"
            ).mean("month_tmp", skipna=True)
        else:
            season = empty.copy(deep=True)

        result.append(season)

    years = xr.DataArray(
        np.arange(start_year, end_year + 1),
        dims="year",
        name="year"
    )

    return xr.concat(result, dim=years)


# ============================================================
# 3. 网格温度区域平均
# ============================================================

def grid_temperature_series(da, name, land_geom):

    rename = {}

    if "latitude" in da.coords:
        rename["latitude"] = "lat"

    if "longitude" in da.coords:
        rename["longitude"] = "lon"

    if rename:
        da = da.rename(rename)

    if np.nanmax(da.lon.values) > 180:
        da = da.assign_coords(
            lon=((da.lon + 180) % 360) - 180
        )

    da = da.sortby("lon")

    all_year = seasonal_mean(
        da, TS_START_YEAR, TS_END_YEAR
    )

    clim_year = seasonal_mean(
        da, CLIM_START, CLIM_END
    )

    anomaly = (
        all_year - clim_year.mean("year", skipna=True)
    )

    box = anomaly.sel(
        lon=slice(BOX_LON_MIN, BOX_LON_MAX)
    )

    if float(box.lat[0]) < float(box.lat[-1]):

        box = box.sel(
            lat=slice(BOX_LAT_MIN, BOX_LAT_MAX)
        )

    else:

        box = box.sel(
            lat=slice(BOX_LAT_MAX, BOX_LAT_MIN)
        )

    lat = np.asarray(box.lat, dtype=float)
    lon = np.asarray(box.lon, dtype=float)

    lat_edge = np.empty(len(lat) + 1)
    lon_edge = np.empty(len(lon) + 1)

    lat_edge[1:-1] = (lat[:-1] + lat[1:]) / 2
    lon_edge[1:-1] = (lon[:-1] + lon[1:]) / 2

    lat_edge[0] = lat[0] - (lat[1] - lat[0]) / 2
    lat_edge[-1] = lat[-1] + (lat[-1] - lat[-2]) / 2

    lon_edge[0] = lon[0] - (lon[1] - lon[0]) / 2
    lon_edge[-1] = lon[-1] + (lon[-1] - lon[-2]) / 2

    try:
        from shapely import contains_xy

        contains = lambda x, y: contains_xy(
            land_geom, x, y
        )

    except Exception:

        try:
            from shapely.vectorized import contains as vectorized_contains

            contains = lambda x, y: vectorized_contains(
                land_geom, x, y
            )

        except Exception:

            prepared = prep(land_geom)

            contains = np.vectorize(
                lambda x, y: prepared.contains(
                    Point(float(x), float(y))
                )
            )

    mask = np.zeros(
        (len(lat), len(lon)),
        dtype=bool
    )

    for i in range(len(lat)):

        ys = np.linspace(
            min(lat_edge[i:i + 2]),
            max(lat_edge[i:i + 2]),
            LAND_SUBSAMPLE + 2
        )[1:-1]

        for j in range(len(lon)):

            xs = np.linspace(
                min(lon_edge[j:j + 2]),
                max(lon_edge[j:j + 2]),
                LAND_SUBSAMPLE + 2
            )[1:-1]

            xx, yy = np.meshgrid(xs, ys)

            mask[i, j] = (
                np.asarray(contains(xx, yy)).mean()
                >= LAND_THRESHOLD
            )

    mask = xr.DataArray(
        mask,
        coords={
            "lat": box.lat,
            "lon": box.lon
        },
        dims=("lat", "lon")
    )

    weights = np.cos(np.deg2rad(box.lat))

    values = (
        box.where(mask)
        .weighted(weights)
        .mean(("lat", "lon"), skipna=True)
        .values.astype(float)
    )

    print(f"   {name} 完成")

    return anomaly.year.values.astype(int), values


# ============================================================
# 4. 读取五套温度资料
# ============================================================

print("1. 构建陆地几何...")

land_geom = unary_union(
    list(
        shpreader.Reader(
            shpreader.natural_earth(
                resolution=LAND_RESOLUTION,
                category="physical",
                name="land"
            )
        ).geometries()
    )
)


# ---------- GISTEMP ----------
print("2. 计算 GISTEMP...")

with xr.open_dataset(PATH_GIS) as ds:

    if not np.issubdtype(ds.time.dtype, np.datetime64):

        first = np.ravel(ds.time.values)[0]

        if not hasattr(first, "year"):
            ds = ds.assign_coords(
                time=("time", pd.to_datetime(ds.time.values))
            )

    gis_years, gistemp = grid_temperature_series(
        ds.tempanomaly.load(),
        "GISTEMP",
        land_geom
    )

years_plot = gis_years[
    (gis_years >= XMIN)
    & (gis_years <= XMAX)
]

gistemp = pd.Series(
    gistemp, index=gis_years
).reindex(years_plot).values.astype(float)


# ---------- CRU ----------
print("3. 计算 CRU...")

with xr.open_dataset(PATH_CRU) as ds:

    if not np.issubdtype(ds.time.dtype, np.datetime64):

        first = np.ravel(ds.time.values)[0]

        if not hasattr(first, "year"):
            ds = ds.assign_coords(
                time=("time", pd.to_datetime(ds.time.values))
            )

    cru_years, cru = grid_temperature_series(
        ds.tmp.load(),
        "CRU",
        land_geom
    )

cru = pd.Series(
    cru, index=cru_years
).reindex(years_plot).values.astype(float)


# ---------- Berkeley ----------
print("4. 计算 Berkeley...")

with xr.open_dataset(PATH_BERKELEY) as ds:

    if not np.issubdtype(ds.time.dtype, np.datetime64):

        t = np.asarray(ds.time.values, dtype=float)

        year = np.floor(t).astype(int)

        month = np.clip(
            np.floor((t - year) * 12 + 1).astype(int),
            1, 12
        )

        ds = ds.assign_coords(
            time=(
                "time",
                pd.to_datetime([
                    f"{y:04d}-{m:02d}-15"
                    for y, m in zip(year, month)
                ])
            )
        )

    berk_years, berkeley = grid_temperature_series(
        ds.temperature.load(),
        "Berkeley",
        land_geom
    )

berkeley = pd.Series(
    berkeley, index=berk_years
).reindex(years_plot).values.astype(float)


# ---------- Nansen ----------
print("5. 计算 Nansen SAT...")

with xr.open_dataset(PATH_NANSEN) as ds:

    if "time" not in ds.coords:

        ds = ds.assign_coords(
            time=(
                "time",
                pd.date_range(
                    NANSEN_START_DATE,
                    periods=ds.sizes["time"],
                    freq="MS"
                )
            )
        )

    elif not np.issubdtype(ds.time.dtype, np.datetime64):

        first = np.ravel(ds.time.values)[0]

        if not hasattr(first, "year"):
            ds = ds.assign_coords(
                time=("time", pd.to_datetime(ds.time.values))
            )

    nan_years, nansen = grid_temperature_series(
        ds.SAT.load(),
        "Nansen SAT",
        land_geom
    )

nansen = pd.Series(
    nansen, index=nan_years
).reindex(years_plot).values.astype(float)


# ---------- GHCNM station ----------
print("6. 计算 GHCNM 站点平均...")

with xr.open_dataset(PATH_STA) as ds:

    if not np.issubdtype(ds.time.dtype, np.datetime64):

        first = np.ravel(ds.time.values)[0]

        if not hasattr(first, "year"):
            ds = ds.assign_coords(
                time=("time", pd.to_datetime(ds.time.values))
            )

    tavg = (
        ds.tavg.astype(float)
        .load()
        .where(ds.tavg > -9000)
    )

    station_dim = next(
        d for d in tavg.dims if d != "time"
    )

    station_lat = np.asarray(ds.lat, dtype=float)
    station_lon = np.asarray(ds.lon, dtype=float)

    if np.nanmax(station_lon) > 180:
        station_lon = (
            (station_lon + 180) % 360
        ) - 180

    use_station = (
        np.isfinite(station_lat)
        & np.isfinite(station_lon)
        & (station_lat >= BOX_LAT_MIN)
        & (station_lat <= BOX_LAT_MAX)
        & (station_lon >= BOX_LON_MIN)
        & (station_lon <= BOX_LON_MAX)
    )

    tavg = tavg.where(
        xr.DataArray(
            use_station,
            coords={
                station_dim: tavg[station_dim]
            },
            dims=station_dim
        )
    )

    print(
        "研究区内站点数量:",
        int(use_station.sum())
    )

    if "is_complete_1961_1990" in ds:

        print(
            "1961–1990完整站点数:",
            int(np.nansum(
                ds.is_complete_1961_1990.values
            ))
        )

    sta_all = seasonal_mean(
        tavg, TS_START_YEAR, TS_END_YEAR
    )

    sta_clim = seasonal_mean(
        tavg, CLIM_START, CLIM_END
    ).mean("year", skipna=True)

    sta_anom = sta_all - sta_clim

    station = (
        sta_anom.sel(year=years_plot)
        .mean(station_dim, skipna=True)
        .values.astype(float)
    )

print("站点维度:", station_dim)

print(
    "用于绘图的站点数据年份:",
    int(years_plot.min()),
    "-",
    int(years_plot.max())
)


# ============================================================
# 5. 读取同期DJ PDO和ENSO
# ============================================================

print("7. 同时去除PDO和ENSO...")

with xr.open_dataset(PATH_PDO) as ds:
    pdo_monthly = ds["pdo"].astype(float).load()

with xr.open_dataset(PATH_ENSO) as ds:
    enso_monthly = ds["value"].astype(float).load()


for name, da in [
    ("pdo", pdo_monthly),
    ("enso", enso_monthly)
]:

    try:
        _ = da.time.dt.year

    except Exception:
        da = da.assign_coords(
            time=(
                "time",
                pd.to_datetime(da.time.values)
            )
        )

    if name == "pdo":
        pdo_monthly = da.sortby("time")
    else:
        enso_monthly = da.sortby("time")


# ---------- ENSO逐月距平 ----------
enso_base = enso_monthly.sel(
    time=slice("1971-01-01", "2000-12-31")
)

if enso_base.sizes.get("time", 0) != 360:
    raise ValueError(
        "ENSO 1971–2000基准期不完整，应有360个月"
    )

enso_clim = enso_base.groupby(
    "time.month"
).mean("time", skipna=True)

enso_monthly = (
    enso_monthly.groupby("time.month")
    - enso_clim
)


# ---------- DJ合成 ----------
def dj_index(da):

    da = da.where(
        da.time.dt.month.isin([12, 1]),
        drop=True
    )

    season_year = (
        da.time.dt.year
        - xr.where(
            da.time.dt.month == 1, 1, 0
        )
    )

    da = da.assign_coords(
        season_year=(
            "time",
            np.asarray(
                season_year.values,
                dtype=int
            )
        )
    )

    dj = da.groupby(
        "season_year"
    ).mean("time", skipna=True)

    count = (
        da.notnull()
        .groupby("season_year")
        .sum("time")
    )

    return dj.where(
        count == 2
    ).sortby("season_year")


pdo_dj = dj_index(pdo_monthly).to_series()
enso_dj = dj_index(enso_monthly).to_series()

pdo_raw = (
    pdo_dj.reindex(years_plot)
    .to_numpy(dtype=float)
)

enso_raw = (
    enso_dj.reindex(years_plot)
    .to_numpy(dtype=float)
)


# ============================================================
# 6. PDO和ENSO标准化
# ============================================================

common = (
    np.isfinite(pdo_raw)
    & np.isfinite(enso_raw)
)

if common.sum() < 30:
    raise ValueError(
        "PDO与ENSO共同有效年份不足30年"
    )

pdo_scale = np.nanstd(pdo_raw[common])
enso_scale = np.nanstd(enso_raw[common])

if pdo_scale == 0 or enso_scale == 0:
    raise ValueError("PDO或ENSO标准差为零")

pdo_std = (
    pdo_raw - np.nanmean(pdo_raw[common])
) / pdo_scale

enso_std = (
    enso_raw - np.nanmean(enso_raw[common])
) / enso_scale


# ============================================================
# 7. 双变量回归：同时去除PDO和ENSO
# ============================================================

def remove_pdo_enso(temp, label):

    good = (
        np.isfinite(temp)
        & np.isfinite(pdo_std)
        & np.isfinite(enso_std)
    )

    if good.sum() < 30:
        raise ValueError(
            f"{label}有效年份不足30年"
        )

    X = np.column_stack([
        np.ones(good.sum()),
        pdo_std[good],
        enso_std[good]
    ])

    if np.linalg.matrix_rank(X) < 3:
        raise ValueError(
            f"{label}回归矩阵秩不足"
        )

    beta = np.linalg.lstsq(
        X,
        temp[good],
        rcond=None
    )[0]

    corrected = np.full_like(
        temp,
        np.nan,
        dtype=float
    )

    corrected[good] = (
        temp[good]
        - beta[1] * pdo_std[good]
        - beta[2] * enso_std[good]
    )

    # 以1971–2000为基准重新计算距平
    ref = (
        good
        & (years_plot >= CLIM_START)
        & (years_plot <= CLIM_END)
    )

    if ref.sum() < 20:
        raise ValueError(
            f"{label}在1971–2000的有效年份不足20年"
        )

    corrected[good] -= np.nanmean(
        corrected[ref]
    )

    print(
        f"{label}: "
        f"有效年份={good.sum()}, "
        f"PDO系数={beta[1]:.3f}, "
        f"ENSO系数={beta[2]:.3f}"
    )

    return corrected


gistemp = remove_pdo_enso(
    gistemp, "GISTEMP"
)

cru = remove_pdo_enso(
    cru, "CRU"
)

berkeley = remove_pdo_enso(
    berkeley, "Berkeley"
)

nansen = remove_pdo_enso(
    nansen, "Nansen"
)

station = remove_pdo_enso(
    station, "GHCNM station"
)


# ============================================================
# 8. 五年中心滑动平均
# ============================================================

gistemp_s = pd.Series(gistemp).rolling(
    RUNNING_WINDOW,
    center=RUNNING_CENTER,
    min_periods=RUNNING_MIN_PERIODS
).mean().values

cru_s = pd.Series(cru).rolling(
    RUNNING_WINDOW,
    center=RUNNING_CENTER,
    min_periods=RUNNING_MIN_PERIODS
).mean().values

berkeley_s = pd.Series(berkeley).rolling(
    RUNNING_WINDOW,
    center=RUNNING_CENTER,
    min_periods=RUNNING_MIN_PERIODS
).mean().values

nansen_s = pd.Series(nansen).rolling(
    RUNNING_WINDOW,
    center=RUNNING_CENTER,
    min_periods=RUNNING_MIN_PERIODS
).mean().values

station_s = pd.Series(station).rolling(
    RUNNING_WINDOW,
    center=RUNNING_CENTER,
    min_periods=RUNNING_MIN_PERIODS
).mean().values


SERIES = [
    (
        "GISTEMP",
        gistemp, gistemp_s,
        COLOR_GISTEMP,
        ANNUAL_LW, "-", 20, 200
    ),
    (
        "Berkeley",
        berkeley, berkeley_s,
        COLOR_BERKELEY,
        ANNUAL_LW, "-", 21, 201
    ),
    (
        "Nansen",
        nansen, nansen_s,
        COLOR_NANSEN,
        ANNUAL_LW, "-", 22, 202
    ),
    (
        "CRU",
        cru, cru_s,
        COLOR_CRU,
        ANNUAL_LW, "-", 80, 800
    ),
    (
        "GHCNM station",
        station, station_s,
        COLOR_GHCNM,
        STATION_ANNUAL_LW,
        STATION_DASH, 100, 1000
    )
]


# ============================================================
# 9. 绘图函数
# 与Fig1.py的draw_panel接口保持一致
# ============================================================

def draw_panel(ax_a):

    ax_a.axvspan(
        SHADE_START,
        SHADE_END,
        color=SHADE_COLOR,
        alpha=SHADE_ALPHA,
        zorder=0
    )

    ax_a.axhline(
        0,
        color=ZERO_LINE_COLOR,
        lw=0.65,
        zorder=1
    )

    ax_a.set_xlim(XMIN, XMAX)
    ax_a.set_ylim(YMIN, YMAX)
    ax_a.set_xticks(XTICKS)
    ax_a.set_yticks(YTICKS)

    ax_a.set_xticklabels(
        [str(x) for x in XTICKS]
    )

    ax_a.set_xticks([], minor=True)

    ax_a.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        pad=TICK_PAD,
        labelsize=FS_SMALL
    )

    for side in (
        "left", "right", "bottom", "top"
    ):

        ax_a.spines[side].set_visible(True)
        ax_a.spines[side].set_linewidth(0.6)
        ax_a.spines[side].set_color(SPINE_COLOR)

    for label in (
        ax_a.get_xticklabels()
        + ax_a.get_yticklabels()
    ):

        label.set_fontname("Arial")
        label.set_fontweight("normal")
        label.set_fontsize(FS_SMALL)

    # ---------- 逐年细线 ----------
    for (
        name, raw, smooth, color,
        annual_lw, ls, z1, z2
    ) in SERIES:

        ax_a.plot(
            years_plot,
            raw,
            color=color,
            linewidth=annual_lw,
            linestyle=ls,
            alpha=ANNUAL_ALPHA,
            zorder=z1
        )

    # ---------- 五年滑动平均 ----------
    for (
        name, raw, smooth, color,
        annual_lw, ls, z1, z2
    ) in SERIES:

        lw = (
            STATION_SMOOTH_LW
            if name == "GHCNM station"
            else SMOOTH_LW
        )

        ax_a.plot(
            years_plot,
            smooth,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=1.0,
            label=name,
            zorder=z2
        )

    # ---------- 1940–1942平均温度 ----------
    stack = np.vstack([
        gistemp,
        cru,
        berkeley,
        nansen,
        station
    ])

    count = np.isfinite(stack).sum(axis=0)

    mean_all = np.divide(
        np.nansum(stack, axis=0),
        count,
        out=np.full(
            years_plot.size,
            np.nan
        ),
        where=count > 0
    )

    use = (
        (years_plot >= SHADE_START)
        & (years_plot <= SHADE_END)
        & np.isfinite(mean_all)
    )

    mean_4042 = float(
        np.nanmean(mean_all[use])
    )

    ax_a.text(
        MEAN_TEXT_X,
        MEAN_TEXT_Y,
        f"{SHADE_START}–{SHADE_END}: "
        f"{mean_4042:.0f}°C".replace("-", "−"),
        ha="left",
        va="center",
        fontsize=FS_SMALL,
        color="black",
        zorder=21,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.88,
            pad=0.25
        )
    )

    ax_a.set_ylabel(
        Y_LABEL_TEXT,
        fontsize=FS_MEDIUM,
        fontname="Arial",
        fontweight="normal"
    )

    ax_a.set_xlabel(
        X_LABEL_TEXT,
        fontsize=FS_MEDIUM,
        fontname="Arial",
        fontweight="normal"
    )

    ax_a.text(
        0.0, 1.02, "(a)",
        transform=ax_a.transAxes,
        fontsize=FS_LARGE,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
        zorder=400
    )

    # ---------- 图例 ----------
    legend_order = [
        (
            "GISTEMP",
            COLOR_GISTEMP,
            "-",
            SMOOTH_LW
        ),
        (
            "Berkeley",
            COLOR_BERKELEY,
            "-",
            SMOOTH_LW
        ),
        (
            "Nansen",
            COLOR_NANSEN,
            "-",
            SMOOTH_LW
        ),
        (
            "CRU",
            COLOR_CRU,
            "-",
            SMOOTH_LW
        ),
        (
            "GHCNM station",
            COLOR_GHCNM,
            STATION_DASH,
            STATION_SMOOTH_LW
        )
    ]

    handles = [
        Line2D(
            [0], [0],
            color=c,
            linewidth=lw,
            linestyle=ls,
            label=n
        )
        for n, c, ls, lw in legend_order
    ]

    legend = ax_a.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.006, 0.995),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.475,
        handletextpad=0.35,
        columnspacing=1.35,
        borderaxespad=0.0,
        labelspacing=0.22
    )

    for text in legend.get_texts():
        text.set_fontweight("normal")
        text.set_fontname("Arial")

    return ax_a


# ============================================================
# 10. 单独绘制Fig1a
# ============================================================

if __name__ == "__main__":

    fig = plt.figure(
        figsize=FIGSIZE,
        dpi=DPI,
        facecolor="white"
    )

    ax = fig.add_axes(AX_POS)

    draw_panel(ax)

    if SAVE_FIG:
        fig.savefig(
            SAVE_FIG,
            dpi=DPI,
            bbox_inches="tight"
        )

    plt.show()
