# -*- coding: utf-8 -*-
import os
import warnings
from functools import partial

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

from Fig4common import (
    fix_time, prep_field, seasonal_mean, detrend_field, member_file, select_level,
    signflip_test, calculate_waf as _calculate_waf, setup_polar_map, add_sig_contour,
    heading_quiverkey, VECTOR_KEY_X, PANEL_HEADING_Y,
)

warnings.filterwarnings("ignore")

# -------------------- 字号与基础样式 --------------------
SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE = 8.0, 9.0, 10.0
GEO_TICK_SIZE = SMALL_SIZE
CBAR_TICK_SIZE = SMALL_SIZE
TIME_LABEL_SIZE = LARGE_SIZE
CBAR_LABEL_SIZE = MEDIUM_SIZE
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "SimHei"],
    "font.weight": "normal",
    "font.size": SMALL_SIZE,
    "axes.labelsize": MEDIUM_SIZE,
    "axes.titlesize": LARGE_SIZE,
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "axes.linewidth": 0.6,
    "xtick.labelsize": SMALL_SIZE,
    "ytick.labelsize": SMALL_SIZE,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.unicode_minus": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
})

# -------------------- 数据与时段 --------------------
PATH_ZT = r"F:\model hgt T\ZTlevel"
PATH_UV = r"F:\model UV\UVlevel"
ZT_SUFFIX = ".cam.h0.extract05-fulldims_plev.nc"
UV_SUFFIX = ".cam.h0.extract04-fulldims_plev.nc"

MONTHS = [12, 1]
PERIOD_TITLE = "Dec-Jan"
TARGET_YEARS = np.arange(1940, 1943)
CLIM_YEARS = np.arange(1971, 2001)
CROSSYEAR_MONTHS = [1, 2, 3]

# 原代码中 Z500 始终去趋势；保持不变。
Z500_MC_ALPHA = 0.1
Z500_MC_MIN_N = 20
MC_N_RANDOM = 10000
MC_RANDOM_SEED = 20260625
MC_CHUNK_POINTS = 2500

# -------------------- 地图 --------------------
MAP_EXTENT = [-180, 180, 30, 90]
PROJ = ccrs.NorthPolarStereo(central_longitude=0)
DATA_CRS = ccrs.PlateCarree()
COAST_COLOR, COAST_LW = "0.25", 0.45
GRID_COLOR, GRID_ALPHA, GRID_LW = "#8A9299", 0.22, 0.30
POLAR_LAT_LABELS = ()
POLAR_LAT_LABEL_LONS = {}
LAT_LABEL_SIZE = LON_LABEL_SIZE = SMALL_SIZE

# -------------------- Z500 --------------------
Z500_VMIN, Z500_VMAX, Z500_STEP = -30.0, 30.0, 5.0
SIG_COLOR, SIG_LW, SIG_ZORDER = "#4B1D6B", 1.4, 525

# -------------------- WAF --------------------
WAF_PRESSURE_HPA = 500.0
WAF_MIN_BASIC_SPEED = 2.0
WAF_EARTH_RADIUS = 6371000.0
WAF_OMEGA = 7.2921159e-05
WAF_GRAVITY = 9.80665
WAF_PLOT_LAT_MIN, WAF_PLOT_LAT_MAX = 30.0, 75.0
WAF_4, WAF_6 = 4, 8  # 纬度/经度抽样间隔：越小越密
WAF_VECTOR_COLOR = "#666666"
WAF_VECTOR_WIDTH = 0.0070
WAF_VECTOR_HEADWIDTH = 5.5
WAF_VECTOR_HEADLENGTH = 6.5
WAF_VECTOR_HEADAXISLENGTH = 5.8
WAF_VECTOR_MINSHAFT = 1.2
WAF_VECTOR_MINLENGTH = 0.16
# 绘制模长不低于全部有效WAF第10百分位数、且东西向分量Wx>0的矢量；剔除向西通量。
WAF_MAG_PERCENTILE = 10.0
WAF_CAP_PERCENTILE = 95.0
WAF_REFERENCE_PERCENTILE = 85.0
WAF_REFERENCE_ARROW_FRACTION = 0.11
WAF_ARROW_SHORTEN_FACTOR = 7.0
WAF_DISPLAY_AMPLIFICATION = 6.0
WAF_KEY_X, WAF_KEY_Y = VECTOR_KEY_X, PANEL_HEADING_Y

# -------------------- 城市和图5区域框 --------------------
FIG5_BOX_COLOR, FIG5_BOX_LW = "#D81B60", 3.0
REGION_BOX = (300.0, 30.0, 65.0, 80.0)
CITY_POINTS = []
LABEL_CITIES = False

CBAR_EDGE_COLOR, CBAR_EDGE_LW = "#333333", 0.5

# Repeated low-level operations are kept in Fig4common.py.
prep = partial(prep_field, extent=MAP_EXTENT)
seasonal = partial(seasonal_mean, months=MONTHS, crossyear_months=CROSSYEAR_MONTHS)
detrend = detrend_field
member_path = member_file
level = select_level
monte_carlo = partial(
    signflip_test, alpha=Z500_MC_ALPHA, min_n=Z500_MC_MIN_N,
    n_random=MC_N_RANDOM, seed=MC_RANDOM_SEED, chunk_points=MC_CHUNK_POINTS,
)
calculate_waf = partial(
    _calculate_waf, prep=prep, pressure_hpa=WAF_PRESSURE_HPA,
    min_basic_speed=WAF_MIN_BASIC_SPEED, earth_radius=WAF_EARTH_RADIUS,
    omega=WAF_OMEGA, gravity=WAF_GRAVITY,
    plot_lat_min=WAF_PLOT_LAT_MIN, plot_lat_max=WAF_PLOT_LAT_MAX,
)
setup_map = partial(
    setup_polar_map, data_crs=DATA_CRS, extent=MAP_EXTENT,
    coast_color=COAST_COLOR, coast_lw=COAST_LW, grid_color=GRID_COLOR,
    grid_alpha=GRID_ALPHA, grid_lw=GRID_LW, polar_lat_labels=POLAR_LAT_LABELS,
    lon_label_size=LON_LABEL_SIZE, lat_label_size=LAT_LABEL_SIZE,
)
add_sig = partial(
    add_sig_contour, data_crs=DATA_CRS, color=SIG_COLOR,
    linewidth=SIG_LW, zorder=SIG_ZORDER,
)


def make_cmap():
    levels = np.round(np.arange(Z500_VMIN, Z500_VMAX + Z500_STEP*.5, Z500_STEP), 10)
    levels = levels[np.abs(levels) > Z500_STEP*.2]
    anchors = ["#4B3B78", "#7C72A8", "#CECBE0", "#FFFFFF", "#F2D19B", "#D58A32", "#9A4F12"]
    base = LinearSegmentedColormap.from_list("z500_base", anchors, N=256)
    colors = base(np.linspace(0, 1, len(levels)-1))
    centers = (levels[:-1] + levels[1:]) / 2
    colors[np.argmin(np.abs(centers)), :3] = mpl.colors.to_rgb("#FFFFFF")
    cmap = ListedColormap(colors, name="z500_center_white")
    return levels, cmap, BoundaryNorm(levels, cmap.N, clip=False)

Z500_LEVELS, Z500_CMAP, Z500_NORM = make_cmap()

def draw_box_and_cities(ax, clip):
    lon1, lon2, lat1, lat2 = REGION_BOX
    while lon2 <= lon1:
        lon2 += 360
    parts = [
        (np.linspace(lon1, lon2, 360), np.full(360, lat1)),
        (np.full(180, lon2), np.linspace(lat1, lat2, 180)),
        (np.linspace(lon2, lon1, 360), np.full(360, lat2)),
        (np.full(180, lon1), np.linspace(lat2, lat1, 180)),
    ]
    for lons, lats in parts:
        line, = ax.plot(lons, lats, transform=DATA_CRS, color=FIG5_BOX_COLOR,
                        lw=FIG5_BOX_LW, zorder=565)
        line.set_clip_path(clip)
        line.set_path_effects([pe.Stroke(linewidth=FIG5_BOX_LW+0.9,foreground="white"),pe.Normal()])

    for name, lon, lat, marker in CITY_POINTS:
        m = ax.scatter(lon, lat, s=28, marker=marker, facecolor="black", edgecolor="black",
                       linewidths=.45, transform=DATA_CRS, zorder=370)
        m.set_clip_path(clip)
        if LABEL_CITIES:
            dy = 2.2 if name == "Moscow" else -2.75
            txt = ax.text(lon, lat+dy, name, transform=DATA_CRS, ha="center",
                          va="bottom" if dy > 0 else "top", fontsize=SMALL_SIZE,
                          fontweight="bold", zorder=372)
            txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])
            txt.set_clip_path(clip)

def waf_style(*items):
    """以全部方向的WAF模长第10百分位数为绘图阈值，并统一箭头比例。"""
    arrays, magnitudes = [], []
    for item in items:
        wx = prep(item["waf_x"]).sel(lat=slice(WAF_PLOT_LAT_MIN, WAF_PLOT_LAT_MAX))
        wy = prep(item["waf_y"]).sel(lat=wx.lat, lon=wx.lon)
        x, y = np.asarray(wx, float), np.asarray(wy, float)
        finite = np.isfinite(x) & np.isfinite(y)
        mag = np.hypot(x, y)
        valid = finite & (mag > 0)
        if np.any(valid):
            magnitudes.append(mag[valid])
        arrays.append((x, y, finite, mag))

    if not magnitudes:
        raise ValueError("WAF没有可用于绘图的有限非零通量。")

    mag_min = float(np.nanpercentile(np.concatenate(magnitudes), WAF_MAG_PERCENTILE))
    kept = [mag[finite & (mag > 0) & (mag >= mag_min)]
            for x, y, finite, mag in arrays]
    kept = np.concatenate([values for values in kept if values.size])
    cap = float(np.nanpercentile(kept, WAF_CAP_PERCENTILE))
    raw_ref = float(np.nanpercentile(np.minimum(kept, cap), WAF_REFERENCE_PERCENTILE))
    exp = np.floor(np.log10(raw_ref))
    frac = raw_ref / 10**exp
    ref = (1 if frac <= 1 else 2 if frac <= 2 else 5 if frac <= 5 else 10) * 10**exp
    scale = ref / WAF_REFERENCE_ARROW_FRACTION * WAF_ARROW_SHORTEN_FACTOR / WAF_DISPLAY_AMPLIFICATION
    return mag_min, cap, ref, scale

def prepare_data():
    print(f"计算图4 (c)：{PERIOD_TITLE} Z500 + WAF", flush=True)

    # 1) HadISST 20成员 500 hPa 背景风
    umean = vmean = None
    for member in range(20):
        ds = fix_time(xr.open_dataset(member_path(PATH_UV, "HadISST", member, UV_SUFFIX)))
        u = prep(level(ds, ["U500","uwnd500","u500","U_500","U","u","UWND","uwnd","ua"], 500), margin=5).load()
        v = prep(level(ds, ["V500","vwnd500","v500","V_500","V","v","VWND","vwnd","va"], 500), margin=5).load()
        ds.close()
        if umean is None:
            umean, vmean = u.copy(deep=True), v.copy(deep=True)
        else:
            umean.data[...] += (u.data - umean.data) / np.float32(member+1)
            vmean.data[...] += (v.data - vmean.data) / np.float32(member+1)

    uy, vy = seasonal(umean), seasonal(vmean)
    raw = umean.where(umean.time.dt.month.isin(MONTHS), drop=True)
    mo = raw.time.dt.month.values.astype(int)
    sy = raw.time.dt.year.values.astype(int)
    sy[np.isin(mo, CROSSYEAR_MONTHS)] -= 1
    complete = [int(y) for y in np.unique(sy) if set(MONTHS).issubset(set(mo[sy == y]))]
    use = np.intersect1d(np.intersect1d(uy.time, vy.time), np.intersect1d(complete, CLIM_YEARS))
    basic_u = uy.sel(time=use).mean("time", skipna=True).astype("float32").load()
    basic_v = vy.sel(time=use).mean("time", skipna=True).astype("float32").load()

    # 2) Z500 20成员响应差 + Monte Carlo
    member_diffs, had_mean, rec_mean = [], None, None
    zvars = ["Z500","HGT500","zg500","z500","Z3_500","Z_500","HGT_500",
             "Z3","z3","Z","z","zg","ZG","HGT","hgt","geopotential_height","height","PHI","phi"]
    for member in range(20):
        ds = fix_time(xr.open_dataset(member_path(PATH_ZT, "HadISST", member, ZT_SUFFIX)))
        had = prep(level(ds, zvars, 500, height=True), margin=2).load(); ds.close()
        ds = fix_time(xr.open_dataset(member_path(PATH_ZT, "reconstructed", member, ZT_SUFFIX)))
        rec = prep(level(ds, zvars, 500, height=True), margin=2).load(); ds.close()
        had, rec = xr.align(had, rec, join="inner")

        if had_mean is None:
            had_mean, rec_mean = had.copy(deep=True), rec.copy(deep=True)
        else:
            had_mean.data[...] += (had.data - had_mean.data) / np.float32(member+1)
            rec_mean.data[...] += (rec.data - rec_mean.data) / np.float32(member+1)

        hy, ry = detrend(seasonal(had)), detrend(seasonal(rec))
        common = np.intersect1d(hy.time, ry.time)
        target, climate = np.intersect1d(common, TARGET_YEARS), np.intersect1d(common, CLIM_YEARS)
        hd = hy.sel(time=target).mean("time") - hy.sel(time=climate).mean("time")
        rd = ry.sel(time=target).mean("time") - ry.sel(time=climate).mean("time")
        member_diffs.append((rd-hd).astype("float32").load())

    hy, ry = detrend(seasonal(had_mean)), detrend(seasonal(rec_mean))
    common = np.intersect1d(hy.time, ry.time)
    target, climate = np.intersect1d(common, TARGET_YEARS), np.intersect1d(common, CLIM_YEARS)
    z500 = ((ry.sel(time=target).mean("time") - ry.sel(time=climate).mean("time"))
            - (hy.sel(time=target).mean("time") - hy.sel(time=climate).mean("time"))).astype("float32")
    ens = xr.DataArray(np.arange(20), dims="ensemble", name="ensemble")
    zsig = monte_carlo(xr.concat(member_diffs, dim=ens, join="exact"))
    wx, wy = calculate_waf(z500, basic_u, basic_v)
    return {"z500": z500, "z500_sig": zsig, "waf_x": wx, "waf_y": wy, "title": PERIOD_TITLE}

def draw_panel(ax, data, style=None, panel="c"):
    qk = None
    clip = setup_map(ax)

    values, lon = add_cyclic_point(data["z500"].values, coord=data["z500"].lon.values,
                                   axis=data["z500"].get_axis_num("lon"))
    lon2d, lat2d = np.meshgrid(lon, data["z500"].lat.values)
    image = ax.contourf(lon2d, lat2d, values, levels=Z500_LEVELS, cmap=Z500_CMAP,
                        norm=Z500_NORM, extend="both", transform=DATA_CRS, zorder=10,
                        antialiased=True, corner_mask=True)
    for coll in getattr(image, "collections", []):
        coll.set_clip_path(clip)
    add_sig(ax, data["z500_sig"], clip)

    mag_min, cap, ref, scale = waf_style(data) if style is None else style
    wx = prep(data["waf_x"]).sel(lat=slice(WAF_PLOT_LAT_MIN,WAF_PLOT_LAT_MAX))
    wy = prep(data["waf_y"]).sel(lat=wx.lat, lon=wx.lon)
    wx = wx.isel(lat=slice(None,None,WAF_4), lon=slice(None,None,WAF_6))
    wy = wy.sel(lat=wx.lat, lon=wx.lon)
    x, y = np.asarray(wx,float), np.asarray(wy,float)
    mag = np.hypot(x,y)
    # 保留大于第10百分位数且向东传播（Wx>0）的通量；向西矢量整体不画。
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (mag > 0) & (mag >= mag_min)
    factor = np.ones_like(mag)
    hi = keep & (mag > cap)
    factor[hi] = cap / mag[hi]
    x, y = np.where(keep,x*factor,np.nan), np.where(keep,y*factor,np.nan)

    if np.isfinite(x).any():
        qlon, qlat = np.meshgrid(wx.lon.values, wx.lat.values)
        q = ax.quiver(qlon, qlat, x, y, transform=DATA_CRS, color=WAF_VECTOR_COLOR,
                      angles="uv", scale_units="width", scale=scale, pivot="middle",
                      width=WAF_VECTOR_WIDTH, headwidth=WAF_VECTOR_HEADWIDTH,
                      headlength=WAF_VECTOR_HEADLENGTH, headaxislength=WAF_VECTOR_HEADAXISLENGTH,
                      minshaft=WAF_VECTOR_MINSHAFT, minlength=WAF_VECTOR_MINLENGTH, zorder=340)
        q.set_clip_path(clip)
        q.set_path_effects([pe.Stroke(linewidth=1.0, foreground="white"), pe.Normal()])
        qk = heading_quiverkey(ax, q, X=WAF_KEY_X, Y=WAF_KEY_Y, U=ref,
                         label=rf"${ref:g}\ \mathrm{{m^2\,s^{{-2}}}}$", labelpos="E",
                         coordinates="axes", fontproperties={"size":SMALL_SIZE,"weight":"normal"},
                         color=WAF_VECTOR_COLOR, labelcolor="black")
        qk.text.set_fontweight("normal")
        qk.text.set_bbox({"facecolor":"white","edgecolor":"none","alpha":.92,"pad":1.5})

    draw_box_and_cities(ax, clip)
    theta = np.linspace(0, 2*np.pi, 361)
    ax.plot(.5+.5*np.sin(theta), .5+.5*np.cos(theta), transform=ax.transAxes,
            color="black", lw=.65, zorder=380, clip_on=False)
    ax.text(-.035,PANEL_HEADING_Y,f"({panel})",transform=ax.transAxes,ha="left",va="bottom",
            fontsize=LARGE_SIZE,fontweight="bold",clip_on=False,zorder=900)
    heading = ax.text(.085,PANEL_HEADING_Y,data["title"],transform=ax.transAxes,ha="left",va="bottom",
            fontsize=LARGE_SIZE,fontweight="bold",clip_on=False,zorder=900)
    if qk is not None:
        qk.heading_artist = heading
    return image

def add_colorbar(fig, ax, image):
    p = ax.get_position()
    cbar_w=p.width*1.50
    cax=fig.add_axes([p.x0+(p.width-cbar_w)/2,p.y0-.055,cbar_w,.018])
    cb = fig.colorbar(ScalarMappable(norm=Z500_NORM,cmap=Z500_CMAP), cax=cax,
                      orientation="horizontal", extend="both", boundaries=Z500_LEVELS,
                      ticks=np.arange(Z500_VMIN,Z500_VMAX+Z500_STEP*.5,Z500_STEP))
    ticks = np.asarray(cb.get_ticks()); ticks = ticks[~np.isclose(ticks,0)]
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels([
        (f"−{abs(float(v)):g}" if float(v) < 0 else f"{float(v):g}")
        for v in ticks
    ])
    cb.set_label("500 hPa Geopotential Height (gpm)", fontsize=MEDIUM_SIZE,
                 fontweight="normal", labelpad=2)
    cb.ax.tick_params(labelsize=SMALL_SIZE,length=2.2,width=.6,pad=1.5)
    cb.outline.set_linewidth(CBAR_EDGE_LW); cb.outline.set_edgecolor(CBAR_EDGE_COLOR)
    return cb


if __name__ == "__main__":
    data = prepare_data()
    fig = plt.figure(figsize=(3.65,4.15), dpi=1500, facecolor="white")
    ax = fig.add_subplot(111, projection=PROJ)
    fig.subplots_adjust(left=.06,right=.94,top=.92,bottom=.20)
    image = draw_panel(ax, data)
    add_colorbar(fig, ax, image)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "图4_深紫玫红版_outputs")
    os.makedirs(out, exist_ok=True)
    stem = "Figure4_c_DecJan_Z500_WAF"
    fig.savefig(os.path.join(out, stem+".png"), dpi=1500, facecolor="white")
    fig.savefig(os.path.join(out, stem+".pdf"), facecolor="white")
    if os.environ.get("FIG4_SHOW","1").lower() not in {"0","false","no"}:
        plt.show()
    else:
        plt.close(fig)
