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
    signflip_test, setup_polar_map, add_sig_contour, draw_jet_axis, month_offsets,
)

warnings.filterwarnings("ignore")

# ============================================================
# 用户参数区 —— 通常只需要修改这里
# ============================================================

# 要分析的气压层，单位 hPa。
# 例如：850、500、300、200
PRESSURE_HPA = 300

# 要平均的月份，必须按照季节实际先后顺序填写。
# 非跨年示例：[10, 11]、[1, 2, 3]
# 跨年示例：  [12, 1]、[11, 12, 1, 2]
MONTHS = [10, 11]

# 目标时段和气候态
TARGET_YEARS = np.arange(1940, 1943)
CLIM_YEARS = np.arange(1971, 2001)

# 是否对季节平均序列去趋势
DO_DETREND_ALL = False

# 填色/colorbar范围；更换气压层后如有需要可在这里一起调整
U_VMIN, U_VMAX, U_STEP = -3.0, 3.0, 0.5


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

# -------------------- 数据路径 --------------------
PATH_UV = r"F:\model UV\UVlevel"
UV_SUFFIX = ".cam.h0.extract04-fulldims_plev.nc"
MAP_EXTENT = [-180, 180, 30, 90]
PROJ = ccrs.NorthPolarStereo(central_longitude=0)
DATA_CRS = ccrs.PlateCarree()
COAST_COLOR, COAST_LW = "0.25", 0.45
GRID_COLOR, GRID_ALPHA, GRID_LW = "#8A9299", 0.22, 0.30
POLAR_LAT_LABELS = ()
POLAR_LAT_LABEL_LONS = {}

U_MC_ALPHA, U_MC_MIN_N = 0.1, 20
MC_N_RANDOM, MC_RANDOM_SEED, MC_CHUNK_POINTS = 10000, 20260625, 2500
SIG_COLOR, SIG_LW, SIG_ZORDER = "#4B1D6B", 1.4, 525

JET_AXIS_LAT_MIN, JET_AXIS_LAT_MAX = 40.0, 70.0
JET_AXIS_LON_MIN, JET_AXIS_LON_MAX = -180.0, 30.0
JET_AXIS_CONTINUITY_PENALTY = 0.018
JET_AXIS_COLOR, JET_AXIS_LW, JET_AXIS_ARROW_SIZE = "#666666", 2.0, 10.5
JET_AXIS_SMOOTH_POINTS, JET_AXIS_ZORDER = 7, 540

FIG5_BOX_COLOR, FIG5_BOX_LW = "#D81B60", 3.0
# 两个玫红色框：
# 240–270°E 在当前 -180–180° 经度体系中等价于 -120～-90°
REGION_BOXES = [
    (-120.0, -90.0, 30.0, 45.0),  # 240–270°E, 30–45°N
    (-120.0, -90.0, 50.0, 65.0),  # 240–270°E, 50–65°N
]
CITY_POINTS = []
CBAR_EDGE_COLOR, CBAR_EDGE_LW = "#333333", 0.5


_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# Validate the editable settings once, then reuse them everywhere below.
if not np.isfinite(float(PRESSURE_HPA)) or float(PRESSURE_HPA) <= 0:
    raise ValueError("PRESSURE_HPA must be a positive number in hPa.")
MONTHS = [int(m) for m in MONTHS]
MONTH_OFFSETS = month_offsets(MONTHS)
PERIOD_TITLE = "-".join(_MONTH_ABBR[m] for m in MONTHS)

# Repeated low-level operations are kept in Fig4common.py.
prep = partial(prep_field, extent=MAP_EXTENT)
seasonal = partial(
    seasonal_mean, months=MONTHS, require_complete=True, offsets=MONTH_OFFSETS,
)
detrend = detrend_field
member_path = partial(member_file, PATH_UV, suffix=UV_SUFFIX)
level = select_level
monte_carlo = partial(
    signflip_test, alpha=U_MC_ALPHA, min_n=U_MC_MIN_N,
    n_random=MC_N_RANDOM, seed=MC_RANDOM_SEED, chunk_points=MC_CHUNK_POINTS,
)
setup_map = partial(
    setup_polar_map, data_crs=DATA_CRS, extent=MAP_EXTENT,
    coast_color=COAST_COLOR, coast_lw=COAST_LW, grid_color=GRID_COLOR,
    grid_alpha=GRID_ALPHA, grid_lw=GRID_LW, polar_lat_labels=POLAR_LAT_LABELS,
    lon_label_size=SMALL_SIZE, lat_label_size=SMALL_SIZE,
)
add_sig = partial(
    add_sig_contour, data_crs=DATA_CRS, color=SIG_COLOR,
    linewidth=SIG_LW, zorder=SIG_ZORDER,
)
plot_jet_axis = partial(
    draw_jet_axis, prep=prep, projection=PROJ, data_crs=DATA_CRS,
    lat_min=JET_AXIS_LAT_MIN, lat_max=JET_AXIS_LAT_MAX,
    lon_min=JET_AXIS_LON_MIN, lon_max=JET_AXIS_LON_MAX,
    continuity_penalty=JET_AXIS_CONTINUITY_PENALTY, color=JET_AXIS_COLOR,
    linewidth=JET_AXIS_LW, arrow_size=JET_AXIS_ARROW_SIZE,
    smooth_points=JET_AXIS_SMOOTH_POINTS, zorder=JET_AXIS_ZORDER,
)


def make_cmap():
    levels=np.round(np.arange(U_VMIN,U_VMAX+U_STEP*.5,U_STEP),10)
    levels=levels[np.abs(levels)>U_STEP*.2]
    anchors=["#1D6B4F","#6AAA78","#D5E8D6","#FFFFFF","#E4D7E7","#A36AA1","#6B2E70"]
    base=LinearSegmentedColormap.from_list("u_pressure_base",anchors,N=256)
    colors=base(np.linspace(0,1,len(levels)-1))
    centers=(levels[:-1]+levels[1:])/2
    colors[np.argmin(np.abs(centers)),:3]=mpl.colors.to_rgb("#FFFFFF")
    cmap=ListedColormap(colors,name="u_pressure_center_white")
    return levels,cmap,BoundaryNorm(levels,cmap.N,clip=False)

U_LEVELS, U_CMAP, U_NORM = make_cmap()

def draw_box_and_cities(ax,clip):
    for lon1,lon2,lat1,lat2 in REGION_BOXES:
        parts=[(np.linspace(lon1,lon2,360),np.full(360,lat1)),
               (np.full(180,lon2),np.linspace(lat1,lat2,180)),
               (np.linspace(lon2,lon1,360),np.full(360,lat2)),
               (np.full(180,lon1),np.linspace(lat2,lat1,180))]
        for lons,lats in parts:
            line,=ax.plot(lons,lats,transform=DATA_CRS,color=FIG5_BOX_COLOR,lw=FIG5_BOX_LW,zorder=565)
            line.set_clip_path(clip)
            line.set_path_effects([pe.Stroke(linewidth=FIG5_BOX_LW+0.9,foreground="white"),pe.Normal()])

def prepare_data():
    months = MONTHS
    title = PERIOD_TITLE

    pressure_tag = (
        str(int(round(float(PRESSURE_HPA))))
        if np.isclose(float(PRESSURE_HPA), round(float(PRESSURE_HPA)))
        else f"{float(PRESSURE_HPA):g}"
    )

    print(
        f"计算图4(b)：{title} U{pressure_tag} + {PRESSURE_HPA:g} hPa 急流轴",
        flush=True
    )
    print(
        f"月份设置 MONTHS={months}；跨年偏移={MONTH_OFFSETS}",
        flush=True
    )

    u_diffs = []
    had_u_mean = rec_u_mean = had_v_mean = None

    # 优先寻找带气压层数字的变量名；若数据是 U/V + plev 形式，
    # 后面的通用变量名 U、V、ua、va 也可以正常工作。
    uvars = [
        f"U{pressure_tag}",
        f"uwnd{pressure_tag}",
        f"u{pressure_tag}",
        f"U_{pressure_tag}",
        "U", "u", "UWND", "uwnd", "ua"
    ]
    vvars = [
        f"V{pressure_tag}",
        f"vwnd{pressure_tag}",
        f"v{pressure_tag}",
        f"V_{pressure_tag}",
        "V", "v", "VWND", "vwnd", "va"
    ]

    for member in range(20):
        ds = fix_time(xr.open_dataset(member_path("HadISST", member)))
        had_u = prep(level(ds, uvars, PRESSURE_HPA), margin=2).load()
        had_v = prep(level(ds, vvars, PRESSURE_HPA), margin=2).load()
        ds.close()

        ds = fix_time(xr.open_dataset(member_path("reconstructed", member)))
        rec_u = prep(level(ds, uvars, PRESSURE_HPA), margin=2).load()
        ds.close()

        had_u, rec_u = xr.align(had_u, rec_u, join="inner")

        if had_u_mean is None:
            had_u_mean = had_u.copy(deep=True)
            rec_u_mean = rec_u.copy(deep=True)
            had_v_mean = had_v.copy(deep=True)
        else:
            had_u_mean.data[...] += (
                had_u.data - had_u_mean.data
            ) / np.float32(member + 1)

            rec_u_mean.data[...] += (
                rec_u.data - rec_u_mean.data
            ) / np.float32(member + 1)

            had_v_mean.data[...] += (
                had_v.data - had_v_mean.data
            ) / np.float32(member + 1)

        hy, ry = seasonal(had_u), seasonal(rec_u)

        if DO_DETREND_ALL:
            hy, ry = detrend(hy), detrend(ry)

        common = np.intersect1d(hy.time, ry.time)
        target = np.intersect1d(common, TARGET_YEARS)
        climate = np.intersect1d(common, CLIM_YEARS)

        if target.size == 0:
            raise ValueError(
                f"目标年份 {TARGET_YEARS[0]}–{TARGET_YEARS[-1]} "
                f"在 {title} 季节平均后没有有效数据。"
            )
        if climate.size == 0:
            raise ValueError(
                f"气候态年份 {CLIM_YEARS[0]}–{CLIM_YEARS[-1]} "
                f"在 {title} 季节平均后没有有效数据。"
            )

        hd = (
            hy.sel(time=target).mean("time")
            - hy.sel(time=climate).mean("time")
        )
        rd = (
            ry.sel(time=target).mean("time")
            - ry.sel(time=climate).mean("time")
        )

        u_diffs.append((rd - hd).astype("float32").load())

    # HadISST 气候态风用于画当前所选气压层的急流轴
    had_raw = seasonal(had_u_mean)
    rec_raw = seasonal(rec_u_mean)
    v_raw = seasonal(had_v_mean)

    climate = np.intersect1d(
        np.intersect1d(had_raw.time, rec_raw.time),
        CLIM_YEARS
    )

    uclim = (
        had_raw.sel(time=climate)
        .mean("time", skipna=True)
        .astype("float32")
    )
    vclim = (
        v_raw.sel(time=climate)
        .mean("time", skipna=True)
        .astype("float32")
    )

    if DO_DETREND_ALL:
        hy, ry = detrend(had_raw), detrend(rec_raw)
    else:
        hy, ry = had_raw, rec_raw

    common = np.intersect1d(hy.time, ry.time)
    target = np.intersect1d(common, TARGET_YEARS)
    climate = np.intersect1d(common, CLIM_YEARS)

    u_field = (
        (
            ry.sel(time=target).mean("time")
            - ry.sel(time=climate).mean("time")
        )
        -
        (
            hy.sel(time=target).mean("time")
            - hy.sel(time=climate).mean("time")
        )
    ).astype("float32")

    ens = xr.DataArray(
        np.arange(20),
        dims="ensemble",
        name="ensemble"
    )
    sig = monte_carlo(
        xr.concat(u_diffs, dim=ens, join="exact")
    )

    return {
        "u": u_field,
        "u_sig": sig,
        "u_clim": uclim,
        "v_clim": vclim,
        "title": title,
        "pressure_hpa": float(PRESSURE_HPA),
    }

def draw_panel(ax,data,panel="b"):
    clip=setup_map(ax)
    values,lon=add_cyclic_point(data["u"].values,coord=data["u"].lon.values,
                                axis=data["u"].get_axis_num("lon"))
    lon2d,lat2d=np.meshgrid(lon,data["u"].lat.values)
    image=ax.contourf(lon2d,lat2d,values,levels=U_LEVELS,cmap=U_CMAP,norm=U_NORM,
                      extend="both",transform=DATA_CRS,zorder=10,antialiased=True,corner_mask=True)
    for c in getattr(image,"collections",[]):c.set_clip_path(clip)
    add_sig(ax,data["u_sig"],clip)
    plot_jet_axis(ax,data["u_clim"],data["v_clim"],clip)
    draw_box_and_cities(ax,clip)
    theta=np.linspace(0,2*np.pi,361)
    ax.plot(.5+.5*np.sin(theta),.5+.5*np.cos(theta),transform=ax.transAxes,
            color="black",lw=.65,zorder=380,clip_on=False)
    ax.text(-.035,1.025,f"({panel})",transform=ax.transAxes,ha="left",va="bottom",
            fontsize=LARGE_SIZE,fontweight="bold",clip_on=False,zorder=900)
    ax.text(.085,1.025,data["title"],transform=ax.transAxes,ha="left",va="bottom",
            fontsize=LARGE_SIZE,fontweight="bold",clip_on=False,zorder=900)
    return image

def add_colorbar(fig,ax,image):
    p=ax.get_position(); cbar_w=p.width*1.50
    cax=fig.add_axes([p.x0+(p.width-cbar_w)/2,p.y0-.055,cbar_w,.018])
    cb=fig.colorbar(ScalarMappable(norm=U_NORM,cmap=U_CMAP),cax=cax,orientation="horizontal",
                    extend="both",boundaries=U_LEVELS,
                    ticks=np.arange(U_VMIN,U_VMAX+U_STEP*.5,U_STEP))
    ticks=np.asarray(cb.get_ticks());ticks=ticks[~np.isclose(ticks,0)]
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels([
        (f"−{abs(float(v)):g}" if float(v) < 0 else f"{float(v):g}")
        for v in ticks
    ])
    cb.set_label(f"{PRESSURE_HPA:g} hPa Zonal Wind (m s$^{-1}$)",fontsize=MEDIUM_SIZE,fontweight="normal",labelpad=2)
    cb.ax.tick_params(labelsize=SMALL_SIZE,length=2.2,width=.6,pad=1.5)
    cb.outline.set_linewidth(CBAR_EDGE_LW);cb.outline.set_edgecolor(CBAR_EDGE_COLOR)
    return cb


if __name__ == "__main__":
    data=prepare_data()
    fig=plt.figure(figsize=(3.65,4.15),dpi=1500,facecolor="white")
    ax=fig.add_subplot(111,projection=PROJ)
    fig.subplots_adjust(left=.06,right=.94,top=.92,bottom=.20)
    image=draw_panel(ax,data)
    add_colorbar(fig,ax,image)
    plt.show()
