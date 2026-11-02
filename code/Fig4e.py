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
import cartopy.io.shapereader as shpreader
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import Point
from shapely.prepared import prep as shapely_prep
from shapely.ops import unary_union

from Fig4common import (
    fix_time, prep_field, seasonal_mean, detrend_field, signflip_test,
    extend_to_plot_edges, nice_reference, remove_labels_outside,
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

PATH_TS_HAD = r"E:\20century\data\sea_ice_exps_HadISST\2deg_TS_HadISST_"
PATH_TS_REC = r"E:\20century\data\sea_ice_exps_reconstructed\2deg_TS_reconstructed_"
PATH_MEAN_HAD = r"E:\20century\data\mean\HadISST_1deg_ensmean.nc"
PATH_MEAN_REC = r"E:\20century\data\mean\reconstructed_1deg_ensmean.nc"
LAND_MASK_CACHE = r"E:\20century\data\model_ts_land_mask_110m_cache.nc"

TARGET_YEARS = np.arange(1940,1943)
CLIM_YEARS = np.arange(1971,2001)
MONTHS = [12,1]
CROSSYEAR_MONTHS = [1,2,3]
DO_DETREND_ALL = False

MAP_EXTENT = [-180,180,30,90]
SECTOR_EXTENT = [-30,90,30,85]
SECTOR_LON_MIN,SECTOR_LON_MAX,SECTOR_LAT_MIN,SECTOR_LAT_MAX = SECTOR_EXTENT
DATA_CRS = ccrs.PlateCarree()
SECTOR_PROJ = ccrs.PlateCarree()
SECTOR_XTICKS = np.arange(-30,91,30)
SECTOR_YTICKS = np.arange(30,86,15)
# 矩形地图比例与参考图一致：地图宽2.55英寸、高1.944英寸。
SECTOR_MAP_ASPECT = (3.75 * 0.68) / (2.70 * 0.72)
PANEL_LABEL_X, PANEL_LABEL_Y = -0.035, PANEL_HEADING_Y
TIME_LABEL_X, TIME_LABEL_Y = 0.09, PANEL_HEADING_Y
SECTOR_LON_LABEL_LAT_OFFSET = 3.8
SECTOR_EDGE_LON_LABEL_EXTRA_OFFSET = 1.0
SECTOR_LAT_LABEL_LON_OFFSET = 3.8

TEMP_VMIN,TEMP_VMAX,TEMP_STEP = -1.0,1.0,0.2
SLP_VMIN,SLP_VMAX,SLP_STEP = -8.0,8.0,0.5
SLP_COLOR,SLP_LW = "darkgreen",1.0
TEMP_MC_ALPHA,TEMP_MC_MIN_N = 0.1,20
MC_N_RANDOM,MC_RANDOM_SEED,MC_CHUNK_POINTS = 10000,20260625,2500
SIG_COLOR,SIG_LW,SIG_ZORDER = "#4B1D6B",1.4,525

WIND_COLOR = "#666666"
WIND_EDGE_COLOR = "white"
WIND_EDGE_WIDTH = .35
WIND_WIDTH = .0070  # 与a、c的箭杆宽度一致
WIND_HEADWIDTH,WIND_HEADLENGTH,WIND_HEADAXISLENGTH = 5.5,6.5,5.8
WIND_MINSHAFT,WIND_MINLENGTH = 1.2,.16
WIND_STRIDE_LAT,WIND_STRIDE_LON = 8,10  # 纬度/经度抽样间隔：越大越稀
WIND_PLOT_LAT_MAX = 85.0
WIND_REFERENCE_PERCENTILE = 85.0
WIND_REFERENCE_ARROW_FRACTION = .11
WAF_ARROW_SHORTEN_FACTOR,WAF_DISPLAY_AMPLIFICATION = 7.0,6.0
WIND_KEY_X,WIND_KEY_Y = VECTOR_KEY_X,PANEL_HEADING_Y  # a/c/e共用横向位置，纵向贴齐标题

FIG5_BOX_COLOR,FIG5_BOX_LW = "#D81B60",3.0
REGION_BOX = (30.0,60.0,45.0,60.0)
COAST_COLOR,COAST_LW = "0.25",.45
CBAR_EDGE_COLOR,CBAR_EDGE_LW = "#333333",.5
CITY_POINTS = []

# Repeated low-level operations are kept in Fig4common.py.
prep = prep_field
seasonal = partial(seasonal_mean, months=MONTHS, crossyear_months=CROSSYEAR_MONTHS)
detrend = detrend_field
monte_carlo = partial(
    signflip_test, alpha=TEMP_MC_ALPHA, min_n=TEMP_MC_MIN_N,
    n_random=MC_N_RANDOM, seed=MC_RANDOM_SEED, chunk_points=MC_CHUNK_POINTS,
)


def make_cmap():
    # SAT anomaly: -1 to 1 °C
    # The interval -0.2–0.2 °C is exactly pure white
    levels=np.array([
        -1.0,-0.8,-0.6,-0.4,-0.2,
         0.2,0.4,0.6,0.8,1.0
    ])

    colors=[
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

    cmap=ListedColormap(colors,name="sat_center_white")
    return levels,cmap,BoundaryNorm(levels,cmap.N,clip=False)

TEMP_LEVELS, TEMP_CMAP, TEMP_NORM = make_cmap()
SLP_NEG = np.round(np.arange(SLP_VMIN, -1.0 + SLP_STEP * .5, SLP_STEP), 10)
SLP_POS = np.round(np.arange(1.0, SLP_VMAX + SLP_STEP * .5, SLP_STEP), 10)
SLP_LEVELS = np.concatenate([SLP_NEG, SLP_POS])

def land_mask(da):
    da=prep(da,[float(da.lon.min()),float(da.lon.max()),float(da.lat.min()),float(da.lat.max())])
    lon,lat=np.asarray(da.lon,float),np.asarray(da.lat,float)
    if os.path.exists(LAND_MASK_CACHE):
        try:
            ds=xr.open_dataset(LAND_MASK_CACHE)
            ok=("land_mask" in ds and ds.sizes.get("lat")==len(lat) and ds.sizes.get("lon")==len(lon)
                and np.allclose(ds.lat,lat) and np.allclose(ds.lon,lon))
            if ok:
                out=ds["land_mask"].astype(bool).load();ds.close();return out
            ds.close()
        except Exception:pass
    shp=shpreader.natural_earth(resolution="110m",category="physical",name="land")
    union=unary_union(list(shpreader.Reader(shp).geometries()))
    lon2d,lat2d=np.meshgrid(lon,lat)
    try:
        from shapely import contains_xy
        mask=contains_xy(union,lon2d,lat2d)
    except Exception:
        prepared=shapely_prep(union);mask=np.zeros(lon2d.shape,bool)
        for i in range(len(lat)):
            for j in range(len(lon)):
                p=Point(float(lon2d[i,j]),float(lat2d[i,j]))
                mask[i,j]=prepared.contains(p) or union.touches(p)
    out=xr.DataArray(mask,coords={"lat":da.lat,"lon":da.lon},dims=("lat","lon"),name="land_mask")
    try:out.to_dataset().to_netcdf(LAND_MASK_CACHE)
    except Exception:pass
    return out

def setup_sector(ax):
    """矩形经纬度地图：30°W–90°E，30–85°N。"""
    ax.set_extent(
        [SECTOR_LON_MIN, SECTOR_LON_MAX, SECTOR_LAT_MIN, SECTOR_LAT_MAX],
        crs=DATA_CRS,
    )
    ax.set_facecolor("white")
    ax.set_aspect("auto")  # 矩形地图比例由主Axes的宽高确定

    # PlateCarree 下直接使用标准矩形 Axes 边界，不再设置扇形裁剪路径。
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_edgecolor("black")

    return ax.patch, None

def draw_box_and_labels(ax,clip,path):
    """矩形地图的海岸线、经纬度刻度、城市和玫红色区域框。"""
    coast=ax.coastlines(
        resolution="50m",
        linewidth=COAST_LW,
        color=COAST_COLOR,
        zorder=95,
    )
    try:
        coast.set_clip_path(clip)
    except Exception:
        pass

    # 标准矩形经纬度坐标。
    ax.set_xticks(SECTOR_XTICKS, crs=DATA_CRS)
    ax.set_yticks(SECTOR_YTICKS, crs=DATA_CRS)

    lon_labels=[]
    for lon in SECTOR_XTICKS:
        if lon < 0:
            lon_labels.append(f"{abs(int(lon))}°W")
        elif lon > 0:
            lon_labels.append(f"{int(lon)}°E")
        else:
            lon_labels.append("0°")

    lat_labels=["" if np.isclose(lat,30) else f"{int(lat)}°N" for lat in SECTOR_YTICKS]

    ax.set_xticklabels(
        lon_labels,
        fontsize=GEO_TICK_SIZE,
        fontweight="normal",
    )
    ax.set_yticklabels(
        lat_labels,
        fontsize=GEO_TICK_SIZE,
        fontweight="normal",
    )
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2.4,
        width=0.6,
        pad=1.5,
    )

    # 与参考矩形地图一致，不绘制背景经纬网格。
    ax.grid(
        False
    )

    # 城市标志保持不变。
    for _,lon,lat,marker in CITY_POINTS:
        m=ax.scatter(
            lon,lat,
            s=28,
            marker=marker,
            facecolor="black",
            edgecolor="black",
            linewidths=.45,
            transform=DATA_CRS,
            zorder=370,
        )
        m.set_clip_path(clip)

    # EEWR box：玫红色加粗。
    lon1,lon2,lat1,lat2=REGION_BOX
    parts=[
        (np.linspace(lon1,lon2,360),np.full(360,lat1)),
        (np.full(180,lon2),np.linspace(lat1,lat2,180)),
        (np.linspace(lon2,lon1,360),np.full(360,lat2)),
        (np.full(180,lon1),np.linspace(lat2,lat1,180)),
    ]
    for lons,lats in parts:
        line,=ax.plot(
            lons,lats,
            transform=DATA_CRS,
            color=FIG5_BOX_COLOR,
            lw=FIG5_BOX_LW,
            zorder=565,
        )
        line.set_clip_path(clip)
        line.set_path_effects([pe.Stroke(linewidth=FIG5_BOX_LW+0.9,foreground="white"),pe.Normal()])

def prepare_data():
    print("计算图4(e)：Dec-Jan SAT + PSL + 850 hPa Wind",flush=True)

    # 1) SAT 20成员
    had_list,rec_list=[],[]
    for member in range(20):
        ds=fix_time(xr.open_dataset(f"{PATH_TS_HAD}{member:02d}.nc"))
        had_list.append((prep(ds["TS"],MAP_EXTENT)-np.float32(273.15)).astype("float32").load());ds.close()
        ds=fix_time(xr.open_dataset(f"{PATH_TS_REC}{member:02d}.nc"))
        rec_list.append((prep(ds["TS"],MAP_EXTENT)-np.float32(273.15)).astype("float32").load());ds.close()

    ens=xr.DataArray(np.arange(20),dims="ensemble",name="ensemble")
    had=xr.concat(had_list,dim=ens,join="exact");rec=xr.concat(rec_list,dim=ens,join="exact")
    had,rec=xr.align(had,rec,join="inner")

    hy,ry=seasonal(had.mean("ensemble")),seasonal(rec.mean("ensemble"))
    if DO_DETREND_ALL:hy,ry=detrend(hy),detrend(ry)
    hcl=hy.where(hy.time.isin(CLIM_YEARS),drop=True).mean("time")
    rcl=ry.where(ry.time.isin(CLIM_YEARS),drop=True).mean("time")
    years=np.intersect1d(np.intersect1d(hy.time,ry.time),TARGET_YEARS)
    ts_diff=((ry-rcl).sel(time=years).mean("time")-(hy-hcl).sel(time=years).mean("time")).astype("float32")

    hym,rym=seasonal(had),seasonal(rec)
    if DO_DETREND_ALL:hym,rym=detrend(hym),detrend(rym)
    hcl=hym.where(hym.time.isin(CLIM_YEARS),drop=True).mean("time")
    rcl=rym.where(rym.time.isin(CLIM_YEARS),drop=True).mean("time")
    years=np.intersect1d(np.intersect1d(hym.time,rym.time),TARGET_YEARS)
    member_diff=(rym-rcl).sel(time=years).mean("time")-(hym-hcl).sel(time=years).mean("time")
    ts_sig=monte_carlo(member_diff)

    # 2) PSL/U850/V850 集合平均
    dh=fix_time(xr.open_dataset(PATH_MEAN_HAD));dr=fix_time(xr.open_dataset(PATH_MEAN_REC))
    psl_scale=.01 if float(dh["PSL"].mean())>5000 else 1.0
    result={}
    for name,scale in [("PSL",psl_scale),("U850",1.0),("V850",1.0)]:
        h=prep(dh[name],MAP_EXTENT,2)*scale;r=prep(dr[name],MAP_EXTENT,2)*scale
        hy,ry=seasonal(h),seasonal(r)
        if DO_DETREND_ALL:hy,ry=detrend(hy),detrend(ry)
        hd=hy.where(hy.time.isin(TARGET_YEARS),drop=True).mean("time")-hy.where(hy.time.isin(CLIM_YEARS),drop=True).mean("time")
        rd=ry.where(ry.time.isin(TARGET_YEARS),drop=True).mean("time")-ry.where(ry.time.isin(CLIM_YEARS),drop=True).mean("time")
        result[name]=(rd-hd).astype("float32")
    dh.close();dr.close()
    return {"ts":ts_diff,"ts_sig":ts_sig,"psl":result["PSL"],"u850":result["U850"],"v850":result["V850"],"title":"Dec-Jan"}

def draw_panel(ax,data,panel="e",show_slp=True):
    qk = None
    ts=prep(data["ts"],SECTOR_EXTENT)
    psl=prep(data["psl"],SECTOR_EXTENT,2) if show_slp else None
    u=prep(data["u850"],SECTOR_EXTENT);v=prep(data["v850"],SECTOR_EXTENT)
    sig=prep(data["ts_sig"],SECTOR_EXTENT)
    land=land_mask(ts)
    ts=ts.where(land);sig=sig.where(land,False)

    # TS is stored at 2-degree grid-cell centres (for example 29W, 31N, 89E).
    # Duplicate only the outermost cell values to the exact map limits so the
    # filled contours reach the frame instead of leaving a white half-cell rim.
    ts=extend_to_plot_edges(ts,SECTOR_EXTENT)
    sig=extend_to_plot_edges(sig,SECTOR_EXTENT)

    clip,path=setup_sector(ax)
    lon2d,lat2d=np.meshgrid(ts.lon.values,ts.lat.values)
    image=ax.contourf(lon2d,lat2d,ts.values,levels=TEMP_LEVELS,cmap=TEMP_CMAP,norm=TEMP_NORM,
                      extend="both",transform=DATA_CRS,zorder=10,antialiased=True,corner_mask=True)
    for c in getattr(image,"collections",[]):c.set_clip_path(clip)

    if float(sig.astype(float).max())>=.5:
        slon,slat=np.meshgrid(sig.lon.values,sig.lat.values)
        cs=ax.contour(slon,slat,sig.astype(float).values,levels=[.5],colors=SIG_COLOR,linewidths=SIG_LW,
                      transform=DATA_CRS,zorder=SIG_ZORDER)
        for c in getattr(cs,"collections",[cs]):
            c.set_clip_path(clip)
            c.set_path_effects([pe.Stroke(linewidth=SIG_LW+0.9,foreground="white"),pe.Normal()])

    labels=[]
    if show_slp:
        plon,plat=np.meshgrid(psl.lon.values,psl.lat.values)
        for levs,style in [(SLP_NEG,"dashed"),(SLP_POS,"solid")]:
            cs=ax.contour(plon,plat,psl.values,levels=levs,colors=SLP_COLOR,linewidths=SLP_LW,
                          linestyles=style,transform=DATA_CRS,zorder=90)
            for c in getattr(cs,"collections",[]):c.set_clip_path(clip)
            lab=ax.clabel(cs,levels=levs,inline=False,inline_spacing=2,fmt="%g",fontsize=SMALL_SIZE,
                          colors="black",use_clabeltext=True)
            for t in lab:
                t.set_fontweight("normal");t.set_path_effects([pe.withStroke(linewidth=1.2,foreground="white"),pe.Normal()])
                t.set_clip_path(clip)
            labels.extend(lab)

    u,v=xr.align(u,v,join="inner")
    uw=u.sel(lat=slice(SECTOR_LAT_MIN,min(SECTOR_LAT_MAX,WIND_PLOT_LAT_MAX)))
    vw=v.sel(lat=uw.lat,lon=uw.lon)
    speed=np.hypot(uw,vw).values
    valid=speed[np.isfinite(speed)&(speed>0)]
    if valid.size:
        ref=nice_reference(np.nanpercentile(valid,WIND_REFERENCE_PERCENTILE))
        scale=ref/WIND_REFERENCE_ARROW_FRACTION*WAF_ARROW_SHORTEN_FACTOR/WAF_DISPLAY_AMPLIFICATION
        uu=uw.isel(lat=slice(None,None,WIND_STRIDE_LAT),lon=slice(None,None,WIND_STRIDE_LON))
        vv=vw.sel(lat=uu.lat,lon=uu.lon)
        qlon,qlat=np.meshgrid(uu.lon.values,uu.lat.values)
        # 参照a、c：显式屏蔽缺测点，灰色箭头、白色描边、相同箭头形状和缩放单位。
        # 这里是实际850 hPa风矢量，保留正负方向与真实大小，不应用WAF的东向筛选。
        x,y=np.asarray(uu,float),np.asarray(vv,float)
        keep=np.isfinite(x)&np.isfinite(y)
        x,y=np.where(keep,x,np.nan),np.where(keep,y,np.nan)
        q=ax.quiver(qlon,qlat,x,y,transform=DATA_CRS,color=WIND_COLOR,
                    angles="uv",scale_units="width",scale=scale,pivot="middle",width=WIND_WIDTH,
                    headwidth=WIND_HEADWIDTH,headlength=WIND_HEADLENGTH,
                    headaxislength=WIND_HEADAXISLENGTH,minshaft=WIND_MINSHAFT,
                    minlength=WIND_MINLENGTH,zorder=430)
        q.set_clip_path(clip)
        q.set_path_effects([pe.Stroke(linewidth=1.0, foreground="white"), pe.Normal()])
        qk=heading_quiverkey(ax,q,X=WIND_KEY_X,Y=WIND_KEY_Y,U=ref,label=rf"${ref:g}\ \mathrm{{m\ s^{{-1}}}}$",
                        labelpos="E",coordinates="axes",fontproperties={"size":SMALL_SIZE,"weight":"normal"},
                        color=WIND_COLOR,labelcolor="black")
        qk.text.set_fontweight("normal")
        qk.text.set_bbox({"facecolor":"white","edgecolor":"none","alpha":.92,"pad":1.5})

    draw_box_and_labels(ax,clip,path)
    ax.text(PANEL_LABEL_X,PANEL_LABEL_Y,f"({panel})",transform=ax.transAxes,ha="left",va="bottom",
            fontsize=LARGE_SIZE,fontweight="bold",clip_on=False,zorder=900)
    heading = ax.text(TIME_LABEL_X,TIME_LABEL_Y,data["title"],transform=ax.transAxes,ha="left",va="bottom",
            fontsize=TIME_LABEL_SIZE,fontweight="bold",clip_on=False,zorder=900)
    if qk is not None:
        qk.heading_artist = heading
    return image,labels

def add_colorbar(fig,ax,image):
    # 绝对定位 colorbar，避免因主图位置或比例变化产生额外空白。
    p=ax.get_position()
    cax=fig.add_axes([p.x0,p.y0-0.13,p.width,0.025])
    cb=fig.colorbar(ScalarMappable(norm=TEMP_NORM,cmap=TEMP_CMAP),cax=cax,orientation="horizontal",
                    extend="both",boundaries=TEMP_LEVELS,
                    ticks=np.arange(TEMP_VMIN,TEMP_VMAX+TEMP_STEP*.5,TEMP_STEP))
    ticks=np.asarray(cb.get_ticks());ticks=ticks[~np.isclose(ticks,0)]
    cb.set_ticks(ticks);cb.ax.set_xticklabels([
        (f"−{abs(float(v)):g}" if float(v) < 0 else f"{float(v):g}")
        for v in ticks
    ])
    cb.set_label("Surface Air Temperature (°C)",fontsize=CBAR_LABEL_SIZE,fontweight="normal",labelpad=2)
    cb.ax.tick_params(labelsize=CBAR_TICK_SIZE,length=2.2,width=.6,pad=1.5)
    cb.outline.set_linewidth(CBAR_EDGE_LW);cb.outline.set_edgecolor(CBAR_EDGE_COLOR)
    return cb


if __name__ == "__main__":
    data=prepare_data()
    fig=plt.figure(figsize=(3.75,3.60),dpi=150,facecolor="white")
    map_width=0.77
    map_height=map_width*fig.get_figwidth()/SECTOR_MAP_ASPECT/fig.get_figheight()
    ax=fig.add_axes([0.15,0.235,map_width,map_height],projection=SECTOR_PROJ)
    image,labels=draw_panel(ax,data)
    remove_labels_outside(ax,labels)
    add_colorbar(fig,ax,image)
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"图4_深紫玫红版_outputs")
    os.makedirs(out,exist_ok=True)
    fig.savefig(os.path.join(out,"Figure4_e_modified.pdf"),facecolor="white")
    fig.savefig(os.path.join(out,"Figure4_e_modified.png"),dpi=1200,facecolor="white")
    if os.environ.get("FIG4_SHOW","1").lower() not in {"0","false","no"}:
        plt.show()
    else:
        plt.close(fig)
