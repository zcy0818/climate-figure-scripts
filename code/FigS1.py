# -*- coding: utf-8 -*-
"""FigS1: Europe historical background on 31 January 1942."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import cartopy.crs as ccrs

from shapely.geometry import box
from shapely.ops import unary_union, linemerge


# ============================================================
# 1. 可编辑参数
# ============================================================

# ---------- 1942-01-31 历史国界 ----------
SHP_PATH = Path(
    r"E:\EuropeanBorders_WWII_extracted"
    r"\EuropeanBorders_WWII"
    r"\January_31_1942.shp"
)

# ---------- 地图 ----------
MAP_EXTENT = [-25, 70, 35, 72]
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = MAP_EXTENT
PROJ = ccrs.PlateCarree()

# ---------- 研究区 BOX ----------
BOX = [30, 60, 45, 60]          # lon_min, lon_max, lat_min, lat_max
BOX_COLOR = "#D81B60"
BOX_LINEWIDTH = 3.0
BOX_OPACITY = 1.0
BOX_WHITE_OUTLINE = 0.9

# ---------- 历史区域样式 ----------
USSR_FACE_COLOR = "0.93"
AXIS_FACE_COLOR = "white"
BACKGROUND_EDGE_COLOR = "0.55"

LW_BACKGROUND = 0.30
LW_COUNTRY = 0.60
LW_MAJOR = 0.90
LW_FRONT = 2.6

HATCH_STYLE = "///"
HATCH_LINEWIDTH = 0.30

# ---------- 文字 ----------
FS_COUNTRY = 10
FS_GEO = 20

COUNTRY_LABELS = [
    (-18.3, 65.0, "Iceland"),
    (-8.2, 53.3, "Eire"),
    (-2.6, 54.4, "U.K."),
    (-8.4, 39.4, "Portugal"),
    (-3.5, 40.2, "Spain"),
    (8.6, 46.6, "Switzerland"),
    (10.8, 52.0, "Germany"),
    (15.4, 61.2, "Sweden"),
    (26.1, 63.5, "Finland"),
    (48.0, 59.0, "U.S.S.R."),
    (34.0, 39.0, "Turkey"),
    (22.5, 38.7, "Greece"),
    (2.2, 28.4, "Algeria"),
    (17.0, 22.4, "Libya"),
    (29.5, 22.2, "Egypt"),
    (47.5, 22.3, "Saudi Arabia"),
    (34.2, 68.5, "Kola Peninsula"),
]

# ---------- 经纬度刻度 ----------
XTICKS = np.arange(-20, 61, 20)
YTICKS = np.array([40, 50, 60, 70])
TICK_LENGTH = 2.4
TICK_WIDTH = 0.6
TICK_PAD = 1.5

# ---------- 画布 ----------
FIG_SIZE = (10.5, 8.0)
AX_POS = [0.015, 0.015, 0.97, 0.97]
DPI = 600

# ---------- 保存 ----------
SAVE_FIGURE = False
SHOW_FIGURE = True
SAVE_PNG = "FigS1.png"
SAVE_PDF = "FigS1.pdf"


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.linewidth": 0.6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": HATCH_LINEWIDTH,
})


# ============================================================
# 2. 读取并整理 1942-01-31 shapefile
# ============================================================

if not SHP_PATH.exists():
    raise FileNotFoundError(
        "\n没有找到1942年1月历史国界shp文件：\n"
        f"{SHP_PATH}\n\n请检查该路径。"
    )

print("\n" + "=" * 80)
print("本图使用的唯一历史 shp 文件：")
print(SHP_PATH)
print("=" * 80)

wwii = gpd.read_file(SHP_PATH)

print("shp字段：")
print(list(wwii.columns))
print(f"shp原始feature数量：{len(wwii)}")

if wwii.crs is None:
    raise ValueError(
        "January_31_1942.shp没有CRS信息。"
        "请确认同目录中存在正确的.prj文件。"
    )

wwii = wwii.to_crs(epsg=4326)

try:
    wwii["geometry"] = wwii.geometry.make_valid()
except Exception:
    wwii["geometry"] = wwii.geometry.buffer(0)

wwii = wwii[wwii.geometry.notna()].copy()
wwii = wwii[~wwii.geometry.is_empty].copy()

# Axis-controlled / USSR 分类，只用于填色、斜线和前线识别。
name = wwii["Name"].astype(str) if "Name" in wwii.columns else ""
status = (
    wwii["Foreign_Po"].fillna("").astype(str)
    if "Foreign_Po" in wwii.columns
    else ""
)

axis_mask = (
    (name == "Italy")
    | status.str.contains("Axis", case=False, na=False)
    | status.str.contains("occupied", case=False, na=False)
    | status.str.contains("Protectorate", case=False, na=False)
)
ussr_mask = name == "Soviet Union"

wwii["Group"] = "Background"
wwii.loc[axis_mask, "Group"] = "Axis-controlled"
wwii.loc[ussr_mask, "Group"] = "USSR"

# 裁剪到当前地图范围；不 dissolve，保留原始 feature 国界。
clipper = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

wwii_clip = wwii.copy()
wwii_clip["geometry"] = wwii_clip.geometry.intersection(clipper)
wwii_clip = wwii_clip[wwii_clip.geometry.notna()].copy()
wwii_clip = wwii_clip[~wwii_clip.geometry.is_empty].copy()

print(f"地图范围内可绘制feature数量：{len(wwii_clip)}")

axis_part = wwii[wwii["Group"] == "Axis-controlled"]
ussr_part = wwii[wwii["Group"] == "USSR"]

axis_geom = None if axis_part.empty else unary_union(list(axis_part.geometry))
ussr_geom = None if ussr_part.empty else unary_union(list(ussr_part.geometry))


# ============================================================
# 3. Axis-USSR 共同边界
# ============================================================

front_lines = []

if (
    axis_geom is not None
    and ussr_geom is not None
    and not axis_geom.is_empty
    and not ussr_geom.is_empty
):
    shared = axis_geom.boundary.intersection(ussr_geom.boundary)

    pieces = []
    if not shared.is_empty:
        if shared.geom_type == "LineString":
            pieces = [shared]
        elif shared.geom_type == "MultiLineString":
            pieces = list(shared.geoms)
        elif hasattr(shared, "geoms"):
            for geom in shared.geoms:
                if geom.geom_type == "LineString":
                    pieces.append(geom)
                elif geom.geom_type == "MultiLineString":
                    pieces.extend(list(geom.geoms))

    pieces = [geom for geom in pieces if geom.length > 0.8]

    if pieces:
        merged = linemerge(unary_union(pieces))

        if merged.geom_type == "LineString":
            front_lines = [merged]
        elif merged.geom_type == "MultiLineString":
            front_lines = [geom for geom in merged.geoms if geom.length > 0.8]
        elif hasattr(merged, "geoms"):
            front_lines = [
                geom for geom in merged.geoms
                if geom.geom_type == "LineString" and geom.length > 0.8
            ]


# ============================================================
# 4. 绘图
# ============================================================

fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor="white")
ax = fig.add_axes(AX_POS, projection=PROJ)
ax.set_facecolor("white")

# ---------- 历史底图 ----------
wwii_clip.plot(
    ax=ax,
    facecolor="white",
    edgecolor=BACKGROUND_EDGE_COLOR,
    linewidth=LW_BACKGROUND,
    zorder=1,
)

# ---------- USSR 浅灰填色 ----------
if ussr_geom is not None and not ussr_geom.is_empty:
    geom = ussr_geom.intersection(clipper)
    if not geom.is_empty:
        gpd.GeoDataFrame(
            {"geometry": [geom]}, crs="EPSG:4326"
        ).plot(
            ax=ax,
            facecolor=USSR_FACE_COLOR,
            edgecolor="black",
            linewidth=LW_MAJOR,
            zorder=3,
        )

# ---------- Axis-controlled 白底 + 黑色斜线 ----------
if axis_geom is not None and not axis_geom.is_empty:
    geom = axis_geom.intersection(clipper)

    if not geom.is_empty:
        axis_gdf = gpd.GeoDataFrame(
            {"geometry": [geom]}, crs="EPSG:4326"
        )

        axis_gdf.plot(
            ax=ax,
            facecolor=AXIS_FACE_COLOR,
            edgecolor="black",
            linewidth=LW_MAJOR,
            zorder=4,
        )

        axis_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=LW_MAJOR,
            hatch=HATCH_STYLE,
            zorder=5,
        )

# ---------- Axis-USSR 前线 ----------
for line in front_lines:
    clipped = line.intersection(clipper)

    if clipped.is_empty:
        continue

    if clipped.geom_type == "LineString":
        geoms = [clipped]
    elif clipped.geom_type == "MultiLineString":
        geoms = list(clipped.geoms)
    else:
        geoms = [
            geom for geom in getattr(clipped, "geoms", [])
            if geom.geom_type == "LineString"
        ]

    for geom in geoms:
        xs, ys = geom.xy
        ax.plot(
            xs, ys,
            transform=PROJ,
            color="black",
            linewidth=LW_FRONT,
            linestyle=(0, (1.2, 2.0)),
            solid_capstyle="round",
            zorder=12,
        )

# ---------- 重新叠加 shp 中所有历史国界 ----------
geometries = [
    geom for geom in wwii_clip.geometry
    if geom is not None and not geom.is_empty
]

ax.add_geometries(
    geometries,
    crs=PROJ,
    facecolor="none",
    edgecolor="black",
    linewidth=LW_COUNTRY,
    zorder=15,
)

# ---------- 玫红研究区 BOX ----------
box_segments = [
    (np.linspace(BOX[0], BOX[1], 250), np.full(250, BOX[2])),
    (np.full(160, BOX[1]), np.linspace(BOX[2], BOX[3], 160)),
    (np.linspace(BOX[1], BOX[0], 250), np.full(250, BOX[3])),
    (np.full(160, BOX[0]), np.linspace(BOX[3], BOX[2], 160)),
]

for x, y in box_segments:
    ax.plot(
        x, y,
        transform=PROJ,
        color=BOX_COLOR,
        linewidth=BOX_LINEWIDTH,
        alpha=BOX_OPACITY,
        zorder=30,
        path_effects=[
            pe.Stroke(
                linewidth=BOX_LINEWIDTH + BOX_WHITE_OUTLINE,
                foreground="white",
            ),
            pe.Normal(),
        ],
    )

# ---------- 国家 / 地区名称 ----------
for x, y, text in COUNTRY_LABELS:
    if LON_MIN <= x <= LON_MAX and LAT_MIN <= y <= LAT_MAX:
        ax.text(
            x, y, text,
            transform=PROJ,
            ha="center",
            va="center",
            fontsize=FS_COUNTRY,
            color="black",
            zorder=40,
            clip_on=True,
            path_effects=[
                pe.withStroke(linewidth=1.3, foreground="white")
            ],
        )

# ---------- 地图范围和比例 ----------
ax.set_extent(MAP_EXTENT, crs=PROJ)
ax.set_aspect("auto")

# ---------- 经纬度标签 ----------
ax.set_xticks(XTICKS, crs=PROJ)
ax.set_yticks(YTICKS, crs=PROJ)

xlabels = [
    f"{abs(int(lon))}°W" if lon < 0
    else f"{int(lon)}°E" if lon > 0
    else "0°"
    for lon in XTICKS
]
ylabels = [f"{int(lat)}°N" for lat in YTICKS]

ax.set_xticklabels(xlabels, fontsize=FS_GEO, fontweight="normal")
ax.set_yticklabels(ylabels, fontsize=FS_GEO, fontweight="normal")

ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    length=TICK_LENGTH,
    width=TICK_WIDTH,
    pad=TICK_PAD,
    labelsize=FS_GEO,
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.grid(False)

for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(0.6)
    ax.spines[side].set_color("#333333")


# ============================================================
# 5. 保存
# ============================================================

if SAVE_FIGURE:
    fig.savefig(SAVE_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(SAVE_PDF, dpi=DPI, bbox_inches="tight", facecolor="white")
    print("PNG已保存：", SAVE_PNG)
    print("PDF已保存：", SAVE_PDF)

if SHOW_FIGURE:
    plt.show()
