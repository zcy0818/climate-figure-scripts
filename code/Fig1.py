
# -*- coding: utf-8 -*-
"""图1(a,b)合并脚本：空间分布图在左，时间序列在右"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.text import Text
import cartopy.crs as ccrs


# ============================================================
# 1. 所有可调参数
# ============================================================

PANEL_A_FILE = "Fig1a.py"   # 原时间序列
PANEL_B_FILE = "Fig1b.py"   # 原空间分布

BASE_DIR = Path(__file__).resolve().parent

DPI = 600
FIGSIZE = (6.83, 3.25)

PANEL_BOTTOM = 0.19
PANEL_HEIGHT = 0.72

# 左图：空间分布
MAP_LEFT = 0.065
MAP_WIDTH = 0.35

# 右图：时间序列
TIME_LEFT = 0.605
TIME_WIDTH = 0.34

# 空间分布图右侧Colorbar
CAX_MAP_POS = [0.435, 0.255, 0.018, 0.59]

TICK_LENGTH = 2.4
TICK_WIDTH = 0.6
TICK_PAD = 1.5

SAVE_FIG = None


# ============================================================
# 2. 载入两个子图脚本
# ============================================================

spec_a = spec_from_file_location(
    "figure1_panel_a",
    BASE_DIR / PANEL_A_FILE
)

panel_a = module_from_spec(spec_a)
spec_a.loader.exec_module(panel_a)


spec_b = spec_from_file_location(
    "figure1_panel_b",
    BASE_DIR / PANEL_B_FILE
)

panel_b = module_from_spec(spec_b)
spec_b.loader.exec_module(panel_b)


# ============================================================
# 3. 创建画布并交换图ab位置
# ============================================================

fig = plt.figure(
    figsize=FIGSIZE,
    dpi=DPI,
    facecolor="white"
)

# 左图：原Fig1b（空间分布）
ax_map = fig.add_axes(
    [
        MAP_LEFT,
        PANEL_BOTTOM,
        MAP_WIDTH,
        PANEL_HEIGHT
    ],
    projection=getattr(
        panel_b,
        "PROJECTION",
        ccrs.PlateCarree()
    )
)

# 右图：原Fig1a（时间序列）
ax_time = fig.add_axes(
    [
        TIME_LEFT,
        PANEL_BOTTOM,
        TIME_WIDTH,
        PANEL_HEIGHT
    ]
)

# 地图Colorbar
cax_map = fig.add_axes(CAX_MAP_POS)


# ============================================================
# 4. 分别绘制
# ============================================================

panel_b.draw_panel(ax_map, cax_map)

panel_a.draw_panel(ax_time)


# ============================================================
# 5. 交换子图编号
# 原图(b)改为(a)，原图(a)改为(b)
# ============================================================

for txt in ax_map.texts:
    if txt.get_text() == "(b)":
        txt.set_text("(a)")

for txt in ax_time.texts:
    if txt.get_text() == "(a)":
        txt.set_text("(b)")


# ============================================================
# 6. 坐标轴样式
# ============================================================

for ax in (ax_map, ax_time):

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        pad=TICK_PAD
    )


# 纬度标签白色描边
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


# ============================================================
# 7. 保存与显示
# ============================================================

if SAVE_FIG:
    fig.savefig(
        SAVE_FIG,
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
