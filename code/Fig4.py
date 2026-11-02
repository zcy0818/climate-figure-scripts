# -*- coding: utf-8 -*-
from pathlib import Path
import os
import matplotlib.pyplot as plt

from Fig4common import load_panels, build_figure

# ============================== Editable settings ==============================
OUTPUT_DIR = Path(__file__).resolve().parent / "Fig4_outputs"
FIGURE_SIZE = (183 / 25.4, 205 / 25.4)
FIGURE_DPI = 150
SAVE_DPI = 1200
SHOW_FIGURE = os.environ.get("FIG4_SHOW", "1").lower() not in {"0", "false", "no"}

SIGNIFICANCE_COLOR = "#4B1D6B"
REGION_BOX_COLOR = "#D81B60"
REGION_BOX_WIDTH = 3.0
WAF_MIN_MAGNITUDE = 0.15
COLORBAR_TICKS = [
    [-30, -20, -10, 10, 20, 30],
    [-3, -2, -1, 1, 2, 3],
    [-30, -20, -10, 10, 20, 30],
    [-1.2, -0.6, 0.6, 1.2],
    [-0.8, -0.4, 0.4, 0.8],
]
# ===============================================================================

PANELS = load_panels("fig4_main")
a, b, c, d, e = PANELS
for module in PANELS:
    module.SIG_COLOR = SIGNIFICANCE_COLOR
    module.FIG5_BOX_COLOR = REGION_BOX_COLOR
    module.FIG5_BOX_LW = REGION_BOX_WIDTH
a.WAF_MIN_MAGNITUDE = c.WAF_MIN_MAGNITUDE = WAF_MIN_MAGNITUDE
a.REGION_BOX = (120.0, 240.0, 60.0, 75.0)
d.REGION_BOXES = [(330.0, 30.0, 40.0, 75.0)]

print("Calculating Figure 4 panels ...", flush=True)
data = [module.prepare_data() for module in PANELS]
fig = build_figure(
    PANELS, *data,
    colorbar_ticks=COLORBAR_TICKS,
    figure_size=FIGURE_SIZE,
    dpi=FIGURE_DPI,
)

# 关闭图(a)–(d)的经纬度网格线，同时保留经纬度标签。
# Fig4common.setup_polar_map() 中的可见网格不是由 Gridliner 绘制，
# 而是用 ax.plot() 手动绘制，颜色为 #8A9299、zorder=300；
# 因此仅关闭 ax.grid() 或 Gridliner.xlines/ylines 并不能去掉这些线。
map_axes = [ax for ax in fig.axes if hasattr(ax, "projection")]
for ax in map_axes[:4]:
    ax.grid(False)

    # 隐藏 setup_polar_map() 手动绘制的纬线和经线。
    for line in ax.lines:
        color = line.get_color()
        if (
            line.get_zorder() == 300
            and isinstance(color, str)
            and color.lower() == "#8a9299"
        ):
            line.set_visible(False)

    # 关闭 Gridliner 的线条，但保留其经纬度标签。
    for artist in ax.get_children():
        if artist.__class__.__name__ == "Gridliner":
            artist.xlines = False
            artist.ylines = False
            for line in getattr(artist, "xline_artists", []):
                line.set_visible(False)
            for line in getattr(artist, "yline_artists", []):
                line.set_visible(False)

OUTPUT_DIR.mkdir(exist_ok=True)
fig.savefig(OUTPUT_DIR / "Fig4.pdf", facecolor="white")
fig.savefig(OUTPUT_DIR / "Fig4.png", dpi=SAVE_DPI, facecolor="white")

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)
