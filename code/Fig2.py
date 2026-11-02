"""Figure 2: 图(a)(b)交换位置；图b无填充，图a/c/d无经纬网线，图c/d共用配色。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

BASE_DIR = Path(__file__).resolve().parent

PANEL_FILES = {
    "a": "Fig2a.py",
    "b": "Fig2b.py",
    "c": "Fig2c.py",
    "d": "Fig2d.py",
}

DPI = 600
FIG_SIZE = (6.83, 6.15)

OUTER_LEFT, OUTER_RIGHT = 0.075, 0.985
OUTER_TOP, OUTER_BOTTOM = 0.955, 0.075

OUTER_HSPACE = 0.23
PANEL_WSPACE = 0.22
PANEL_HSPACE = 0.12

PANEL_HEIGHT_RATIOS = [1.0, 0.075]

CBAR_THICKNESS_INCH = 0.12
CBAR_VERTICAL_FACTOR = 0.75

panels = {}

for key, filename in PANEL_FILES.items():

    path = BASE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    spec = spec_from_file_location(
        f"figure2_panel_{key}", path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"无法导入子图脚本: {path}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "draw_panel"):
        raise AttributeError(
            f"{path.name}没有draw_panel接口"
        )

    panels[key] = module

fig = plt.figure(
    figsize=FIG_SIZE,
    dpi=DPI,
    facecolor="white"
)

outer = fig.add_gridspec(
    2, 1,
    left=OUTER_LEFT,
    right=OUTER_RIGHT,
    top=OUTER_TOP,
    bottom=OUTER_BOTTOM,
    hspace=OUTER_HSPACE,
)

top = outer[0, 0].subgridspec(
    2, 2,
    width_ratios=[1.0, 1.0],
    height_ratios=PANEL_HEIGHT_RATIOS,
    wspace=PANEL_WSPACE,
    hspace=PANEL_HSPACE,
)

bottom = outer[1, 0].subgridspec(
    2, 2,
    width_ratios=[1.0, 1.0],
    height_ratios=PANEL_HEIGHT_RATIOS,
    wspace=PANEL_WSPACE,
    hspace=PANEL_HSPACE,
)

ax_b = fig.add_subplot(
    top[0, 0],
    projection=ccrs.NorthPolarStereo()
)

cax_b = fig.add_subplot(top[1, 0])

ax_a = fig.add_subplot(top[0, 1])

# 交换图(c)、图(d)的位置：图(d)放左下，图(c)放右下
ax_c = fig.add_subplot(
    bottom[0, 1],
    projection=ccrs.NorthPolarStereo()
)

cax_c = fig.add_subplot(bottom[1, 1])

ax_d = fig.add_subplot(
    bottom[0, 0],
    projection=ccrs.NorthPolarStereo()
)

cax_d = fig.add_subplot(bottom[1, 0])

panels["b"].draw_panel(ax_b, cax_b)

panels["a"].draw_panel(ax_a)

# 合并绘图时再次强制HadISST所有折线为浅灰色，防止旧版子图脚本覆盖颜色。
HADISST_COLOR = "#BDBDBD"
for line in ax_a.lines:
    if line.get_gid() in ("hadisst_raw", "hadisst_smooth") or line.get_label().strip() == "HadISST":
        line.set_color(HADISST_COLOR)
        line.set_alpha(0.48 if line.get_gid() == "hadisst_raw" else 1.0)

# 兼容未带gid的旧版Fig2a.py：该脚本按“重建数据原始线、HadISST原始线、重建数据平滑线、HadISST平滑线”绘制。
if len(ax_a.lines) >= 4:
    ax_a.lines[1].set_color(HADISST_COLOR)
    ax_a.lines[1].set_alpha(0.48)
    ax_a.lines[3].set_color(HADISST_COLOR)
    ax_a.lines[3].set_alpha(1.0)

# 图例是独立对象，需同步更新，否则图例颜色可能仍保留旧颜色。
legend_a = ax_a.get_legend()
if legend_a is not None:
    for legend_line, legend_text in zip(legend_a.get_lines(), legend_a.get_texts()):
        if legend_text.get_text().strip() == "HadISST":
            legend_line.set_color(HADISST_COLOR)
            legend_line.set_alpha(1.0)

panels["c"].draw_panel(ax_c, cax_c)

panels["d"].draw_panel(ax_d, cax_d)

# 按最终版面位置明确指定子图序号：
# 第一行：左 A（ax_b），右 B（ax_a）；
# 第二行：左 C（ax_d），右 D（ax_c）。
PANEL_LABEL_BY_AX = {
    ax_b: "A",
    ax_a: "B",
    ax_d: "C",
    ax_c: "D",
}

for ax, new_label in PANEL_LABEL_BY_AX.items():
    for txt in ax.texts:
        old_label = txt.get_text().strip()
        normalized_label = old_label.strip("()").strip().lower()
        if normalized_label in ("a", "b", "c", "d"):
            txt.set_text(new_label)

pos_b = ax_b.get_position()
pos_c = ax_c.get_position()
pos_d = ax_d.get_position()

common_w = min(
    pos_b.width,
    pos_c.width,
    pos_d.width
)

common_h = min(
    pos_b.height,
    pos_c.height,
    pos_d.height
)

for ax, pos in (
    (ax_b, pos_b),
    (ax_c, pos_c),
    (ax_d, pos_d)
):

    cx = pos.x0 + pos.width / 2.0
    cy = pos.y0 + pos.height / 2.0

    ax.set_position([
        cx - common_w / 2.0,
        cy - common_h / 2.0,
        common_w,
        common_h
    ])

cbar_h = (
    CBAR_THICKNESS_INCH
    / fig.get_figheight()
)

for ax, cax in (
    (ax_b, cax_b),
    (ax_c, cax_c),
    (ax_d, cax_d)
):

    mpos = ax.get_position()
    cpos = cax.get_position()

    cax.set_position([
        mpos.x0,
        cpos.y0 + (
            cpos.height - cbar_h
        ) * CBAR_VERTICAL_FACTOR,
        mpos.width,
        cbar_h
    ])

plt.show()
