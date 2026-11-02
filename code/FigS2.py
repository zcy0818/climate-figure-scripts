# -*- coding: utf-8 -*-
"""FigS2: Moscow monthly and daily temperature."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# 1. 可编辑参数
# ============================================================

MONTHLY_NC = r"E:\all station\Moscow_monthly_Tmean_1933_1947.nc"
DAILY_NC = r"E:\Moscow_daily_temperature_1936_1945.nc"
SAVE_PNG = r"E:\Moscow_temperature_two_panel_combined.png"
SAVE_PDF = r"E:\Moscow_temperature_two_panel_combined.pdf"

PLOT_VAR_MONTHLY = "tas"
PLOT_VAR_DAILY = "tmin"
REQUIRE_BOTH_MONTHS = True  # 每条季节平均线必须包含对应的两个月

DPI = 600
SHOW_FIGURE = True
FIG_SIZE = (7.50, 2.70)
AX1_POS = [0.08, 0.18, 0.40, 0.72]
AX2_POS = [0.56, 0.18, 0.40, 0.72]

FS_SMALL, FS_MEDIUM, FS_LARGE = 8, 9, 10
ZERO_COLOR = "#A8A8A8"
ZERO_LW = 0.65
SHADE_COLOR = "#D3D3D3"
SHADE_ALPHA = 0.65

SEASON_LABELS = {
    "ON": "Oct-Nov",
    "DJ": "Dec-Jan",
}
SEASON_COLORS = {
    "ON": "#0072B2",
    "DJ": "#D55E00",
}
DAILY_MONTH_COLORS = {12: "#707070", 1: "#D55E00"}
LINE_WIDTH = 1.40

MONTHLY_LABELS = {
    "tas": "Monthly Mean Temperature (°C)",
    "tmin_mean": "Monthly Mean Tmin (°C)",
    "tmax_mean": "Monthly Mean Tmax (°C)",
}
DAILY_LABELS = {
    "tmean": "Daily Mean Temperature (°C)",
    "tmin": "Daily Minimum Temperature (°C)",
    "tmax": "Daily Maximum Temperature (°C)",
}

LEFT_XLIM = (1935.7, 1945.3)
LEFT_XTICKS = np.arange(1936, 1946, 1)
LEFT_YLIM = (-22, 8)
LEFT_YTICKS = np.arange(-20, 6, 5)
SHADE_RANGE = (1940, 1942)
LEFT_LEGEND_ANCHOR = (0.006, 0.02)

RIGHT_XLIM = (1, 31)
RIGHT_XTICKS = np.arange(1, 32, 5)
RIGHT_Y_DEFAULT = (-25, 5)
RIGHT_Y_STEP = 5
RIGHT_Y_MARGIN = 1.5
RIGHT_LEGEND_ANCHOR = (0.46, 0.03)

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
# 2. 读取数据
# ============================================================

with xr.open_dataset(MONTHLY_NC) as ds:
    if PLOT_VAR_MONTHLY not in ds:
        raise ValueError(f"月尺度文件没有 {PLOT_VAR_MONTHLY}，现有变量：{list(ds.data_vars)}")
    da = ds[PLOT_VAR_MONTHLY].load()

time = pd.to_datetime(da.time.values)
monthly = pd.DataFrame({
    "year": time.year,
    "month": time.month,
    "value": np.asarray(da.values, float),
})
# 按自然年整理月平均气温。
# Oct-Nov：同一自然年的10月和11月平均。
# Dec-Jan：12月归属当年冬季，次年1月归入前一年12月所属的冬季。
monthly_table = monthly.pivot_table(
    index="year", columns="month", values="value", aggfunc="mean"
)

# Oct-Nov：当年10月与11月的平均值。
on_values = monthly_table.reindex(columns=[10, 11])
on_mean = on_values.mean(axis=1)
if REQUIRE_BOTH_MONTHS:
    on_mean = on_mean.where(on_values.notna().sum(axis=1) == 2)

# Dec-Jan：当年12月 + 次年1月，冬季年份标记为12月所在年份。
dj_values = pd.DataFrame({
    "Dec": monthly_table[12] if 12 in monthly_table.columns
           else pd.Series(index=monthly_table.index, dtype=float),
    "Jan_next": monthly_table[1].shift(-1) if 1 in monthly_table.columns
                else pd.Series(index=monthly_table.index, dtype=float),
}, index=monthly_table.index)

dj_mean = dj_values.mean(axis=1)
if REQUIRE_BOTH_MONTHS:
    dj_mean = dj_mean.where(dj_values.notna().sum(axis=1) == 2)

seasonal = pd.DataFrame({
    "ON": on_mean,
    "DJ": dj_mean,
})

with xr.open_dataset(DAILY_NC) as ds:
    if PLOT_VAR_DAILY not in ds:
        raise ValueError(f"日尺度文件没有 {PLOT_VAR_DAILY}，现有变量：{list(ds.data_vars)}")
    da = ds[PLOT_VAR_DAILY].load()
    time = pd.to_datetime(ds.time.values)

daily = pd.DataFrame({
    "year": time.year,
    "month": time.month,
    "day": time.day,
    "value": np.asarray(da.values, float),
})
dec = daily[(daily.year == 1941) & (daily.month == 12)].sort_values("day")
jan = daily[(daily.year == 1942) & (daily.month == 1)].sort_values("day")


# ============================================================
# 3. 绘图
# ============================================================

fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor="white")
ax1 = fig.add_axes(AX1_POS)
ax2 = fig.add_axes(AX2_POS)

# ---------- (a) 月平均 ----------
ax1.axvspan(*SHADE_RANGE, color=SHADE_COLOR, alpha=SHADE_ALPHA, zorder=0)
ax1.axhline(0, color=ZERO_COLOR, lw=ZERO_LW, zorder=1)

for season in ("ON", "DJ"):
    sub = seasonal[season].dropna()
    ax1.plot(
        sub.index, sub.values,
        color=SEASON_COLORS[season], lw=LINE_WIDTH, ls="-",
        label=SEASON_LABELS[season],
        zorder=5 if season == "ON" else 6,
    )

ax1.set(
    xlim=LEFT_XLIM, xticks=LEFT_XTICKS,
    ylim=LEFT_YLIM, yticks=LEFT_YTICKS,
    xlabel="Year",
    ylabel=MONTHLY_LABELS.get(PLOT_VAR_MONTHLY, PLOT_VAR_MONTHLY),
)
leg1 = ax1.legend(
    loc="lower left", bbox_to_anchor=LEFT_LEGEND_ANCHOR, ncol=1,
    frameon=False, fontsize=FS_SMALL, handlelength=2.3,
    handletextpad=0.35, borderaxespad=0.0,
    labelspacing=0.30,
)
ax1.text(
    0, 1.02, "(a)", transform=ax1.transAxes,
    fontsize=FS_LARGE, fontweight="bold",
    ha="left", va="bottom", clip_on=False,
)

# ---------- (b) 逐日 ----------
ax2.axhline(0, color=ZERO_COLOR, lw=ZERO_LW, zorder=1)
ax2.plot(
    dec.day, dec.value, color=DAILY_MONTH_COLORS[12], lw=LINE_WIDTH,
    label="Dec 1941", zorder=5,
)
ax2.plot(
    jan.day, jan.value, color=DAILY_MONTH_COLORS[1], lw=LINE_WIDTH,
    label="Jan 1942", zorder=6,
)

values = np.r_[dec.value.values.astype(float), jan.value.values.astype(float)]
values = values[np.isfinite(values)]
if values.size:
    ymin = RIGHT_Y_STEP * np.floor((values.min() - RIGHT_Y_MARGIN) / RIGHT_Y_STEP)
    ymax = RIGHT_Y_STEP * np.ceil((values.max() + RIGHT_Y_MARGIN) / RIGHT_Y_STEP)
    if ymin == ymax:
        ymin, ymax = ymin - RIGHT_Y_STEP, ymax + RIGHT_Y_STEP
else:
    ymin, ymax = RIGHT_Y_DEFAULT

ax2.set(
    xlim=RIGHT_XLIM, xticks=RIGHT_XTICKS,
    ylim=(ymin, ymax), yticks=np.arange(ymin, ymax + 0.1, RIGHT_Y_STEP),
    xlabel="Day",
    ylabel=DAILY_LABELS.get(PLOT_VAR_DAILY, f"{PLOT_VAR_DAILY} (°C)"),
)
leg2 = ax2.legend(
    loc="lower right", bbox_to_anchor=RIGHT_LEGEND_ANCHOR,
    frameon=False, fontsize=FS_SMALL, handlelength=2.3,
    handletextpad=0.35, borderaxespad=0.0, labelspacing=0.25,
)
ax2.text(
    0, 1.02, "(b)", transform=ax2.transAxes,
    fontsize=FS_LARGE, fontweight="bold",
    ha="left", va="bottom", clip_on=False,
)

# ---------- 公共坐标轴样式 ----------
for ax in (ax1, ax2):
    ax.tick_params(
        axis="both", which="major", direction="out",
        length=2.4, width=0.6, pad=1.5, labelsize=FS_SMALL,
    )
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#333333")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")
        label.set_fontweight("normal")
        label.set_fontsize(FS_SMALL)

for legend in (leg1, leg2):
    for text in legend.get_texts():
        text.set_fontname("Arial")
        text.set_fontweight("normal")


# ============================================================
# 4. 保存
# ============================================================

plt.savefig(SAVE_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.savefig(SAVE_PDF, dpi=DPI, bbox_inches="tight", facecolor="white")
print(f"PNG已保存：{SAVE_PNG}")
print(f"PDF已保存：{SAVE_PDF}")

if SHOW_FIGURE:
    plt.show()
