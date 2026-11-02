# -*- coding: utf-8 -*-
# Fig. 5: paired-member raincloud plots
# All parameters normally edited by the user are collected here.

from pathlib import Path
import re
import warnings
import cftime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ============================================================
# Parameters
# ============================================================
SMALL_FONT_SIZE = 18
MEDIUM_FONT_SIZE = 19
LARGE_FONT_SIZE = 20

EXPECTED_MEMBERS = 20
TARGET_YEARS = np.arange(1940, 1943)
BASELINE_YEARS = np.arange(1971, 2001)
DO_DETREND = False
DETREND_TAG = "detrended" if DO_DETREND else "raw"

# Oct-Nov Arctic warming: 500-1000 hPa mean temperature
WARM_LON1, WARM_LON2 = 120.0, 240.0
WARM_LAT1, WARM_LAT2 = 60.0, 75.0
PRESSURE_TOP, PRESSURE_BOTTOM = 500.0, 1000.0

# Oct-Nov U300 north-minus-south zonal-wind difference
JET300_LON1, JET300_LON2 = 240.0, 270.0
JET300_SOUTH_LAT1, JET300_SOUTH_LAT2 = 30.0, 45.0
JET300_NORTH_LAT1, JET300_NORTH_LAT2 = 50.0, 65.0
JET300_LEVEL_HPA = 300.0
JET300_MONTHS = [10, 11]

# Dec-Jan U850 regional zonal wind
JET850_LON1, JET850_LON2 = 330.0, 30.0
JET850_LAT1, JET850_LAT2 = 40.0, 75.0
JET850_LEVEL_HPA = 850.0
JET850_MONTHS = [12, 1]

# Dec-Jan Z500 blocking index
BLOCK_LON1, BLOCK_LON2 = 300.0, 30.0
BLOCK_LAT1, BLOCK_LAT2 = 65.0, 80.0

# Dec-Jan Eastern Europe-western Russia SAT
EEWR_LON1, EEWR_LON2 = 30.0, 60.0
EEWR_LAT1, EEWR_LAT2 = 45.0, 60.0

SD_MULTIPLIER = 0.5

PATH_ZT_PLEV_DIR = Path(r"F:\model hgt T\ZTlevel")
ZT_FILE_SUFFIX = ".cam.h0.extract05-fulldims_plev.nc"
PATH_UV_PLEV_DIR = Path(r"F:\model UV\UVlevel")
UV_FILE_SUFFIX = ".cam.h0.extract04-fulldims_plev.nc"
PATH_TS_HAD_BASE = Path(r"E:\20century\data\sea_ice_exps_HadISST")
PATH_TS_REC_BASE = Path(r"E:\20century\data\sea_ice_exps_reconstructed")

ALLOW_VERIFIED_EEWR_FALLBACK = True
VERIFIED_EEWR_FALLBACK = np.array([
    0.133, 1.331, -1.222, -1.330, -0.355,
    -2.742, -1.766, 0.568, -0.068, 1.404,
    0.367, -1.191, 1.093, -2.453, -1.647,
    -0.745, -0.312, 2.159, -2.318, -0.742,
], dtype=float)

# Figure / colors / scatter
DISTRIBUTION_FIGURE_SIZE = (28.0, 17.0)
FIGURE_DPI = 150
SAVE_FIGURE = True
SHOW_FIGURE = True
OUTPUT_STEM = Path(__file__).resolve().parent / "outputs" / "Fig5_6panels"

TEMP_POSITIVE_COLOR = "#CB4942"
TEMP_NEGATIVE_COLOR = "#3885BC"
Z500_POSITIVE_COLOR = "#EA9530"
U300_NEGATIVE_COLOR = "#75BE77"
U850_NEGATIVE_COLOR = "#B45AA3"
UNSATISFIED_COLOR = "#9A9A9A"
SPECIAL_TRACKED_MEMBERS = {2, 10, 13, 18}
PDF_FILL_LIGHTEN_AMOUNT = 0.72

MEMBER_SCATTER_SIZE = 800
GRAY_MARKER_SIZE = 1000
MEMBER_NUMBER_FONT_SIZE = 21.0
GRAY_MARKER_NUMBER_FONT_SIZE = 23.0
MEMBER_NUMBER_STROKE_WIDTH = 2.20
MEMBER_SWARM_X_LEFT = -27.6
MEMBER_SWARM_X_RIGHT = -4.4
MEMBER_SWARM_MARGIN = 1.00

Y_AXIS_SPINE_LINEWIDTH = 2.4
Y_AXIS_TICK_WIDTH = 2.0
Y_AXIS_TICK_LABEL_WEIGHT = "normal"
Y_AXIS_LABEL_WEIGHT = "normal"
AXIS_TICK_FONT_SIZE = 25
AXIS_LABEL_FONT_SIZE = 21
Y_AXIS_LABEL_FONT_SIZE = 23

PANEL_LABELS = ["A", "B", "C", "D", "E", "F"]
PANEL_TITLES = [
    "Oct–Nov Arctic Warming",
    "Oct–Nov Arctic 500-hPa Geopotential Height",
    "Oct–Nov 300-hPa Zonal-Wind Difference",
    "Dec–Jan North Atlantic Blocking",
    "Dec–Jan Atlantic 850-hPa Zonal-Wind Weakening",
    "Dec–Jan East Europe-West Russia Cooling",
]
PANEL_XLABELS = [
    "1000-500 hPa Temperature (°C)",
    "500-hPa Geopotential Height (gpm)",
    "300 hPa Zonal Wind Difference (m s$^{-1}$)",
    "500 hPa Geopotential Height (gpm)",
    "850 hPa Zonal Wind Difference (m s$^{-1}$)",
    "Surface Air Temperature (°C)",
]
PANEL_YLABELS = [
    "Standardized 1000-500-hPa Temperature Index",
    "Standardized 500-hPa Geopotential Height Index",
    "Standardized 300-hPa Zonal-Wind Shear Index",
    "Standardized 500-hPa Geopotential Height Index",
    "Standardized 850-hPa Zonal-Wind Index",
    "Standardized SAT Index",
]
PANEL_COLORS = [
    TEMP_POSITIVE_COLOR, Z500_POSITIVE_COLOR, U300_NEGATIVE_COLOR,
    Z500_POSITIVE_COLOR, U850_NEGATIVE_COLOR, TEMP_NEGATIVE_COLOR,
]
PANEL_EXPECTED_SIGNS = [1, 1, -1, 1, -1, -1]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": MEDIUM_FONT_SIZE,
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "axes.unicode_minus": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
MATH_MINUS = "\N{MINUS SIGN}"


# ============================================================
# Small repeated utilities
# ============================================================
def find_member_file(directory, experiment, member, suffix=None, ts=False):
    directory = Path(directory)
    if ts:
        names = [f"2deg_TS_{experiment}_{member:02d}.nc",
                 f"2deg_TS_{experiment}_{member}.nc"]
    else:
        prefix = f"{experiment}_1deg"
        if member == 0:
            names = [f"{prefix}{suffix}", f"{prefix}_0{suffix}", f"{prefix}_00{suffix}"]
        else:
            names = [f"{prefix}_{member}{suffix}", f"{prefix}_{member:02d}{suffix}"]
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError("\n".join(str(directory / name) for name in names))


def add_year_month(ds, label):
    time_name = next((x for x in ("time", "Time", "TIME")
                      if x in ds.coords or x in ds.variables), None)
    if time_name is None:
        raise KeyError(f"{label}: 没有时间坐标")
    if time_name != "time":
        ds = ds.rename({time_name: "time"})

    bounds_name = str(ds["time"].attrs.get("bounds", "")).strip()
    if bounds_name not in ds.variables:
        bounds_name = next((x for x in ("time_bnds", "time_bounds")
                            if x in ds.variables), None)

    source = ds["time"]
    if bounds_name is not None:
        bounds = ds[bounds_name]
        bdim = next(d for d in bounds.dims if d != "time")
        source = bounds.isel({bdim: 0})
        print(f"  {Path(str(label)).name}: 使用 {bounds_name} 起点月份")
    else:
        print(f"  {Path(str(label)).name}: 无 time_bnds，使用 time 月份")

    values = np.asarray(source.values)
    units = (source.attrs.get("units") or ds["time"].attrs.get("units")
             or source.encoding.get("units") or ds["time"].encoding.get("units"))
    calendar = (source.attrs.get("calendar") or ds["time"].attrs.get("calendar")
                or source.encoding.get("calendar") or ds["time"].encoding.get("calendar")
                or "standard")
    flat = values.ravel()

    if np.issubdtype(values.dtype, np.datetime64):
        text = flat.astype("datetime64[M]").astype(str)
        years = np.array([int(x[:4]) for x in text])
        months = np.array([int(x[5:7]) for x in text])
    elif values.dtype == object and flat.size and hasattr(flat[0], "year"):
        years = np.array([int(x.year) for x in flat])
        months = np.array([int(x.month) for x in flat])
    else:
        if not units:
            raise ValueError(f"{label}: 数值时间缺少 units")
        match = re.match(r"\s*months?\s+since\s+([+-]?\d+)-(\d+)-(\d+)", str(units), re.I)
        if match:
            y0, m0 = int(match.group(1)), int(match.group(2))
            total = y0 * 12 + m0 - 1 + np.rint(flat.astype(float)).astype(int)
            years, months = total // 12, total % 12 + 1
        else:
            dates = cftime.num2date(flat, units=str(units), calendar=str(calendar),
                                    only_use_cftime_datetimes=True)
            years = np.array([int(x.year) for x in dates])
            months = np.array([int(x.month) for x in dates])

    if years.size != ds.sizes["time"]:
        raise ValueError(f"{label}: 时间长度不一致")
    return ds.assign_coords(year=("time", years), month=("time", months))


def get_field(ds, candidates, label, keyword=None, pressure=False,
              level_hpa=None, geopotential=False):
    name = next((x for x in candidates if x in ds.data_vars), None)
    if name is None and keyword:
        for var_name, da0 in ds.data_vars.items():
            text = " ".join([var_name, str(da0.attrs.get("long_name", "")),
                             str(da0.attrs.get("standard_name", "")),
                             str(da0.attrs.get("units", ""))]).lower()
            if keyword.lower() in text:
                name = var_name
                break
    if name is None:
        raise KeyError(f"{label}: 找不到变量，现有={list(ds.data_vars)}")

    da = ds[name]
    lat_name = next((x for x in ("lat", "latitude", "Latitude", "LAT") if x in da.dims), None)
    lon_name = next((x for x in ("lon", "longitude", "Longitude", "LON") if x in da.dims), None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"{label}: 找不到lat/lon维")
    rename = {}
    if lat_name != "lat": rename[lat_name] = "lat"
    if lon_name != "lon": rename[lon_name] = "lon"
    if rename: da = da.rename(rename)
    da = da.assign_coords(lon=np.mod(da.lon, 360.0)).sortby("lon").sortby("lat")
    _, keep = np.unique(np.round(da.lon.values, 8), return_index=True)
    if len(keep) != da.sizes["lon"]:
        da = da.isel(lon=np.sort(keep))

    if pressure or level_hpa is not None:
        lev = next((x for x in ("plev", "lev", "level", "pressure", "pres",
                                "isobaric", "isobaricInhPa") if x in da.dims), None)
        if lev is None:
            if level_hpa is not None:
                print(f"  {label}: 变量本身为单层场，直接作为{level_hpa:g} hPa使用")
        else:
            if lev != "plev": da = da.rename({lev: "plev"})
            p = np.asarray(da.plev.values, float)
            units = str(da.plev.attrs.get("units", "")).lower()
            if units == "pa" or ("pa" in units and "hpa" not in units and "millibar" not in units) or np.nanmax(np.abs(p)) > 2000:
                p = p / 100.0
            da = da.assign_coords(plev=("plev", p)).sortby("plev")
            if level_hpa is not None:
                p = np.asarray(da.plev.values, float)
                index = int(np.nanargmin(np.abs(p - level_hpa)))
                if abs(float(p[index]) - level_hpa) > 2.0:
                    raise ValueError(f"{label}: 没有{level_hpa:g} hPa")
                da = da.isel(plev=index, drop=True)

    if geopotential:
        units = str(da.attrs.get("units", "")).lower()
        sample = da
        for dim in ("time", "lat", "lon"):
            if dim in sample.dims:
                sample = sample.isel({dim: slice(0, min(10, sample.sizes[dim]))})
        if ("m2" in units or "m^2" in units or "s-2" in units or "s^-2" in units
                or float(sample.mean(skipna=True)) > 20000):
            da = da / 9.80665
        da.attrs["units"] = "gpm"
    return da


def select_analysis_months(da, months):
    if DO_DETREND:
        return da
    years = np.asarray(da.year.values, int)
    mm = np.asarray(da.month.values, int)
    season_year = years.copy()
    season_year[np.isin(mm, [1, 2, 3])] -= 1
    analysis_years = np.concatenate((TARGET_YEARS, BASELINE_YEARS))
    keep = np.isin(mm, months) & np.isin(season_year, analysis_years)
    return da.isel(time=np.flatnonzero(keep))


def area_mean(da, lon1, lon2, lat1, lat2, label):
    south, north = sorted((float(lat1), float(lat2)))
    region = da.sel(lat=slice(south, north))
    raw_span = float(lon2) - float(lon1)
    if not np.isclose(abs(raw_span), 360.0):
        start, end = float(lon1) % 360.0, float(lon2) % 360.0
        if start < end:
            region = region.sel(lon=slice(start, end))
        else:
            region = region.where((region.lon >= start) | (region.lon <= end), drop=True)
    if region.sizes.get("lat", 0) == 0 or region.sizes.get("lon", 0) == 0:
        raise ValueError(f"{label}: 区域筛选后为空")
    weights = np.cos(np.deg2rad(region.lat))
    return region.weighted(weights).mean(("lat", "lon"), skipna=True)


def seasonal_anomaly(series, months, label):
    selected = series.where(series.month.isin(months), drop=True)
    years = selected.year.values.astype(int)
    mm = selected.month.values.astype(int)
    season_year = years.copy()
    season_year[np.isin(mm, [1, 2, 3])] -= 1

    fields = []
    required = set(int(m) for m in months)
    for y in np.unique(season_year):
        mask = season_year == y
        if not required.issubset(set(mm[mask].tolist())):
            continue
        fields.append(selected.isel(time=np.where(mask)[0]).mean("time", skipna=True)
                      .expand_dims(time=[int(y)]))
    if not fields:
        raise ValueError(f"{label}: 没有完整季节")
    out = xr.concat(fields, dim="time", coords="minimal", compat="override",
                    combine_attrs="drop").sortby("time")

    if DO_DETREND:
        x = np.asarray(out.time.values, float)
        y = np.asarray(out.values, float)
        valid = np.isfinite(x) & np.isfinite(y)
        xmean = float(np.mean(x[valid]))
        slope = float(np.polyfit(x[valid], y[valid], 1)[0])
        values = y.copy()
        values[valid] = y[valid] - slope * (x[valid] - xmean)
        values[~valid] = np.nan
        out = out.copy(deep=True)
        out.data = values

    available = set(out.time.values.astype(int).tolist())
    missing_t = [int(y) for y in TARGET_YEARS if int(y) not in available]
    missing_b = [int(y) for y in BASELINE_YEARS if int(y) not in available]
    if missing_t or missing_b:
        raise ValueError(f"{label}: 缺少年份 target={missing_t}, baseline={missing_b}")
    return float(out.sel(time=TARGET_YEARS).mean("time", skipna=True)
                 - out.sel(time=BASELINE_YEARS).mean("time", skipna=True))


def classify(values, expected_sign):
    values = np.asarray(values, float)
    sd = float(np.std(values, ddof=1))
    threshold = SD_MULTIPLIER * sd
    expected = values > 0 if expected_sign > 0 else values < 0
    outside = values < -threshold if expected_sign > 0 else values > threshold
    within = (~expected) & (~outside)
    return expected, within, outside, sd, threshold


def collision_free_member_x(ax, values, member_numbers, marker_size=MEMBER_SCATTER_SIZE, marker_linewidth=1.25):
    values = np.asarray(values, dtype=float)
    member_numbers = np.asarray(member_numbers, dtype=int)
    if values.size != member_numbers.size:
        raise ValueError('values与member_numbers长度必须一致。')
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = float(x_max - x_min)
    y_span = float(y_max - y_min)
    if x_span <= 0 or y_span <= 0:
        raise ValueError('蜂群避让前必须先设置有效的xlim和ylim。')
    position = ax.get_position()
    axes_width_pt = ax.figure.get_figwidth() * 72.0 * position.width
    axes_height_pt = ax.figure.get_figheight() * 72.0 * position.height
    marker_diameter_pt = (np.sqrt(float(marker_size)) + float(marker_linewidth)) * MEMBER_SWARM_MARGIN
    marker_diameter_y = marker_diameter_pt * y_span / axes_height_pt
    vertical_distance = np.abs(values[:, None] - values[None, :])
    crowding = np.sum(vertical_distance < 1.6 * marker_diameter_y, axis=1)
    placement_order = np.argsort(-crowding, kind='stable')
    x_result = np.full(values.size, np.nan, dtype=float)
    placed_indices = []
    group_centers = np.linspace(-9.0, -20.8, 4)
    for index in placement_order:
        member_no = int(member_numbers[index])
        group_index = int(np.clip((member_no - 1) // 5, 0, 3))
        preferred_x = group_centers[group_index]
        if not placed_indices:
            chosen_x = float(np.clip(preferred_x, MEMBER_SWARM_X_LEFT, MEMBER_SWARM_X_RIGHT))
        else:
            placed = np.asarray(placed_indices, dtype=int)
            dy_pt = np.abs(values[index] - values[placed]) / y_span * axes_height_pt
            close = dy_pt < marker_diameter_pt
            forbidden = []
            for placed_index, one_dy_pt in zip(placed[close], dy_pt[close]):
                required_dx_pt = np.sqrt(max(marker_diameter_pt ** 2 - float(one_dy_pt) ** 2, 0.0))
                required_dx_data = required_dx_pt * x_span / axes_width_pt
                forbidden.append((float(x_result[placed_index] - required_dx_data), float(x_result[placed_index] + required_dx_data)))
            tolerance = 1e-09
            candidates = [float(np.clip(preferred_x, MEMBER_SWARM_X_LEFT, MEMBER_SWARM_X_RIGHT)), MEMBER_SWARM_X_LEFT, MEMBER_SWARM_X_RIGHT]
            for interval_left, interval_right in forbidden:
                candidates.extend([interval_left - tolerance, interval_right + tolerance])
            candidates = np.asarray(candidates, dtype=float)
            candidates = candidates[(candidates >= MEMBER_SWARM_X_LEFT) & (candidates <= MEMBER_SWARM_X_RIGHT)]
            candidates = np.unique(candidates)
            candidate_order = np.argsort(np.abs(candidates - preferred_x), kind='stable')
            chosen_x = None
            for candidate_index in candidate_order:
                candidate_x = float(candidates[candidate_index])
                inside_forbidden = any((interval_left + tolerance < candidate_x and candidate_x < interval_right - tolerance for interval_left, interval_right in forbidden))
                if not inside_forbidden:
                    chosen_x = candidate_x
                    break
            if chosen_x is None:
                probe = np.linspace(MEMBER_SWARM_X_LEFT, MEMBER_SWARM_X_RIGHT, 2001)
                dx_pt = np.abs(probe[:, None] - x_result[placed][None, :]) / x_span * axes_width_pt
                screen_distance = np.hypot(dx_pt, dy_pt[None, :])
                nearest = np.min(screen_distance, axis=1)
                chosen_x = float(probe[int(np.argmax(nearest))])
                warnings.warn(f'member {member_no}: 蜂群区域达到容量上限，已放到屏幕间距最大的横向位置。')
        x_result[index] = chosen_x
        placed_indices.append(int(index))
    return x_result

def draw_vertical_distribution(ax, specification):
    raw_all_values = np.asarray(specification['values'], dtype=float)
    all_values = np.asarray(specification['plot_values'], dtype=float)
    member_numbers = np.arange(1, all_values.size + 1, dtype=int)
    finite = np.isfinite(all_values) & np.isfinite(raw_all_values)
    values = all_values[finite]
    raw_values = raw_all_values[finite]
    member_numbers = member_numbers[finite]
    if values.size == 0:
        raise ValueError(f"{specification['title']}: 没有有效成员值。")
    y_grid = np.linspace(-COMMON_STANDARDIZED_Y_LIMIT, COMMON_STANDARDIZED_Y_LIMIT, 600)
    if values.size >= 2 and np.nanstd(values, ddof=1) > 0:
        kde = gaussian_kde(values, bw_method=0.35)
        true_pdf = np.asarray(kde(y_grid), dtype=float)
    else:
        sigma = max(COMMON_STANDARDIZED_Y_LIMIT * 0.08, 1e-06)
        true_pdf = np.exp(-0.5 * ((y_grid - float(np.nanmean(values))) / sigma) ** 2)
    area = float(np.trapezoid(true_pdf, y_grid) if hasattr(np, 'trapezoid') else np.trapz(true_pdf, y_grid))
    if np.isfinite(area) and area > 0:
        true_pdf = true_pdf / area
    n_bins = max(6, min(10, values.size // 2))
    _, hist_edges = np.histogram(values, bins=n_bins)
    representative_bin_width = float(np.median(np.diff(hist_edges)))
    proportion_curve = true_pdf * representative_bin_width * 100.0
    rgb = np.array(plt.matplotlib.colors.to_rgb(specification['main_color']), float)
    pdf_fill_color = tuple(np.clip(rgb * (1.0 - PDF_FILL_LIGHTEN_AMOUNT) + PDF_FILL_LIGHTEN_AMOUNT, 0.0, 1.0))
    ax.fill_betweenx(y_grid, 0.0, -proportion_curve, facecolor=pdf_fill_color, edgecolor='none', alpha=1.0, zorder=3)
    ax.plot(-proportion_curve, y_grid, color=specification['main_color'], linewidth=2.3, zorder=5)
    rgb = np.array(plt.matplotlib.colors.to_rgb(specification['main_color']), float)
    box_color = tuple(np.clip(rgb * 0.72, 0.0, 1.0))
    box_x = -2.0
    box = ax.boxplot([values], positions=[box_x], widths=2.8, vert=True, patch_artist=True, showfliers=False, whis=(0, 100), manage_ticks=False, medianprops={'color': 'black', 'linewidth': 2.2}, boxprops={'facecolor': specification['main_color'], 'edgecolor': box_color, 'linewidth': 2.0}, whiskerprops={'color': box_color, 'linewidth': 1.8}, capprops={'color': box_color, 'linewidth': 1.8})
    for artist in box['boxes']:
        artist.set_alpha(1.0)
        artist.set_zorder(10)
    for artist in box['medians']:
        artist.set_zorder(12)
    for artist in box['whiskers'] + box['caps']:
        artist.set_zorder(10)
    mean_value = float(np.mean(values))
    ax.scatter(box_x, mean_value, marker='D', s=105, facecolor=box_color, edgecolor='black', linewidth=1.0, zorder=12)
    expected, opposite_within, opposite_outside, _, _ = classify(raw_values, specification['expected_sign'])
    ax.set_ylim(-COMMON_STANDARDIZED_Y_LIMIT, COMMON_STANDARDIZED_Y_LIMIT)
    ax.set_xlim(-30.0, 0.0)
    x_points = collision_free_member_x(ax, values, member_numbers, marker_size=max(MEMBER_SCATTER_SIZE, GRAY_MARKER_SIZE), marker_linewidth=1.25)
    for x, value, member_no, ok, is_light, is_gray in zip(x_points, values, member_numbers, expected, opposite_within, opposite_outside):
        is_special_member = member_no in SPECIAL_TRACKED_MEMBERS
        is_mechanism_panel = specification.get('subplot_label', '') in {'(a)', '(b)', '(c)', '(d)', '(e)'}

        if is_special_member and is_mechanism_panel:
            # 追踪最终无冷却的4个成员：
            # 1) 不符合预期且幅度超过0.5 SD -> 灰色正方形；
            # 2) 符合预期，或虽不符合预期但未超过0.5 SD -> 灰色菱形。
            point_color, point_edge = (UNSATISFIED_COLOR, '#666666')
            point_marker = 's' if is_gray else 'D'
            point_size = GRAY_MARKER_SIZE
            point_number_fontsize = GRAY_MARKER_NUMBER_FONT_SIZE
        elif is_special_member and specification.get('subplot_label', '') == '(f)':
            # 图(f)保留2、10、13、18作为最终无冷却成员的灰色正方形。
            point_color, point_edge = (UNSATISFIED_COLOR, '#666666')
            point_marker = 's'
            point_size = GRAY_MARKER_SIZE
            point_number_fontsize = GRAY_MARKER_NUMBER_FONT_SIZE
        elif ok or is_light:
            point_color = specification['main_color']
            point_edge = tuple(np.clip(np.array(plt.matplotlib.colors.to_rgb(specification['main_color']), float) * 0.76, 0.0, 1.0))
            point_marker = 'o'
            point_size = MEMBER_SCATTER_SIZE
            point_number_fontsize = MEMBER_NUMBER_FONT_SIZE
        elif is_gray:
            point_color, point_edge = (UNSATISFIED_COLOR, '#666666')
            point_marker = 's'
            point_size = GRAY_MARKER_SIZE
            point_number_fontsize = GRAY_MARKER_NUMBER_FONT_SIZE
        else:
            raise RuntimeError(f'member {member_no}: 未被颜色判定覆盖')
        ax.scatter(x, value, s=point_size, marker=point_marker, facecolors=point_color, edgecolors=point_edge, linewidths=1.35, zorder=7)

        # 灰色标记内部的成员编号使用黑色；其他彩色标记仍使用白色编号。
        is_gray_marker = (
            isinstance(point_color, str) and point_color.lower() == UNSATISFIED_COLOR.lower()
        ) or (
            not isinstance(point_color, str)
            and np.allclose(
                np.asarray(point_color, dtype=float),
                np.asarray(plt.matplotlib.colors.to_rgb(UNSATISFIED_COLOR), dtype=float)
            )
        )
        number_color = 'black' if is_gray_marker else 'white'
        outline_color = 'white' if is_gray_marker else '#333333'
        number_text = ax.text(
            x, value, str(member_no),
            ha='center', va='center',
            fontsize=point_number_fontsize,
            fontweight='bold',
            color=number_color,
            zorder=9
        )
        number_text.set_path_effects([
            path_effects.withStroke(
                linewidth=MEMBER_NUMBER_STROKE_WIDTH,
                foreground=outline_color
            )
        ])
    ax.axhline(0.0, color='black', linewidth=1.8, linestyle='-', zorder=6, clip_on=True)
    ax.set_xticks([-30.0, -20.0, -10.0, 0.0])
    ax.set_xticklabels(['30', '20', '10', '0'])
    ax.set_xlabel('PDF (%)', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='normal', labelpad=10)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(specification.get('ylabel_standardized', 'Standardized value (σ)'), fontsize=Y_AXIS_LABEL_FONT_SIZE, fontweight=Y_AXIS_LABEL_WEIGHT, rotation=90, labelpad=8)
    ax.tick_params(axis='x', labelsize=AXIS_TICK_FONT_SIZE, direction='out', length=5, width=1.4)
    ax.tick_params(axis='y', labelsize=AXIS_TICK_FONT_SIZE, direction='out', length=5.5, width=Y_AXIS_TICK_WIDTH)
    ax.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True)
    ax.set_yticks(COMMON_STANDARDIZED_Y_TICKS)
    ax.set_yticklabels([f'{float(tick):g}'.replace('-', MATH_MINUS) for tick in COMMON_STANDARDIZED_Y_TICKS])
    for tick in ax.get_xticklabels():
        tick.set_fontweight('normal')
    for tick in ax.get_yticklabels():
        tick.set_fontweight(Y_AXIS_TICK_LABEL_WEIGHT)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_linewidth(Y_AXIS_SPINE_LINEWIDTH)
    ax.spines['right'].set_color('black')
    subplot_label = specification.get('subplot_label', '')
    if subplot_label:
        ax.text(0.02, 0.98, subplot_label, transform=ax.transAxes, ha='left', va='top', fontsize=31, fontweight='bold', color='black', zorder=20, clip_on=False)
    ax.set_facecolor('white')
    ax.set_zorder(2)
    return ax


# ============================================================
# 20 paired members: reconstructed anomaly minus HadISST anomaly
# ============================================================
warming_member_diff = []
z500_on_member_diff = []
jet300_member_diff = []
blocking_member_diff = []
u850_region_member_diff = []
eewr_member_diff = []

for member in range(EXPECTED_MEMBERS):
    print("\n" + "=" * 72)
    print(f"member {member:02d}")
    print("=" * 72)

    experiment_result = {}

    for experiment in ("HadISST", "reconstructed"):
        # ---- temperature + Z500 ----
        zt_file = find_member_file(PATH_ZT_PLEV_DIR, experiment, member, ZT_FILE_SUFFIX)
        ds = xr.open_dataset(zt_file, decode_times=False, mask_and_scale=True)
        try:
            ds = add_year_month(ds, zt_file)

            temp = get_field(
                ds, ("T", "ta", "temp", "temperature", "air", "air_temperature"),
                f"{experiment} T member {member:02d}", keyword="temperature", pressure=True,
            )
            temp = select_analysis_months(temp, [10, 11])
            temp = temp.sel(plev=slice(PRESSURE_TOP, PRESSURE_BOTTOM))
            if temp.sizes.get("plev", 0) < 2:
                raise ValueError(f"{experiment} member {member:02d}: 500–1000 hPa有效层数不足")
            lower_temp = temp.mean("plev", skipna=True)
            warm_series = area_mean(
                lower_temp, WARM_LON1, WARM_LON2, WARM_LAT1, WARM_LAT2,
                f"{experiment} Arctic warming member {member:02d}",
            ).load()
            warming = seasonal_anomaly(
                warm_series, [10, 11],
                f"{experiment} ON Arctic warming member {member:02d}",
            )

            z500 = get_field(
                ds, ("Z500", "HGT500", "zg500", "z500", "Z3_500", "Z_500",
                     "HGT_500", "Z3", "z3", "Z", "z", "zg", "ZG", "HGT", "hgt",
                     "geopotential_height", "height", "PHI", "phi"),
                f"{experiment} Z500 member {member:02d}", keyword="geopotential",
                level_hpa=500.0, geopotential=True,
            )
            # Oct-Nov 500-hPa geopotential height over the same box as panel (a).
            z500_on = select_analysis_months(z500, [10, 11])
            z500_on_series = area_mean(
                z500_on, WARM_LON1, WARM_LON2, WARM_LAT1, WARM_LAT2,
                f"{experiment} ON Arctic Z500 member {member:02d}",
            ).load()
            z500_on_value = seasonal_anomaly(
                z500_on_series, [10, 11],
                f"{experiment} ON Arctic Z500 member {member:02d}",
            )

            # Keep the existing Dec-Jan North Atlantic blocking index.
            z500 = select_analysis_months(z500, [12, 1])
            block_series = area_mean(
                z500, BLOCK_LON1, BLOCK_LON2, BLOCK_LAT1, BLOCK_LAT2,
                f"{experiment} Atlantic-Arctic Z500 member {member:02d}",
            ).load()
            blocking = seasonal_anomaly(
                block_series, [12, 1],
                f"{experiment} DJ Atlantic-Arctic Z500 member {member:02d}",
            )
        finally:
            ds.close()

        # ---- U300 + U850 ----
        uv_file = find_member_file(PATH_UV_PLEV_DIR, experiment, member, UV_FILE_SUFFIX)
        ds = xr.open_dataset(uv_file, decode_times=False, mask_and_scale=True)
        try:
            ds = add_year_month(ds, uv_file)
            u = get_field(
                ds, ("U", "u", "UWND", "uwnd", "ua", "U300", "uwnd300", "u300",
                     "U_300", "U850", "uwnd850", "u850", "U_850"),
                f"{experiment} U member {member:02d}", keyword="zonal wind", pressure=True,
            )

            p = np.asarray(u.plev.values, float)
            i300 = int(np.nanargmin(np.abs(p - JET300_LEVEL_HPA)))
            i850 = int(np.nanargmin(np.abs(p - JET850_LEVEL_HPA)))
            if abs(float(p[i300]) - JET300_LEVEL_HPA) > 2 or abs(float(p[i850]) - JET850_LEVEL_HPA) > 2:
                raise ValueError(f"{experiment} member {member:02d}: 缺少300或850 hPa")

            u300 = select_analysis_months(u.isel(plev=i300, drop=True), JET300_MONTHS)
            south = area_mean(
                u300, JET300_LON1, JET300_LON2,
                JET300_SOUTH_LAT1, JET300_SOUTH_LAT2, "U300 south",
            ).load()
            north = area_mean(
                u300, JET300_LON1, JET300_LON2,
                JET300_NORTH_LAT1, JET300_NORTH_LAT2, "U300 north",
            ).load()
            jet300 = seasonal_anomaly(
                north - south, JET300_MONTHS,
                f"{experiment} ON U300 difference member {member:02d}",
            )

            u850 = select_analysis_months(u.isel(plev=i850, drop=True), JET850_MONTHS)
            u850_series = area_mean(
                u850, JET850_LON1, JET850_LON2, JET850_LAT1, JET850_LAT2,
                f"{experiment} DJ U850 member {member:02d}",
            ).load()
            u850_value = seasonal_anomaly(
                u850_series, JET850_MONTHS,
                f"{experiment} DJ U850 member {member:02d}",
            )
        finally:
            ds.close()

        # ---- surface air temperature ----
        try:
            ts_base = PATH_TS_HAD_BASE if experiment == "HadISST" else PATH_TS_REC_BASE
            ts_file = find_member_file(ts_base, experiment, member, ts=True)
            ds = xr.open_dataset(ts_file, decode_times=False, mask_and_scale=True)
            try:
                ds = add_year_month(ds, ts_file)
                ts = get_field(
                    ds, ("TS", "ts", "TREFHT", "tas", "t2m"),
                    f"{experiment} TS member {member:02d}", keyword="surface temperature",
                )
                units = str(ts.attrs.get("units", "")).lower()
                if units in ("k", "kelvin") or float(ts.isel(time=0).mean(skipna=True)) > 150:
                    ts = ts - 273.15
                ts = select_analysis_months(ts, [12, 1])
                ts_series = area_mean(
                    ts, EEWR_LON1, EEWR_LON2, EEWR_LAT1, EEWR_LAT2,
                    f"{experiment} EEWR TS member {member:02d}",
                ).load()
                eewr = seasonal_anomaly(
                    ts_series, [12, 1],
                    f"{experiment} DJ EEWR member {member:02d}",
                )
            finally:
                ds.close()
        except FileNotFoundError:
            eewr = np.nan

        experiment_result[experiment] = {
            "warming": warming,
            "z500_on": z500_on_value,
            "jet300": jet300,
            "blocking": blocking,
            "u850": u850_value,
            "eewr": eewr,
        }

    warming_diff = experiment_result["reconstructed"]["warming"] - experiment_result["HadISST"]["warming"]
    z500_on_diff = experiment_result["reconstructed"]["z500_on"] - experiment_result["HadISST"]["z500_on"]
    jet300_diff = experiment_result["reconstructed"]["jet300"] - experiment_result["HadISST"]["jet300"]
    blocking_diff = experiment_result["reconstructed"]["blocking"] - experiment_result["HadISST"]["blocking"]
    u850_diff = experiment_result["reconstructed"]["u850"] - experiment_result["HadISST"]["u850"]

    if np.isfinite(experiment_result["reconstructed"]["eewr"]) and np.isfinite(experiment_result["HadISST"]["eewr"]):
        eewr_diff = experiment_result["reconstructed"]["eewr"] - experiment_result["HadISST"]["eewr"]
    elif ALLOW_VERIFIED_EEWR_FALLBACK:
        eewr_diff = float(VERIFIED_EEWR_FALLBACK[member])
        print(f"  TS文件不可用，member {member:02d} 使用已核验EEWR后备值 {eewr_diff:.3f} °C")
    else:
        raise FileNotFoundError(f"member {member:02d}: TS文件不可用")

    warming_member_diff.append(warming_diff)
    z500_on_member_diff.append(z500_on_diff)
    jet300_member_diff.append(jet300_diff)
    blocking_member_diff.append(blocking_diff)
    u850_region_member_diff.append(u850_diff)
    eewr_member_diff.append(eewr_diff)

    print(
        f"member {member:02d}: warming={warming_diff:.3f} °C, "
        f"ON Z500={z500_on_diff:.3f} gpm, "
        f"U300={jet300_diff:.3f} m s^-1, Z500={blocking_diff:.3f} gpm, "
        f"U850={u850_diff:.3f} m s^-1, EEWR={eewr_diff:.3f} °C"
    )

warming_member_diff = np.asarray(warming_member_diff, float)
z500_on_member_diff = np.asarray(z500_on_member_diff, float)
jet300_member_diff = np.asarray(jet300_member_diff, float)
blocking_member_diff = np.asarray(blocking_member_diff, float)
u850_region_member_diff = np.asarray(u850_region_member_diff, float)
eewr_member_diff = np.asarray(eewr_member_diff, float)

indicator_arrays = {
    "warming": (warming_member_diff, +1),
    "z500_on": (z500_on_member_diff, +1),
    "jet300": (jet300_member_diff, -1),
    "blocking": (blocking_member_diff, +1),
    "jet850": (u850_region_member_diff, -1),
    "eewr": (eewr_member_diff, -1),
}
for name, (values, sign) in indicator_arrays.items():
    expected, light, gray, sd, threshold = classify(values, sign)
    print(
        f"{name}: SD={sd:.4f}, ±0.5SD={threshold:.4f}; "
        f"main={(np.flatnonzero(expected)+1).tolist()}; "
        f"light={(np.flatnonzero(light)+1).tolist()}; "
        f"gray={(np.flatnonzero(gray)+1).tolist()}"
    )


# ============================================================
# Standardize each index and draw the six vertical rainclouds
# ============================================================
# 图(a)、(b)和图(d)仅在绘图时乘以 -1；其他面板保持原值。
panel_values = [
    -warming_member_diff, -z500_on_member_diff, jet300_member_diff,
    -blocking_member_diff, u850_region_member_diff, eewr_member_diff,
]
# 同步翻转(a)、(b)、(d)的预期符号判定，保证点的分类颜色与翻转后的数值一致。
plot_expected_signs = [
    -PANEL_EXPECTED_SIGNS[0], -PANEL_EXPECTED_SIGNS[1],
    PANEL_EXPECTED_SIGNS[2], -PANEL_EXPECTED_SIGNS[3],
    PANEL_EXPECTED_SIGNS[4], PANEL_EXPECTED_SIGNS[5],
]
distribution_specs = []
for values, label, title, xlabel, ylabel, color, sign in zip(
    panel_values, PANEL_LABELS, PANEL_TITLES, PANEL_XLABELS,
    PANEL_YLABELS, PANEL_COLORS, plot_expected_signs,
):
    sd = float(np.std(values[np.isfinite(values)], ddof=1))
    distribution_specs.append({
        "values": values,
        "plot_values": values / sd,
        "raw_sample_sd": sd,
        "subplot_label": label,
        "title": title,
        "xlabel": xlabel,
        "ylabel_standardized": ylabel,
        "main_color": color,
        "expected_sign": sign,
    })
    print(f"{title}: Y轴按20成员样本标准差标准化，SD={sd:.4f}")

GLOBAL_STANDARDIZED_ABSMAX = max(
    float(np.nanmax(np.abs(spec["plot_values"]))) for spec in distribution_specs
)
COMMON_STANDARDIZED_Y_LIMIT = max(
    1.5, np.ceil(GLOBAL_STANDARDIZED_ABSMAX * 1.12 / 0.5) * 0.5
)
if COMMON_STANDARDIZED_Y_LIMIT <= 2.5:
    COMMON_STANDARDIZED_Y_TICK_STEP = 0.5
elif COMMON_STANDARDIZED_Y_LIMIT <= 5.0:
    COMMON_STANDARDIZED_Y_TICK_STEP = 1.0
else:
    COMMON_STANDARDIZED_Y_TICK_STEP = 2.0
COMMON_STANDARDIZED_Y_TICKS = np.arange(
    -COMMON_STANDARDIZED_Y_LIMIT,
    COMMON_STANDARDIZED_Y_LIMIT + 0.5 * COMMON_STANDARDIZED_Y_TICK_STEP,
    COMMON_STANDARDIZED_Y_TICK_STEP,
)

fig_distribution = plt.figure(
    figsize=DISTRIBUTION_FIGURE_SIZE, dpi=FIGURE_DPI, facecolor="white"
)
container_axes = fig_distribution.subplots(2, 3, squeeze=False).ravel().tolist()
fig_distribution.subplots_adjust(
    left=0.075, right=0.935, bottom=0.065, top=0.975,
    wspace=0.28, hspace=0.30
)

distribution_axes = [
    draw_vertical_distribution(ax, spec)
    for ax, spec in zip(container_axes, distribution_specs)
]

# 在每一行左侧、外框内部添加竖排季节标签。
row1_y = np.mean([
    container_axes[i].get_position().y0 +
    container_axes[i].get_position().height / 2
    for i in (0, 1, 2)
])
row2_y = np.mean([
    container_axes[i].get_position().y0 +
    container_axes[i].get_position().height / 2
    for i in (3, 4, 5)
])
row_label_x = 0.047
row_label_fontsize = 32
row_label_1 = fig_distribution.text(
    row_label_x, row1_y, "Oct–Nov",
    rotation=90, ha="center", va="center",
    fontsize=row_label_fontsize, fontweight="bold", color="black"
)
row_label_2 = fig_distribution.text(
    row_label_x, row2_y, "Dec–Jan",
    rotation=90, ha="center", va="center",
    fontsize=row_label_fontsize, fontweight="bold", color="black"
)

# 已移除包围六张子图和季节标签的最外层矩形框。



if SAVE_FIGURE:
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    png_path = OUTPUT_STEM.with_suffix(".png")

    # PDF 为矢量图；PNG 使用 300 dpi，避免 1200 dpi 导致文件过大或内存不足。
    fig_distribution.savefig(
        pdf_path, format="pdf", bbox_inches="tight", facecolor="white"
    )
    fig_distribution.savefig(
        png_path, format="png", dpi=300,
        bbox_inches="tight", facecolor="white"
    )
    print(f"PDF 已保存：{pdf_path.resolve()}")
    print(f"PNG 已保存：{png_path.resolve()}")

print("\n六个指标的集合均值（原始物理单位；未标准化）：")
print(f"Arctic Warming: {warming_member_diff.mean():.3f} °C")
print(f"Oct-Nov Arctic Z500: {z500_on_member_diff.mean():.3f} gpm")
print(f"U300 Zonal-Wind Difference: {jet300_member_diff.mean():.3f} m s^-1")
print(f"Atlantic-Arctic Z500 High: {blocking_member_diff.mean():.3f} gpm")
print(f"Atlantic U850 Regional Zonal-Wind Weakening: {u850_region_member_diff.mean():.3f} m s^-1")
print(f"EEWR Cooling: {eewr_member_diff.mean():.3f} °C")

if SHOW_FIGURE:
    plt.show(block=False)

print("\n全部完成。")
