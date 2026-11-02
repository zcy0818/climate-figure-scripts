# -*- coding: utf-8 -*-
from pathlib import Path
import re
import warnings

import cftime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

warnings.filterwarnings("ignore")


# ============================================================
# 1. 可编辑参数
# ============================================================
TEMP_DIR = Path(r"F:\model hgt T\ZTlevel")
REC_PATTERNS = (
    "reconstructed_1deg*.cam.h0.extract05-fulldims_plev.nc",
    "reconstructed_1deg*.cam.h0.extract04-fulldims_plev.nc",
    "reconstructed_1deg*.cam.h0.*_plev.nc",
)
HAD_PATTERNS = (
    "HadISST_1deg*.cam.h0.extract05-fulldims_plev.nc",
    "HadISST_1deg*.cam.h0.extract04-fulldims_plev.nc",
    "HadISST_1deg*.cam.h0.*_plev.nc",
)
REC_SIC_FILE = Path(r"E:\20century\data\sic_reconstructed_1901_2019_all_globe_data.nc")
HAD_SIC_FILE = Path(r"E:\DATA\HadISST_ice.nc\HadISST_ice.nc")

TEMP_NAMES = ("T", "ta", "temp", "temperature", "air", "air_temperature")
SIC_NAMES = ("sic", "siconc", "ice", "ice_conc", "ice_concentration",
             "seaice", "sea_ice_concentration", "ICEFRAC", "aice")

TARGET_YEARS = np.arange(1940, 1943)
BASELINE_YEARS = np.arange(1971, 2001)
MONTHS = (10, 11)
EXPECTED_MEMBERS = 20

SECTION_LON_MIN, SECTION_LON_MAX = 120.0, 240.0
SECTION_LONGITUDE = 180.0
LAT_MIN, LAT_MAX = 66.5, 90.0
PRESSURE_TOP, PRESSURE_BOTTOM = 300.0, 1000.0
PRESSURE_TICKS = [1000, 925, 850, 700, 500, 400, 300]
LAT_TICKS = (70, 80, 90)

DETREND_TEMPERATURE = False
DETREND_KEEP_MEAN = True
MIN_DETREND_VALID_YEARS = 20
VERTICAL_INTERPOLATION_FACTOR = 8
SECTION_INTERPOLATION_FACTOR = 4

TEMP_VMIN, TEMP_VMAX, TEMP_COLOR_INTERVAL = -1.8, 1.8, 0.3
TEMP_COLOR_ANCHORS = ["#21618C", "#75ADD1", "#DCEBF2", "#F7F7F2",
                      "#F2D3C4", "#D97A63", "#9E2F3F"]
TEMP_CBAR_WIDTH_INCH = 0.12
TEMP_CBAR_LABEL = "Temperature (°C)"

SIGNIFICANCE_LEVEL = 0.95
SIGNIFICANCE_ALPHA = 1.0 - SIGNIFICANCE_LEVEL
MONTE_CARLO_N_RANDOM = 10000
MONTE_CARLO_RANDOM_SEED = 20260627
MONTE_CARLO_CHUNK_POINTS = 5000
SIGNIFICANCE_CONTOUR_COLOR = "#4B1D6B"
SIGNIFICANCE_CONTOUR_LINEWIDTH = 1.4
SIGNIFICANCE_CONTOUR_OPACITY = 1.0
SIGNIFICANCE_CONTOUR_ZORDER = 12

SIC_TICK_INTERVAL = 10.0
SIC_POSITIVE_HEADROOM_RATIO = 0.08
SIC_BAR_WIDTH_FACTOR = 0.90
SIC_COLOR_LIGHTEN = 0.20

BOX_COLOR = "#D81B60"
BOX_LINEWIDTH = 3.0
BOX_OPACITY = 1.0
BOX_PRESSURE_TOP, BOX_PRESSURE_BOTTOM = 500.0, 1000.0

SMALL_FONT_SIZE, MEDIUM_FONT_SIZE, LARGE_FONT_SIZE = 8, 9, 10
AXIS_LINEWIDTH = 0.6
TICK_LENGTH, TICK_WIDTH = 2.5, 0.6
POLE_LINEWIDTH = 0.45

FIGURE_SIZE = (183.0 / 25.4, 92.0 / 25.4)
FIGURE_DPI = 800
PANEL_HORIZONTAL_WSPACE = 0.30
SIC_PANEL_EXTRA_DOWN = 0.020
FIG_LEFT, FIG_RIGHT, FIG_BOTTOM, FIG_TOP = 0.075, 0.915, 0.12, 0.94
TEMP_SIC_HEIGHT_RATIOS = [4.0, 1.10]
CBAR_X_GAP, CBAR_BOTTOM_RATIO, CBAR_HEIGHT_RATIO = 0.012, 0.08, 0.84

SHOW_FIGURE = True
OUTPUT_DIR = TEMP_DIR / "Arctic_T_SIC_section_1940_1942_OctNov_reconstructed_vs_HadISST_180E"
OUTPUT_BASE = OUTPUT_DIR / "figure3_ab_section_only"
OUTPUT_SVG = OUTPUT_BASE.with_suffix(".svg")
OUTPUT_PDF = OUTPUT_BASE.with_suffix(".pdf")
OUTPUT_FIGURE = OUTPUT_BASE.with_suffix(".png")
OUTPUT_NC = OUTPUT_DIR / ("Reconstructed_vs_HadISST_T_SIC_anomaly_1940-1942_"
                          "OctNov_120-240Emean_66.5-90N_SICtick10_MC95_no_detrend.nc")

brbg = plt.get_cmap("BrBG")
neg_rgb = np.array(mcolors.to_rgb(brbg(0.15)))
pos_rgb = np.array(mcolors.to_rgb(brbg(0.85)))
SIC_NEG_COLOR = tuple(neg_rgb * (1 - SIC_COLOR_LIGHTEN) + SIC_COLOR_LIGHTEN)
SIC_POS_COLOR = tuple(pos_rgb * (1 - SIC_COLOR_LIGHTEN) + SIC_COLOR_LIGHTEN)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
    "font.size": MEDIUM_FONT_SIZE, "font.weight": "normal",
    "axes.labelweight": "normal", "axes.titleweight": "bold",
    "axes.linewidth": AXIS_LINEWIDTH, "xtick.major.width": AXIS_LINEWIDTH,
    "ytick.major.width": AXIS_LINEWIDTH, "axes.unicode_minus": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ============================================================
# 2. 三个必要函数
# ============================================================
def add_year_month(ds, label):
    """有time_bnds/time_bounds时按区间起点确定年月。"""
    time_name = next((x for x in ("time", "Time", "TIME") if x in ds), None)
    if time_name is None:
        raise KeyError(f"{label} 没有时间坐标")
    if time_name != "time":
        ds = ds.rename({time_name: "time"})

    bounds_name = ds["time"].attrs.get("bounds")
    if bounds_name not in ds:
        bounds_name = next((x for x in ("time_bnds", "time_bounds") if x in ds), None)
    source = ds["time"]
    if bounds_name:
        bounds = ds[bounds_name]
        source = bounds.isel({next(d for d in bounds.dims if d != "time"): 0})
        print(f"  {Path(str(label)).name}: 使用 {bounds_name} 起点月份")
    else:
        print(f"  {Path(str(label)).name}: 无 time_bnds，使用原 time 月份")

    values = np.asarray(source.values)
    units = source.attrs.get("units") or ds["time"].attrs.get("units") or ds["time"].encoding.get("units")
    calendar = source.attrs.get("calendar") or ds["time"].attrs.get("calendar") or ds["time"].encoding.get("calendar") or "standard"

    if np.issubdtype(values.dtype, np.datetime64):
        text = values.astype("datetime64[M]").astype(str)
        years = np.array([int(x[:4]) for x in text])
        months = np.array([int(x[5:7]) for x in text])
    elif values.dtype == object and values.size and hasattr(values.ravel()[0], "year"):
        years = np.array([int(x.year) for x in values.ravel()])
        months = np.array([int(x.month) for x in values.ravel()])
    else:
        if not units:
            raise ValueError(f"{label} 数值时间缺少 units")
        match = re.match(r"\s*months?\s+since\s+([+-]?\d+)-(\d+)-(\d+)", str(units), re.I)
        if match:
            y0, m0 = int(match.group(1)), int(match.group(2))
            total = y0 * 12 + m0 - 1 + np.rint(values.astype(float).ravel()).astype(int)
            years, months = total // 12, total % 12 + 1
        else:
            dates = cftime.num2date(values.ravel(), units=str(units), calendar=str(calendar),
                                    only_use_cftime_datetimes=True)
            years = np.array([int(x.year) for x in dates])
            months = np.array([int(x.month) for x in dates])

    if np.any(np.diff(years * 100 + months) < 0):
        raise ValueError(f"{label} 月份顺序不是递增")
    return ds.assign_coords(year=("time", years), month=("time", months))


def read_temperature_ensemble(files, label):
    """读取20成员温度并计算异常、集合平均和sign-flip显著性。"""
    members = []
    dim_names = {
        "time": ("time", "Time", "TIME"),
        "lat": ("lat", "latitude", "Latitude", "LAT"),
        "lon": ("lon", "longitude", "Longitude", "LON"),
        "plev": ("plev", "lev", "level", "pressure", "pres", "isobaric", "isobaricInhPa"),
    }

    for i, filepath in enumerate(files, 1):
        print(f"\n========== {label} {i}/{len(files)} ==========")
        print(f"处理温度成员: {filepath.name}")
        ds = add_year_month(xr.open_dataset(filepath, decode_times=False, mask_and_scale=True), filepath)

        var = next((x for x in TEMP_NAMES if x in ds.data_vars), None)
        if var is None:
            for name, da0 in ds.data_vars.items():
                info = (str(da0.attrs.get("standard_name", "")) + " " +
                        str(da0.attrs.get("long_name", ""))).lower()
                if "temperature" in info:
                    var = name
                    break
        if var is None:
            ds.close()
            raise KeyError(f"{filepath.name} 找不到 temperature 变量")

        da = ds[var]
        rename = {}
        for target in ("time", "lat", "lon", "plev"):
            old = next((x for x in dim_names[target] if x in da.dims), None)
            if old is None:
                ds.close()
                raise KeyError(f"{filepath.name} 找不到 {target} 维")
            if old != target:
                rename[old] = target
        da = da.rename(rename)

        for dim in list(da.dims):
            if dim not in ("time", "lat", "lon", "plev"):
                if da.sizes[dim] == 1:
                    da = da.isel({dim: 0}, drop=True)
                else:
                    ds.close()
                    raise ValueError(f"{filepath.name} 额外维度 {dim}={da.sizes[dim]}")

        da = da.assign_coords(lon=np.mod(da.lon, 360.0)).sortby("lon").sortby("lat")
        _, keep = np.unique(np.round(da.lon.values, 8), return_index=True)
        if len(keep) != da.sizes["lon"]:
            da = da.isel(lon=np.sort(keep))

        pressure = np.asarray(da.plev.values, float)
        if str(da.plev.attrs.get("units", "")).lower() == "pa" or np.nanmax(np.abs(pressure)) > 2000:
            pressure = pressure / 100.0
        da = da.assign_coords(plev=pressure).sortby("plev")
        da = da.sel(lat=slice(LAT_MIN, LAT_MAX), plev=slice(PRESSURE_TOP, PRESSURE_BOTTOM))

        section = da.sel(lon=slice(SECTION_LON_MIN, SECTION_LON_MAX)).mean("lon", skipna=True).sortby("lat")
        section_lat = np.asarray(section.lat.values, float)
        section = section.rename({"lat": "section"}).assign_coords(
            section=np.arange(len(section_lat), dtype=int),
            section_lat=("section", section_lat),
            section_lon=("section", np.full(len(section_lat), SECTION_LONGITUDE)),
        )

        annual_list, used_years = [], []
        for year in np.unique(section.year.values.astype(int)):
            oct_data = section.where((section.year == year) & (section.month == MONTHS[0]), drop=True)
            nov_data = section.where((section.year == year) & (section.month == MONTHS[1]), drop=True)
            if oct_data.sizes.get("time", 0) == 0 or nov_data.sizes.get("time", 0) == 0:
                continue
            annual_list.append(((oct_data.mean("time", skipna=True) +
                                 nov_data.mean("time", skipna=True)) / 2.0).expand_dims(year=[int(year)]))
            used_years.append(int(year))

        if not annual_list:
            ds.close()
            raise ValueError(f"{filepath.name} 没有完整的10—11月记录")
        print(f"  {filepath.name}: {used_years[0]}—{used_years[-1]}，共 {len(used_years)} 年")
        annual = xr.concat(annual_list, dim="year", coords="minimal",
                           compat="override", combine_attrs="drop").sortby("year")

        if not DETREND_TEMPERATURE:
            print("  温度不去趋势：直接使用原始10—11月年度序列计算异常。")
        else:
            x = xr.DataArray(annual.year.values.astype(float), coords={"year": annual.year}, dims="year")
            valid = np.isfinite(annual)
            n = valid.sum("year")
            x_mean = x.where(valid).sum("year", skipna=True) / n
            y_mean = annual.where(valid).sum("year", skipna=True) / n
            num = ((x - x_mean) * (annual - y_mean)).where(valid).sum("year", skipna=True)
            den = ((x - x_mean) ** 2).where(valid).sum("year", skipna=True)
            slope = (num / den).where((n >= MIN_DETREND_VALID_YEARS) & (den > 0))
            annual = annual - slope * (x - x_mean) if DETREND_KEEP_MEAN else annual - (y_mean + slope * (x - x_mean))

        available = set(annual.year.values.astype(int).tolist())
        missing = [int(x) for x in np.r_[TARGET_YEARS, BASELINE_YEARS] if int(x) not in available]
        if missing:
            ds.close()
            raise ValueError(f"{filepath.name} 缺少年份: {missing}")

        anomaly = annual.sel(year=TARGET_YEARS).mean("year") - annual.sel(year=BASELINE_YEARS).mean("year")
        members.append(anomaly.transpose("plev", "section").load().expand_dims(member=[i]))
        ds.close()

    all_members = xr.concat(members, dim="member", join="exact", coords="minimal",
                            compat="override", combine_attrs="drop").sortby("plev").sortby("section")
    p0 = all_members.plev.values.astype(float)
    x0 = all_members.section.values.astype(float)
    fine_p = np.exp(np.linspace(np.log(p0.min()), np.log(p0.max()),
                                (len(p0) - 1) * VERTICAL_INTERPOLATION_FACTOR + 1))
    fine_x = np.linspace(x0.min(), x0.max(), (len(x0) - 1) * SECTION_INTERPOLATION_FACTOR + 1)

    work = all_members.assign_coords(log_plev=("plev", np.log(p0))).swap_dims({"plev": "log_plev"})
    if "plev" in work.coords:
        work = work.drop_vars("plev")
    all_members = work.interp(log_plev=np.log(fine_p), section=fine_x, method="linear").rename({"log_plev": "plev"})
    all_members = all_members.assign_coords(
        plev=("plev", fine_p), section=("section", fine_x),
        section_lat=("section", members[0].section_lat.interp(section=fine_x).values),
        section_lon=("section", members[0].section_lon.interp(section=fine_x, method="nearest").values),
    ).transpose("member", "plev", "section")

    mean = all_members.mean("member", skipna=True)
    mean.name = "temperature_anomaly"
    mean.coords["section_lat"] = all_members.section_lat
    mean.coords["section_lon"] = all_members.section_lon

    values = np.asarray(all_members.values, float)
    n_member, shape = values.shape[0], values.shape[1:]
    flat = values.reshape(n_member, -1)
    finite = np.isfinite(flat)
    n_valid = finite.sum(axis=0)
    filled = np.where(finite, flat, 0.0)
    observed = filled.sum(axis=0) / np.where(n_valid > 0, n_valid, 1)
    valid_point = n_valid >= EXPECTED_MEMBERS
    observed[~valid_point] = np.nan

    rng = np.random.default_rng(MONTE_CARLO_RANDOM_SEED)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(MONTE_CARLO_N_RANDOM, n_member))
    count = np.zeros(flat.shape[1], dtype=np.int64)
    for start in range(0, flat.shape[1], MONTE_CARLO_CHUNK_POINTS):
        stop = min(start + MONTE_CARLO_CHUNK_POINTS, flat.shape[1])
        random_mean = (signs @ filled[:, start:stop]) / np.where(
            n_valid[start:stop] > 0, n_valid[start:stop], 1)[None, :]
        extreme = np.abs(random_mean) >= np.abs(observed[start:stop])[None, :]
        extreme[:, ~valid_point[start:stop]] = False
        count[start:stop] = extreme.sum(axis=0)

    p_value = (count + 1.0) / (MONTE_CARLO_N_RANDOM + 1.0)
    p_value[~valid_point] = np.nan
    significant = (p_value < SIGNIFICANCE_ALPHA) & valid_point
    coords = {"plev": all_members.plev, "section": all_members.section}

    obs = xr.DataArray(observed.reshape(shape), coords=coords, dims=("plev", "section"))
    p = xr.DataArray(p_value.reshape(shape), coords=coords, dims=("plev", "section"))
    sig = xr.DataArray(significant.reshape(shape).astype(np.int8), coords=coords, dims=("plev", "section"))
    count_da = xr.DataArray(n_valid.reshape(shape).astype(np.int16), coords=coords, dims=("plev", "section"))
    for da in (obs, p, sig, count_da):
        da.coords["section_lat"] = all_members.section_lat
        da.coords["section_lon"] = all_members.section_lon

    print(f"\n{label}温度异常范围: {float(mean.min(skipna=True)):.3f} 至 {float(mean.max(skipna=True)):.3f} °C")
    print(f"显著格点: {int(sig.sum())}/{sig.size}，显著性水平={SIGNIFICANCE_LEVEL:.0%}")
    return mean, obs, p, sig, count_da


def read_sic_anomaly(filepath, label):
    """读取海冰并计算120–240°E平均的Oct-Nov异常。"""
    print(f"\n读取 {label} 海冰: {filepath}")
    ds = add_year_month(xr.open_dataset(filepath, decode_times=False, mask_and_scale=True), filepath)

    var = next((x for x in SIC_NAMES if x in ds.data_vars), None)
    if var is None:
        for name, da0 in ds.data_vars.items():
            info = (str(da0.attrs.get("standard_name", "")) + " " +
                    str(da0.attrs.get("long_name", ""))).lower()
            if "sea ice" in info or "sea_ice" in info or "ice concentration" in info:
                var = name
                break
    if var is None:
        ds.close()
        raise KeyError(f"{filepath.name} 找不到 sea_ice 变量")

    sic = ds[var]
    dim_names = {"time": ("time", "Time", "TIME"),
                 "lat": ("lat", "latitude", "Latitude", "LAT"),
                 "lon": ("lon", "longitude", "Longitude", "LON")}
    rename = {}
    for target in ("time", "lat", "lon"):
        old = next((x for x in dim_names[target] if x in sic.dims), None)
        if old is None:
            ds.close()
            raise KeyError(f"{filepath.name} 找不到 {target} 维")
        if old != target:
            rename[old] = target
    sic = sic.rename(rename)

    for dim in list(sic.dims):
        if dim not in ("time", "lat", "lon"):
            if sic.sizes[dim] == 1:
                sic = sic.isel({dim: 0}, drop=True)
            else:
                ds.close()
                raise ValueError(f"{filepath.name} 额外维度 {dim}={sic.sizes[dim]}")

    sic = sic.assign_coords(lon=np.mod(sic.lon, 360.0)).sortby("lon").sortby("lat")
    _, keep = np.unique(np.round(sic.lon.values, 8), return_index=True)
    if len(keep) != sic.sizes["lon"]:
        sic = sic.isel(lon=np.sort(keep))
    sic = sic.sel(lat=slice(LAT_MIN, LAT_MAX))

    units = str(sic.attrs.get("units", "")).lower()
    if "%" not in units and "percent" not in units and float(sic.isel(time=0).max(skipna=True)) <= 1.5:
        sic = sic * 100.0
    sic.attrs["units"] = "%"

    section = sic.sel(lon=slice(SECTION_LON_MIN, SECTION_LON_MAX)).mean("lon", skipna=True).sortby("lat")
    section_lat = section.lat.values.astype(float)
    section = section.rename({"lat": "section"}).assign_coords(
        section=np.arange(len(section_lat)),
        section_lat=("section", section_lat),
        section_lon=("section", np.full(len(section_lat), SECTION_LONGITUDE)),
    )

    annual_list, used_years = [], []
    for year in np.unique(section.year.values.astype(int)):
        oct_data = section.where((section.year == year) & (section.month == MONTHS[0]), drop=True)
        nov_data = section.where((section.year == year) & (section.month == MONTHS[1]), drop=True)
        if oct_data.sizes.get("time", 0) == 0 or nov_data.sizes.get("time", 0) == 0:
            continue
        annual_list.append(((oct_data.mean("time", skipna=True) +
                             nov_data.mean("time", skipna=True)) / 2.0).expand_dims(year=[int(year)]))
        used_years.append(int(year))

    if not annual_list:
        ds.close()
        raise ValueError(f"{label} 没有完整的10—11月记录")
    print(f"  {label}: {used_years[0]}—{used_years[-1]}，共 {len(used_years)} 年")
    annual = xr.concat(annual_list, dim="year", coords="minimal",
                       compat="override", combine_attrs="drop").sortby("year")
    anomaly = annual.sel(year=TARGET_YEARS).mean("year") - annual.sel(year=BASELINE_YEARS).mean("year")
    anomaly = anomaly.transpose("section").load()
    anomaly.name = "sea_ice_concentration_anomaly"
    ds.close()
    print(f"{label}海冰异常范围: {float(anomaly.min(skipna=True)):.3f} 至 {float(anomaly.max(skipna=True)):.3f} %")
    return anomaly


# ============================================================
# 3. 顺序计算
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("\n月份规则：存在 time_bnds/time_bounds 时统一使用区间起点。")
print(f"显著性检验：双侧 Monte Carlo sign-flip，{SIGNIFICANCE_LEVEL:.0%}，N={MONTE_CARLO_N_RANDOM}")

natural_key = lambda p: [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]

rec_files = None
for pattern in REC_PATTERNS:
    files = sorted(TEMP_DIR.glob(pattern), key=natural_key)
    if len(files) == EXPECTED_MEMBERS:
        rec_files = files
        print(f"\n重建温度成员: {pattern}")
        for i, file in enumerate(files, 1):
            print(f"  {i:02d}. {file.name}")
        break
if rec_files is None:
    raise RuntimeError(f"重建温度成员没有找到恰好 {EXPECTED_MEMBERS} 个成员，请检查 {TEMP_DIR}")

had_files = None
for pattern in HAD_PATTERNS:
    files = sorted(TEMP_DIR.glob(pattern), key=natural_key)
    if len(files) == EXPECTED_MEMBERS:
        had_files = files
        print(f"\nHadISST温度成员: {pattern}")
        for i, file in enumerate(files, 1):
            print(f"  {i:02d}. {file.name}")
        break
if had_files is None:
    raise RuntimeError(f"HadISST温度成员没有找到恰好 {EXPECTED_MEMBERS} 个成员，请检查 {TEMP_DIR}")

rec_temp, rec_obs, rec_p, rec_sig, rec_count = read_temperature_ensemble(rec_files, "CAM-IAPICE")
had_temp, had_obs, had_p, had_sig, had_count = read_temperature_ensemble(had_files, "CAM-HadISST")
rec_sic = read_sic_anomaly(REC_SIC_FILE, "IAPICE")
had_sic = read_sic_anomaly(HAD_SIC_FILE, "HadISST")


# ============================================================
# 4. 保存计算结果
# ============================================================
save_items = {
    "reconstructed_temperature_anomaly": (rec_temp, "rec_temp"),
    "reconstructed_temperature_monte_carlo_observed_mean": (rec_obs, "rec_temp"),
    "reconstructed_temperature_monte_carlo_p_value": (rec_p, "rec_temp"),
    "reconstructed_temperature_significant_95pct": (rec_sig, "rec_temp"),
    "reconstructed_temperature_valid_member_count": (rec_count, "rec_temp"),
    "reconstructed_sea_ice_concentration_anomaly": (rec_sic, "rec_sic"),
    "hadisst_temperature_anomaly": (had_temp, "had_temp"),
    "hadisst_temperature_monte_carlo_observed_mean": (had_obs, "had_temp"),
    "hadisst_temperature_monte_carlo_p_value": (had_p, "had_temp"),
    "hadisst_temperature_significant_95pct": (had_sig, "had_temp"),
    "hadisst_temperature_valid_member_count": (had_count, "had_temp"),
    "hadisst_sea_ice_concentration_anomaly": (had_sic, "had_sic"),
}
save_data = {}
for name, (da, prefix) in save_items.items():
    rename = {}
    for old, new in (("plev", f"{prefix}_plev"), ("section", f"{prefix}_section"),
                     ("section_lat", f"{prefix}_section_lat"), ("section_lon", f"{prefix}_section_lon")):
        if old in da.dims or old in da.coords:
            rename[old] = new
    save_data[name] = da.rename(rename)

result = xr.Dataset(save_data)
result.attrs.update({
    "target_period": "October-November 1940-1942",
    "baseline_period": "October-November 1971-2000",
    "section_definition": "120-240E longitude-mean meridional section from 66.5N to 90N",
    "temperature_detrended_before_anomaly": str(DETREND_TEMPERATURE),
    "temperature_significance_test": f"two-sided Monte Carlo sign-flipping, level={SIGNIFICANCE_LEVEL}",
})
result.to_netcdf(OUTPUT_NC)
print("结果已保存:", OUTPUT_NC)


# ============================================================
# 5. 绘图
# ============================================================
levels = np.round(np.arange(TEMP_VMIN, TEMP_VMAX + TEMP_COLOR_INTERVAL * 0.5, TEMP_COLOR_INTERVAL), 10)
levels = levels[~np.isclose(levels, 0.0)]
base = LinearSegmentedColormap.from_list("figure4d_sat_base", TEMP_COLOR_ANCHORS, N=256)
colors = base(np.linspace(0.0, 1.0, len(levels) - 1))
colors[(levels[:-1] < 0.0) & (levels[1:] > 0.0), :3] = [1.0, 1.0, 1.0]
cmap = ListedColormap(colors, name="figure4d_sat_center_white")
cmap.set_bad("white")

sic_limit = max(
    np.ceil(float(np.nanmax(np.abs(rec_sic.values))) / SIC_TICK_INTERVAL) * SIC_TICK_INTERVAL,
    np.ceil(float(np.nanmax(np.abs(had_sic.values))) / SIC_TICK_INTERVAL) * SIC_TICK_INTERVAL,
    SIC_TICK_INTERVAL,
)
finite = np.r_[rec_sic.values.ravel(), (-had_sic.values).ravel()]
finite = finite[np.isfinite(finite)]
maximum_up = max(0.0, float(np.nanmax(finite))) if finite.size else 0.0
sic_top = max(sic_limit * SIC_POSITIVE_HEADROOM_RATIO, maximum_up * 1.10)

fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor="white")
outer = GridSpec(1, 2, figure=fig, wspace=PANEL_HORIZONTAL_WSPACE)
left = outer[0, 0].subgridspec(2, 1, height_ratios=TEMP_SIC_HEIGHT_RATIOS, hspace=0.0)
right = outer[0, 1].subgridspec(2, 1, height_ratios=TEMP_SIC_HEIGHT_RATIOS, hspace=0.0)

rec_temp_ax = fig.add_subplot(left[0, 0])
rec_sic_ax = fig.add_subplot(left[1, 0], sharex=rec_temp_ax)
had_temp_ax = fig.add_subplot(right[0, 0], sharey=rec_temp_ax)
had_sic_ax = fig.add_subplot(right[1, 0], sharex=had_temp_ax)
fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, bottom=FIG_BOTTOM, top=FIG_TOP,
                    wspace=PANEL_HORIZONTAL_WSPACE)

for temp_ax, sic_ax in ((rec_temp_ax, rec_sic_ax), (had_temp_ax, had_sic_ax)):
    temp_pos, sic_pos = temp_ax.get_position(), sic_ax.get_position()
    new_bottom = max(0.04, sic_pos.y0 - min(SIC_PANEL_EXTRA_DOWN, 0.030))
    sic_ax.set_position([sic_pos.x0, new_bottom, sic_pos.width, temp_pos.y0 - new_bottom])

panel_data = [
    (rec_temp_ax, rec_sic_ax, rec_temp, rec_sig, rec_sic, "CAM-IAPICE", "IAPICE", "(a)", False),
    (had_temp_ax, had_sic_ax, had_temp, had_sig, had_sic, "CAM-HadISST", "HadISST", "(b)", True),
]
filled_rec = None

for temp_ax, sic_ax, temp, sig, sic, experiment, sic_name, panel, flip_sic in panel_data:
    x = temp.section.values.astype(float)
    lat = temp.section_lat.values.astype(float)
    pressure = temp.plev.values.astype(float)
    temp_values = np.asarray(temp.values, float)
    sig_values = np.asarray(sig.values, bool)
    sic_values = np.asarray(sic.values, float)
    sic_lat = sic.section_lat.values.astype(float)

    order = np.argsort(lat)
    sic_x = np.interp(sic_lat, lat[order], x[order])
    spacing = np.diff(sic_x)
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    bar_width = SIC_BAR_WIDTH_FACTOR * float(np.nanmedian(spacing)) if spacing.size else 0.8
    tick_pos = [float(x[np.nanargmin(np.abs(lat - value))]) for value in LAT_TICKS]
    pole_x = float(x[np.nanargmax(lat)])
    x_left, x_right = float(np.nanmin(x)), float(np.nanmax(x))

    filled = temp_ax.contourf(x, pressure, np.ma.masked_invalid(temp_values),
                              levels=levels, cmap=cmap, extend="both")
    for collection in filled.collections:
        collection.set_rasterized(True)
    if panel == "(a)":
        filled_rec = filled

    sig2d = (sig_values & np.isfinite(temp_values)).astype(float)
    if np.any(sig2d >= 0.5):
        cs = temp_ax.contour(x, pressure, sig2d, levels=[0.5],
                             colors=SIGNIFICANCE_CONTOUR_COLOR,
                             linewidths=SIGNIFICANCE_CONTOUR_LINEWIDTH,
                             linestyles="-", zorder=SIGNIFICANCE_CONTOUR_ZORDER,
                             alpha=SIGNIFICANCE_CONTOUR_OPACITY)
        for collection in getattr(cs, "collections", [cs]):
            collection.set_path_effects([
                pe.Stroke(linewidth=SIGNIFICANCE_CONTOUR_LINEWIDTH + 0.9, foreground="white"),
                pe.Normal()
            ])

    x_edges = (np.array([x[0] - 0.5, x[0] + 0.5]) if x.size == 1 else
               np.r_[x[0] - 0.5 * (x[1] - x[0]), 0.5 * (x[:-1] + x[1:]),
                     x[-1] + 0.5 * (x[-1] - x[-2])])
    p_edges = (np.array([pressure[0] - 0.5, pressure[0] + 0.5]) if pressure.size == 1 else
               np.r_[pressure[0] - 0.5 * (pressure[1] - pressure[0]),
                     0.5 * (pressure[:-1] + pressure[1:]),
                     pressure[-1] + 0.5 * (pressure[-1] - pressure[-2])])

    edge_effect = [pe.Stroke(linewidth=SIGNIFICANCE_CONTOUR_LINEWIDTH + 0.9, foreground="white"), pe.Normal()]
    for edge_mask, p_value in ((sig2d[-1, :] >= 0.5, PRESSURE_BOTTOM),
                               (sig2d[0, :] >= 0.5, PRESSURE_TOP)):
        i = 0
        while i < edge_mask.size:
            if not edge_mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < edge_mask.size and edge_mask[j + 1]:
                j += 1
            temp_ax.plot([max(float(x_edges[i]), x_left), min(float(x_edges[j + 1]), x_right)],
                         [p_value, p_value], color=SIGNIFICANCE_CONTOUR_COLOR,
                         linewidth=SIGNIFICANCE_CONTOUR_LINEWIDTH, linestyle="-",
                         solid_capstyle="butt", zorder=SIGNIFICANCE_CONTOUR_ZORDER + 2,
                         clip_on=True, alpha=SIGNIFICANCE_CONTOUR_OPACITY,
                         path_effects=edge_effect)
            i = j + 1

    for edge_mask, x_value in ((sig2d[:, 0] >= 0.5, x_left), (sig2d[:, -1] >= 0.5, x_right)):
        i = 0
        while i < edge_mask.size:
            if not edge_mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < edge_mask.size and edge_mask[j + 1]:
                j += 1
            temp_ax.plot([x_value, x_value],
                         [max(float(p_edges[i]), PRESSURE_TOP),
                          min(float(p_edges[j + 1]), PRESSURE_BOTTOM)],
                         color=SIGNIFICANCE_CONTOUR_COLOR,
                         linewidth=SIGNIFICANCE_CONTOUR_LINEWIDTH, linestyle="-",
                         solid_capstyle="butt", zorder=SIGNIFICANCE_CONTOUR_ZORDER + 2,
                         clip_on=True, alpha=SIGNIFICANCE_CONTOUR_OPACITY,
                         path_effects=edge_effect)
            i = j + 1

    temp_ax.set_yscale("log")
    temp_ax.set_ylim(PRESSURE_BOTTOM, PRESSURE_TOP)
    temp_ax.set_yticks(PRESSURE_TICKS)
    temp_ax.set_yticklabels([f"{v:g}" for v in PRESSURE_TICKS])
    temp_ax.yaxis.set_minor_formatter(NullFormatter())
    temp_ax.set_ylabel("Pressure Level (hPa)", fontsize=MEDIUM_FONT_SIZE, fontweight="normal")
    temp_ax.tick_params(axis="y", which="major", labelsize=SMALL_FONT_SIZE, direction="out",
                        length=TICK_LENGTH, width=TICK_WIDTH, left=True, labelleft=True,
                        right=False, labelright=False, pad=2)
    temp_ax.tick_params(axis="y", which="minor", left=False, right=False, length=0)
    temp_ax.tick_params(axis="x", which="both", bottom=False, top=False,
                        labelbottom=False, labeltop=False, length=0)
    temp_ax.axvline(pole_x, color="black", lw=POLE_LINEWIDTH, ls="--")

    if panel == "(a)":
        box_effect = [pe.Stroke(linewidth=BOX_LINEWIDTH + 0.9, foreground="white"), pe.Normal()]
        for xx, yy in (([x_left, x_right], [BOX_PRESSURE_TOP, BOX_PRESSURE_TOP]),
                       ([x_left, x_right], [BOX_PRESSURE_BOTTOM, BOX_PRESSURE_BOTTOM]),
                       ([x_left, x_left], [BOX_PRESSURE_TOP, BOX_PRESSURE_BOTTOM]),
                       ([x_right, x_right], [BOX_PRESSURE_TOP, BOX_PRESSURE_BOTTOM])):
            temp_ax.plot(xx, yy, color=BOX_COLOR, linewidth=BOX_LINEWIDTH,
                         solid_capstyle="butt", zorder=50, clip_on=True,
                         alpha=BOX_OPACITY, path_effects=box_effect)

    temp_ax.text(0.0, 1.02, panel, transform=temp_ax.transAxes, ha="left", va="bottom",
                 fontsize=LARGE_FONT_SIZE, fontweight="bold", clip_on=False, zorder=100)
    temp_ax.text(0.985, 0.975, experiment, transform=temp_ax.transAxes, ha="right", va="top",
                 fontsize=MEDIUM_FONT_SIZE, fontweight="normal", clip_on=False,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.5))

    plot_sic = -sic_values if flip_sic else sic_values
    mean_sic = np.nanmean(sic_values)
    bar_color = SIC_NEG_COLOR if mean_sic < 0 else SIC_POS_COLOR
    sic_ax.bar(sic_x, plot_sic, color=bar_color, width=bar_width, align="center", edgecolor="none")
    sic_ax.axhline(0, color="black", lw=AXIS_LINEWIDTH)
    sic_ax.axvline(pole_x, color="black", lw=POLE_LINEWIDTH, ls="--")
    sic_ax.set_ylim(-sic_limit, sic_top)
    temp_ax.set_xlim(x_left, x_right)
    sic_ax.set_xlim(x_left, x_right)
    sic_ax.set_xticks(tick_pos)
    sic_ax.set_xticklabels([f"{v}°N" for v in LAT_TICKS],
                           fontsize=SMALL_FONT_SIZE, fontweight="normal")
    sic_ax.set_ylabel("Ice conc. (%)", fontsize=MEDIUM_FONT_SIZE, fontweight="normal")
    sic_ax.yaxis.set_label_position("right")
    sic_ax.yaxis.tick_right()

    sic_ticks = np.arange(-sic_limit, SIC_TICK_INTERVAL * 0.5, SIC_TICK_INTERVAL)
    sic_ax.set_yticks(sic_ticks)
    if flip_sic:
        sic_labels = ["0" if np.isclose(v, 0.0) else f"{abs(v):g}" for v in sic_ticks]
    else:
        sic_labels = ["0" if np.isclose(v, 0.0) else f"−{abs(v):g}" for v in sic_ticks]
    sic_ax.set_yticklabels(sic_labels, fontsize=SMALL_FONT_SIZE, fontweight="bold")
    sic_ax.tick_params(axis="x", which="major", labelsize=SMALL_FONT_SIZE, direction="out",
                       length=TICK_LENGTH, width=TICK_WIDTH, bottom=True, labelbottom=True,
                       top=False, labeltop=False, pad=2)
    sic_ax.tick_params(axis="x", which="minor", bottom=False, top=False, length=0)

    y_tick_color = SIC_NEG_COLOR if mean_sic < 0 else SIC_POS_COLOR
    sic_ax.tick_params(axis="y", which="major", labelsize=SMALL_FONT_SIZE, direction="out",
                       length=TICK_LENGTH, width=TICK_WIDTH, left=False, labelleft=False,
                       right=True, labelright=True, pad=2, colors=y_tick_color)
    sic_ax.tick_params(axis="y", which="minor", left=False, right=False,
                       length=0, colors=y_tick_color)
    for tick in sic_ax.get_yticklabels():
        tick.set_color(y_tick_color)
        tick.set_fontweight("bold")

    sic_ax.text(0.985, 0.08, sic_name, transform=sic_ax.transAxes, ha="right", va="bottom",
                fontsize=MEDIUM_FONT_SIZE, fontweight="normal", clip_on=False,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.5))

    for axis in (temp_ax, sic_ax):
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(AXIS_LINEWIDTH)
            spine.set_color("#333333")
        for tick in axis.get_xticklabels():
            tick.set_fontsize(SMALL_FONT_SIZE)
            tick.set_fontweight("normal")
        for tick in axis.get_yticklabels():
            tick.set_fontsize(SMALL_FONT_SIZE)
            tick.set_fontweight("bold" if axis is sic_ax else "normal")

rec_temp_ax.tick_params(labelleft=True)
had_temp_ax.tick_params(labelleft=False)
had_temp_ax.set_ylabel("")
rec_sic_ax.yaxis.set_label_position("right")
rec_sic_ax.yaxis.tick_right()
had_sic_ax.yaxis.set_label_position("right")
had_sic_ax.yaxis.tick_right()

rec_pos, had_pos = rec_temp_ax.get_position(), had_temp_ax.get_position()
temp_cax = fig.add_axes([
    had_pos.x1 + CBAR_X_GAP,
    min(rec_pos.y0, had_pos.y0) + had_pos.height * CBAR_BOTTOM_RATIO,
    TEMP_CBAR_WIDTH_INCH / fig.get_figwidth(),
    had_pos.height * CBAR_HEIGHT_RATIO,
])
temp_cb = fig.colorbar(filled_rec, cax=temp_cax, orientation="vertical", ticks=levels, extend="both")
temp_cb.set_label(TEMP_CBAR_LABEL, fontsize=MEDIUM_FONT_SIZE, fontweight="normal", labelpad=3)
temp_cb.ax.minorticks_off()
temp_cb.ax.tick_params(axis="y", which="major", labelsize=SMALL_FONT_SIZE, direction="out",
                       length=TICK_LENGTH, width=0.5, pad=1.5, left=False, labelleft=False,
                       right=True, labelright=True)
temp_cb.outline.set_linewidth(0.5)

fig.canvas.draw()
fig.savefig(OUTPUT_SVG, facecolor="white")
fig.savefig(OUTPUT_PDF, facecolor="white")
fig.savefig(OUTPUT_FIGURE, dpi=FIGURE_DPI, facecolor="white")

print("SVG已保存:", OUTPUT_SVG)
print("PDF已保存:", OUTPUT_PDF)
print("PNG已保存:", OUTPUT_FIGURE)

if SHOW_FIGURE:
    plt.show()
plt.close(fig)
print("全部处理完成。")
