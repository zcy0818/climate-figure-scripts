# -*- coding: utf-8 -*-
"""Shared low-level utilities for Figure 4 scripts.

Normally you only edit Fig4a.py-Fig4e.py, Fig4.py, Fig4m.py or Fig4exp.py.
This file only keeps repeated data-processing and plotting helpers in one place.
"""
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib.quiver import QuiverKey


# -----------------------------------------------------------------------------
# Time / coordinates / seasons
# -----------------------------------------------------------------------------
def fix_time(ds):
    """Use time-bounds starts when available; otherwise retain the CAM month fix."""
    if "time" not in ds.coords:
        return ds
    names = [str(ds["time"].attrs.get("bounds", "")).strip(), "time_bnds", "time_bounds"]
    bname = next((x for x in names if x and x in ds.variables), None)
    if bname:
        b = ds[bname]
        bdim = next(d for d in b.dims if d != "time")
        starts = np.asarray(b.isel({bdim: 0}).values)
        first = starts.ravel()[0]
        if not (np.issubdtype(starts.dtype, np.datetime64) or hasattr(first, "year")):
            import cftime
            units = b.attrs.get("units") or ds["time"].encoding.get("units") or ds["time"].attrs.get("units")
            cal = b.attrs.get("calendar") or ds["time"].encoding.get("calendar") or ds["time"].attrs.get("calendar", "standard")
            starts = np.asarray(
                cftime.num2date(starts.ravel(), units=units, calendar=cal, only_use_cftime_datetimes=True),
                dtype=object,
            ).reshape(starts.shape)
        return ds.assign_coords(time=("time", starts.reshape(-1)))

    t = ds["time"].values
    if len(t) >= 2 and hasattr(t[0], "month") and int(t[0].month) == 2 and int(t[-1].month) == 1:
        new = []
        for tt in t:
            y, m = int(tt.year), int(tt.month)
            y, m = (y - 1, 12) if m == 1 else (y, m - 1)
            try:
                new.append(type(tt)(y, m, 1, has_year_zero=getattr(tt, "has_year_zero", True)))
            except TypeError:
                new.append(type(tt)(y, m, 1))
        ds = ds.assign_coords(time=("time", new))
    return ds


def prep_field(da, extent=(-180, 180, 30, 90), margin=0.0):
    rename = {
        a: b for a, b in {
            "latitude": "lat", "longitude": "lon", "Latitude": "lat", "Longitude": "lon",
            "grid_latitude": "lat", "grid_longitude": "lon", "nav_lat": "lat", "nav_lon": "lon",
            "y": "lat", "x": "lon",
        }.items() if a in da.dims or a in da.coords
    }
    if rename:
        da = da.rename(rename)
    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(f"Field has no lat/lon coordinates: dims={da.dims}")
    if float(da.lat.values[0]) > float(da.lat.values[-1]):
        da = da.sortby("lat")
    da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon")
    lon1, lon2, lat1, lat2 = extent
    return da.sel(lon=slice(lon1 - margin, lon2 + margin), lat=slice(lat1 - margin, lat2 + margin))


def month_offsets(months):
    months = [int(m) for m in months]
    if not months or any(m < 1 or m > 12 for m in months) or len(set(months)) != len(months):
        raise ValueError("MONTHS must contain unique month numbers from 1 to 12.")
    offsets = {months[0]: 0}
    year_offset = 0
    previous = months[0]
    for month in months[1:]:
        if month < previous:
            year_offset += 1
        offsets[month] = year_offset
        previous = month
    return offsets


def seasonal_mean(da, months, crossyear_months=(1, 2, 3), require_complete=False, offsets=None):
    months = [int(m) for m in months]
    selected = da.where(da.time.dt.month.isin(months), drop=True)
    mo = selected.time.dt.month.values.astype(int)
    yr = selected.time.dt.year.values.astype(int)

    if require_complete:
        offsets = month_offsets(months) if offsets is None else offsets
        sy = np.array([y - offsets[int(m)] for y, m in zip(yr, mo)], dtype=int)
        required = set(months)
        complete = [int(y) for y in np.unique(sy) if required.issubset(set(mo[sy == y].tolist()))]
        if not complete:
            raise ValueError(f"No complete seasons for MONTHS={months}.")
        keep = np.isin(sy, complete)
        selected = selected.isel(time=np.where(keep)[0])
        sy = sy[keep]
    else:
        sy = yr.copy()
        sy[np.isin(mo, np.asarray(crossyear_months, int))] -= 1

    out = selected.assign_coords(season_year=("time", sy)).groupby("season_year").mean("time", skipna=True)
    return out.rename({"season_year": "time"}).assign_coords(time=out.season_year.values.astype(int)).sortby("time")


def detrend_field(da):
    x = da.time.values.astype(float)
    def one(y):
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        if ok.sum() < 3:
            return np.full_like(y, np.nan)
        slope = np.polyfit(x[ok], y[ok], 1)[0]
        z = y.copy()
        z[ok] -= slope * (x[ok] - x[ok].mean())
        return z
    return xr.apply_ufunc(
        one, da, input_core_dims=[["time"]], output_core_dims=[["time"]],
        vectorize=True, dask="parallelized", output_dtypes=[float],
    ).transpose(*da.dims)


def member_file(directory, experiment, member, suffix):
    prefix = f"{experiment}_1deg"
    names = ([f"{prefix}{suffix}", f"{prefix}_0{suffix}", f"{prefix}_00{suffix}"]
             if member == 0 else [f"{prefix}_{member}{suffix}", f"{prefix}_{member:02d}{suffix}"])
    for name in names:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("\n".join(os.path.join(directory, n) for n in names))


def select_level(ds, candidates, hpa, height=False):
    name = next((v for v in candidates if v in ds.data_vars), None)
    if name is None and height:
        possible = []
        for v, field in ds.data_vars.items():
            dims = {str(d).lower() for d in field.dims}
            coords = {str(c).lower() for c in field.coords}
            text = f"{v} {field.attrs.get('long_name','')} {field.attrs.get('standard_name','')} {field.attrs.get('units','')}".lower()
            horizontal = ("lat" in dims or "latitude" in dims) and ("lon" in dims or "longitude" in dims)
            has_lev = any(x in dims or x in coords for x in ["plev","lev","level","p","isobaricinhpa","pressure"])
            looks_height = v.lower() in {"z3","z","zg","hgt","height","geopotential"} or "geopotential" in text or "height" in text or "gpm" in text
            not_height = v.upper() in {"T","TS","U","V","PS","PSL"} or "temperature" in text or "wind" in text
            if horizontal and has_lev and looks_height and not not_height:
                possible.append(v)
        if len(possible) == 1:
            name = possible[0]
    if name is None:
        raise KeyError(f"Variable not found; candidates={candidates}; available={list(ds.data_vars)}")
    da = ds[name]
    levname = next((x for x in ["plev", "level", "lev", "p", "isobaricInhPa", "pressure"]
                    if x in da.dims or x in da.coords), None)
    if levname:
        lev = np.asarray(da[levname], float)
        units = str(da[levname].attrs.get("units", "")).lower()
        target = hpa * 100 if (("pa" in units and "hpa" not in units) or np.nanmax(np.abs(lev)) > 2000) else hpa
        da = da.isel({levname: int(np.nanargmin(np.abs(lev - target)))}, drop=True)
    da = da.astype("float32")
    if height:
        sample = da
        for d in ["time", "lat", "lon"]:
            if d in sample.dims:
                sample = sample.isel({d: slice(0, min(12 if d == "time" else 20, sample.sizes[d]))})
        if float(sample.mean(skipna=True)) > 20000:
            da = da / np.float32(9.80665)
    return da


def signflip_test(member_diff, alpha=0.1, min_n=20, n_random=10000, seed=20260625, chunk_points=2500):
    member_diff = member_diff.transpose("ensemble", ...)
    dims = [d for d in member_diff.dims if d != "ensemble"]
    shape = tuple(member_diff.sizes[d] for d in dims)
    x = np.asarray(member_diff, float).reshape(member_diff.sizes["ensemble"], -1)
    finite = np.isfinite(x)
    n = finite.sum(0)
    valid = n >= min_n
    obs = np.divide(np.nansum(x, 0), n, out=np.full(x.shape[1], np.nan), where=n > 0)
    filled = np.where(finite, x, 0.0)
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(n_random, x.shape[0]))
    count = np.zeros(x.shape[1], np.int64)
    for start in range(0, x.shape[1], chunk_points):
        stop = min(start + chunk_points, x.shape[1])
        rnd = signs @ filled[:, start:stop]
        rnd = np.divide(rnd, n[start:stop], where=n[start:stop] > 0)
        extreme = np.abs(rnd) >= np.abs(obs[start:stop])[None, :]
        extreme[:, ~valid[start:stop]] = False
        count[start:stop] = extreme.sum(0)
    sig = ((count + 1.0) / (n_random + 1.0) <= alpha) & valid
    return xr.DataArray(sig.reshape(shape), coords={d: member_diff[d] for d in dims}, dims=dims)


# -----------------------------------------------------------------------------
# WAF / map helpers
# -----------------------------------------------------------------------------
def calculate_waf(z500, ub, vb, prep, pressure_hpa=500.0, min_basic_speed=2.0,
                  earth_radius=6371000.0, omega=7.2921159e-05, gravity=9.80665,
                  plot_lat_min=30.0, plot_lat_max=75.0):
    z = prep(z500).transpose("lat", "lon")
    ub, vb = prep(ub), prep(vb)
    if ub.shape != z.shape:
        ub = ub.interp(lat=z.lat, lon=z.lon)
        vb = vb.interp(lat=z.lat, lon=z.lon)
    lat = np.asarray(z.lat, float)
    lon = np.deg2rad(np.asarray(z.lon, float))
    latr = np.deg2rad(lat)
    dl = float(np.nanmedian(np.diff(lon)))
    f = 2 * omega * np.sin(latr)
    coslat = np.cos(latr)
    zz, uu, vv = np.asarray(z, float), np.asarray(ub, float), np.asarray(vb, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = gravity * zz / f[:, None]
    peast, pwest = np.roll(psi, -1, 1), np.roll(psi, 1, 1)
    dlon = (peast - pwest) / (2 * dl)
    d2lon = (peast - 2 * psi + pwest) / dl**2
    dlat = np.gradient(psi, latr, axis=0, edge_order=2)
    d2lat = np.gradient(dlat, latr, axis=0, edge_order=2)
    d2mix = np.gradient(dlon, latr, axis=0, edge_order=2)
    speed = np.hypot(uu, vv)
    coef = pressure_hpa / 1000 * coslat[:, None] / (2 * speed)
    a2 = earth_radius**2
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        wx = coef * (uu * (dlon**2 - psi * d2lon) / (a2 * coslat[:, None]**2)
                     + vv * (dlon * dlat - psi * d2mix) / (a2 * coslat[:, None]))
        wy = coef * (uu * (dlon * dlat - psi * d2mix) / (a2 * coslat[:, None])
                     + vv * (dlat**2 - psi * d2lat) / a2)
    goodlat = (lat >= plot_lat_min) & (lat <= plot_lat_max)
    bad = (~np.isfinite(wx) | ~np.isfinite(wy) | ~np.isfinite(speed)
           | (speed < min_basic_speed) | ~goodlat[:, None])
    coords = {"lat": z.lat, "lon": z.lon}
    return (
        xr.DataArray(np.where(bad, np.nan, wx), coords=coords, dims=("lat", "lon")).astype("float32"),
        xr.DataArray(np.where(bad, np.nan, wy), coords=coords, dims=("lat", "lon")).astype("float32"),
    )


def setup_polar_map(ax, data_crs, extent=(-180, 180, 30, 90), coast_color="0.25", coast_lw=0.45,
                    grid_color="#8A9299", grid_alpha=0.22, grid_lw=0.30,
                    polar_lat_labels=(), lon_label_size=8.0, lat_label_size=8.0):
    ax.set_extent(extent, crs=data_crs)
    ax.set_facecolor("white")
    theta = np.linspace(0, 2 * np.pi, 240)
    circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * .5 + .5)
    ax.set_boundary(circle, transform=ax.transAxes)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=2)
    ax.coastlines(resolution="50m", linewidth=coast_lw, color=coast_color, zorder=4)
    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0, color="none",
                      x_inline=False, y_inline=False, rotate_labels=False, zorder=350)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(polar_lat_labels)
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlabel_style = {"size": lon_label_size, "weight": "normal"}
    gl.ylabel_style = {"size": lat_label_size, "weight": "normal"}
    gl.xpadding = gl.ypadding = 1
    gl.top_labels = gl.right_labels = gl.left_labels = False
    clip = PathPatch(circle, transform=ax.transAxes, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    for lat in np.arange(30, 91, 10):
        ax.plot(np.linspace(-180, 180, 721), np.full(721, lat), transform=data_crs,
                color=grid_color, lw=grid_lw, alpha=grid_alpha, zorder=300)
    for lon in np.arange(-180, 181, 60):
        ax.plot(np.full(361, lon), np.linspace(30, 90, 361), transform=data_crs,
                color=grid_color, lw=grid_lw, alpha=grid_alpha, zorder=300)
    coast = ax.coastlines(resolution="50m", linewidth=coast_lw, color=coast_color, zorder=305)
    try:
        coast.set_clip_path(clip)
    except Exception:
        pass
    return clip


def add_sig_contour(ax, sig, clip, data_crs, color="#4B1D6B", linewidth=1.4, zorder=525):
    data, lon = add_cyclic_point(sig.fillna(False).astype(float).values, coord=sig.lon.values,
                                 axis=sig.get_axis_num("lon"))
    if np.nanmax(data) < .5:
        return
    lon2d, lat2d = np.meshgrid(lon, sig.lat.values)
    cs = ax.contour(lon2d, lat2d, data, levels=[.5], colors=color, linewidths=linewidth,
                    transform=data_crs, zorder=zorder)
    for coll in getattr(cs, "collections", [cs]):
        coll.set_clip_path(clip)
        coll.set_path_effects([pe.Stroke(linewidth=linewidth + 0.9, foreground="white"), pe.Normal()])


def draw_jet_axis(ax, u_clim, v_clim, clip, prep, projection, data_crs,
                  lat_min=40.0, lat_max=70.0, lon_min=-180.0, lon_max=30.0,
                  continuity_penalty=0.018, color="#666666", linewidth=2.0,
                  arrow_size=10.5, smooth_points=7, zorder=540):
    u = prep(u_clim).sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    v = prep(v_clim).sel(lon=u.lon, lat=u.lat)
    speed = np.hypot(np.asarray(u, float), np.asarray(v, float))
    lats = np.asarray(u.lat, float)
    lons = np.asarray(u.lon, float)
    nlat, nlon = speed.shape
    score = np.full((nlon, nlat), -np.inf)
    back = np.full((nlon, nlat), -1, np.int32)
    score[0] = np.where(np.isfinite(speed[:, 0]), speed[:, 0], -np.inf)
    jump = continuity_penalty * (lats[:, None] - lats[None, :])**2
    for j in range(1, nlon):
        prev = score[j - 1][None, :] - jump
        idx = np.argmax(prev, axis=1)
        best = prev[np.arange(nlat), idx]
        ok = np.isfinite(speed[:, j]) & np.isfinite(best)
        score[j, ok] = speed[ok, j] + best[ok]
        back[j, ok] = idx[ok]
    path = np.full(nlon, -1, np.int32)
    path[-1] = int(np.nanargmax(score[-1]))
    for j in range(nlon - 1, 0, -1):
        path[j - 1] = back[j, path[j]]
    jet_lat = lats[path]
    kernel = np.ones(smooth_points, float)
    jet_lat = np.convolve(jet_lat, kernel, mode="same") / np.convolve(np.ones_like(jet_lat), kernel, mode="same")
    projected = projection.transform_points(data_crs, lons, jet_lat)[:, :2]
    xy = projected[np.isfinite(projected).all(1)]
    codes = np.full(len(xy), mpath.Path.LINETO, np.uint8)
    codes[0] = mpath.Path.MOVETO
    arrow = FancyArrowPatch(
        path=mpath.Path(xy, codes), arrowstyle="-|>", mutation_scale=arrow_size,
        linewidth=linewidth, color=color, facecolor=color, edgecolor=color,
        capstyle="round", joinstyle="round", zorder=zorder,
    )
    ax.add_patch(arrow)
    arrow.set_clip_path(clip)


def extend_to_plot_edges(da, extent):
    lon1, lon2, lat1, lat2 = map(float, extent)
    out = da
    def edge_slice(obj, dim, index, coord):
        edge = obj.isel({dim: index}).expand_dims({dim: [coord]})
        return edge.transpose(*obj.dims)
    if float(out.lon.values[0]) > lon1 and not np.isclose(float(out.lon.values[0]), lon1):
        out = xr.concat([edge_slice(out, "lon", 0, lon1), out], dim="lon")
    if float(out.lon.values[-1]) < lon2 and not np.isclose(float(out.lon.values[-1]), lon2):
        out = xr.concat([out, edge_slice(out, "lon", -1, lon2)], dim="lon")
    if float(out.lat.values[0]) > lat1 and not np.isclose(float(out.lat.values[0]), lat1):
        out = xr.concat([edge_slice(out, "lat", 0, lat1), out], dim="lat")
    if float(out.lat.values[-1]) < lat2 and not np.isclose(float(out.lat.values[-1]), lat2):
        out = xr.concat([out, edge_slice(out, "lat", -1, lat2)], dim="lat")
    return out.transpose(*da.dims)


def nice_reference(value):
    exp = np.floor(np.log10(value))
    frac = value / 10**exp
    return (1 if frac <= 1 else 2 if frac <= 2 else 5 if frac <= 5 else 10) * 10**exp


def remove_labels_outside(ax, labels):
    if not labels:
        return
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    path = ax.patch.get_path()
    trans = ax.patch.get_transform()
    for text in list(labels):
        if text.axes is None:
            continue
        b = text.get_window_extent(renderer=renderer)
        corners = np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]])
        if not np.all(path.contains_points(corners, transform=trans, radius=-2.0)):
            text.remove()


# -----------------------------------------------------------------------------
# Quiver key aligned with panel heading
# -----------------------------------------------------------------------------
VECTOR_KEY_X = 0.84
PANEL_HEADING_Y = 1.025


class HeadingAlignedQuiverKey(QuiverKey):
    def __init__(self, *args, **kwargs):
        self.heading_artist = None
        super().__init__(*args, **kwargs)

    def draw(self, renderer):
        if self.heading_artist is not None:
            box = self.heading_artist.get_window_extent(renderer)
            center_y = (box.y0 + box.y1) / 2.0
            anchor_x = self.axes.transAxes.transform((self.X, 0.0))[0]
            self.Y = self.axes.transAxes.inverted().transform((anchor_x, center_y))[1]
        super().draw(renderer)


def heading_quiverkey(ax, quiver, **kwargs):
    kwargs.update(X=VECTOR_KEY_X, Y=PANEL_HEADING_Y, coordinates="axes", angle=0, labelpos="E")
    key = HeadingAlignedQuiverKey(quiver, **kwargs)
    ax.add_artist(key)
    return key


# -----------------------------------------------------------------------------
# Five-panel layout
# -----------------------------------------------------------------------------
PANEL_FILENAMES = ("Fig4a.py", "Fig4b.py", "Fig4c.py", "Fig4d.py", "Fig4e.py")


def load_panels(namespace="fig4", directory=None):
    here = Path(directory or __file__).resolve().parent
    missing = [name for name in PANEL_FILENAMES if not (here / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing Figure 4 panel scripts:\n" + "\n".join(missing))
    modules = []
    for letter, filename in zip("abcde", PANEL_FILENAMES):
        spec = spec_from_file_location(f"{namespace}_{letter}", here / filename)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        module.SIG_COLOR = "#4B1D6B"
        module.FIG5_BOX_COLOR = "#D81B60"
        module.FIG5_BOX_LW = 3.0
        modules.append(module)
    a, b, c, d, e = modules
    a.WAF_MIN_MAGNITUDE = c.WAF_MIN_MAGNITUDE = 0.15
    a.REGION_BOX = (120.0, 240.0, 60.0, 85.0)
    d.LON1, d.LON2, d.LAT1, d.LAT2 = 330.0, 30.0, 40.0, 75.0
    d.REGION_BOXES = [(d.LON1, d.LON2, d.LAT1, d.LAT2)]
    return tuple(modules)


def add_horizontal_colorbar(cax, cmap, norm, levels, label, ticks, small_size=8.0, medium_size=9.0):
    ticks = np.asarray(ticks, float)
    ticks = ticks[~np.isclose(ticks, 0)]
    cb = cax.figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
        boundaries=levels, ticks=ticks, spacing="uniform", extend="both",
    )
    cb.ax.set_xticklabels([f"−{abs(float(v)):g}" if float(v) < 0 else f"{float(v):g}" for v in ticks])
    cb.ax.tick_params(labelsize=small_size, length=2.2, width=.6, pad=1.5)
    cb.outline.set_linewidth(.5)
    cb.set_label(label, fontsize=medium_size, fontweight="normal", labelpad=2)
    return cb


def build_figure(modules, data_a, data_b, data_c, data_d, data_e,
                 waf_style=None, colorbar_ticks=None, show_slp=True,
                 figure_size=(183 / 25.4, 205 / 25.4), dpi=150):
    a, b, c, d, e = modules
    if waf_style is None:
        waf_style = a.waf_style(data_a, data_c)
    if colorbar_ticks is None:
        colorbar_ticks = [
            [-30, -20, -10, 10, 20, 30],
            [-3, -2, -1, 1, 2, 3],
            [-30, -20, -10, 10, 20, 30],
            [-1.2, -.6, .6, 1.2],
            [-.8, -.4, .4, .8],
        ]
    fig = plt.figure(figsize=figure_size, dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(3, 2, left=.045, right=.975, top=.975, bottom=.075, wspace=.28, hspace=.55)
    ax_a = fig.add_subplot(gs[0, 0], projection=a.PROJ)
    ax_b = fig.add_subplot(gs[0, 1], projection=b.PROJ)
    ax_c = fig.add_subplot(gs[1, 0], projection=c.PROJ)
    ax_d = fig.add_subplot(gs[1, 1], projection=d.PROJ)
    ax_a.set_anchor("E"); ax_c.set_anchor("E"); ax_b.set_anchor("W"); ax_d.set_anchor("W")
    a.draw_panel(ax_a, data_a, style=waf_style, panel="a")
    b.draw_panel(ax_b, data_b, panel="b")
    c.draw_panel(ax_c, data_c, style=waf_style, panel="c")
    d.draw_panel(ax_d, data_d, panel="d")
    fig.canvas.draw()
    left_ref = ax_a.get_position()
    row3 = gs[2, 0].get_position(fig)
    e_height = left_ref.width * fig.get_figwidth() / e.SECTOR_MAP_ASPECT / fig.get_figheight()
    ax_e = fig.add_axes([left_ref.x0, row3.y1 - e_height, left_ref.width, e_height], projection=e.SECTOR_PROJ)
    ax_e.set_anchor("W")
    _, labels = e.draw_panel(ax_e, data_e, panel="e", show_slp=show_slp)
    e.remove_labels_outside(ax_e, labels)
    caxes = []
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        p = ax.get_position()
        gap = .045 if ax is ax_e else .035
        caxes.append(fig.add_axes([p.x0, p.y0 - gap, p.width, .012]))
    specs = [
        (a.Z500_CMAP, a.Z500_NORM, a.Z500_LEVELS, "500 hPa Geopotential Height (gpm)"),
        (b.U_CMAP, b.U_NORM, b.U_LEVELS, f"{b.PRESSURE_HPA:g} hPa Zonal Wind (m s$^{{-1}}$)"),
        (c.Z500_CMAP, c.Z500_NORM, c.Z500_LEVELS, "500 hPa Geopotential Height (gpm)"),
        (d.U_CMAP, d.U_NORM, d.U_LEVELS, f"{d.PRESSURE_HPA:g} hPa Zonal Wind (m s$^{{-1}}$)"),
        (e.TEMP_CMAP, e.TEMP_NORM, e.TEMP_LEVELS, "Surface Air Temperature (°C)"),
    ]
    for cax, spec, ticks in zip(caxes, specs, colorbar_ticks):
        add_horizontal_colorbar(cax, *spec, ticks, small_size=a.SMALL_SIZE, medium_size=a.MEDIUM_SIZE)
    fig.canvas.draw()
    return fig


# Small helpers used by Fig4m.py

def mean_arrays(arrays):
    if len(arrays) == 1:
        return arrays[0].copy(deep=True)
    ens = xr.DataArray(np.arange(len(arrays)), dims="ensemble", name="ensemble")
    return xr.concat(arrays, dim=ens, join="exact").mean("ensemble", skipna=True)


def stack_arrays(arrays):
    ens = xr.DataArray(np.arange(len(arrays)), dims="ensemble", name="ensemble")
    return xr.concat(arrays, dim=ens, join="exact")


def anomaly_from_year_series(da_year, target_years, clim_years):
    common = np.intersect1d(da_year.time, np.union1d(target_years, clim_years))
    target = np.intersect1d(common, target_years)
    climate = np.intersect1d(common, clim_years)
    if target.size == 0 or climate.size == 0:
        raise ValueError("Target or climatology years are missing.")
    return (da_year.sel(time=target).mean("time", skipna=True)
            - da_year.sel(time=climate).mean("time", skipna=True)).astype("float32")



def running_mean_update(current, new, n_seen):
    if current is None:
        return new.copy(deep=True)
    current, new = xr.align(current, new, join="exact")
    current.data[...] += (new.data - current.data) / np.float32(n_seen)
    return current


def false_sig_like(da):
    return xr.full_like(da, False, dtype=bool)
