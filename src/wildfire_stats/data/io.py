"""Load and prepare fire perimeter datasets (MTBS, Idaho HFD, GFA, LANDFIRE EVT)."""

import os
import pickle
import re

import geopandas as gpd
import pandas as pd

from wildfire_stats.config import resolve

# Rough CONUS bounding box (lon/lat) used to subset GFA in notebook 3
CONUS_BBOX = {"minx": -125.0, "maxx": -66.0, "miny": 24.0, "maxy": 50.0}

# Shapefiles truncate column names to 10 characters; this restores the ones we write
SHAPEFILE_RENAMES = {
    "modis_cl": "modis_class_timevary",
    "modis_cl_1": "modis_class_static",
    "biome_refi": "modis_class_static_grouped",
}

GFA_PERIMETER_PATTERN = re.compile(r"^Global_fire_atlas_V1_perimeter_(\d{4})$")

# MTBS region subsets, chosen by the 2-letter state prefix of FIRE_ID (e.g. "AK...", "CA...").
# "all" = CONUS + Alaska; Hawai'i and Puerto Rico are always excluded.
MTBS_REGIONS = ("conus", "ak", "all")
MTBS_EXCLUDED_STATES = ("HI", "PR")


def _find_gfa_shapefiles(folder):
    """Map year -> path for GFA perimeter shapefiles under ``folder``.

    Supports both layouts:
      * flat:   ``folder/Global_fire_atlas_V1_perimeter_YYYY.shp``
      * nested: ``folder/Global_fire_atlas_V1_perimeter_YYYY/Global_fire_atlas_V1_perimeter_YYYY.shp``
    Flat files take precedence if both exist for a year.
    """
    found = {}
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        full = os.path.join(folder, name)
        if ext.lower() == ".shp":
            m = GFA_PERIMETER_PATTERN.match(stem)
            if m:
                found[int(m.group(1))] = full
        elif os.path.isdir(full):
            m = GFA_PERIMETER_PATTERN.match(name)
            nested = os.path.join(full, name + ".shp")
            if m and os.path.exists(nested):
                found.setdefault(int(m.group(1)), nested)
    return dict(sorted(found.items()))


def load_shapefile(shp_path, projection, area_col=None, equal_area_crs=None, where=None):
    """Load a vector file, set its CRS, reproject to equal-area, and optionally compute area.

    Args:
        shp_path (str): Path to the shapefile (or other vector file) to load.
        projection (str): CRS string or EPSG code for the source projection.
        area_col (str, optional): Name of column to store polygon area in km².
            If provided and missing, it will be computed.
        equal_area_crs (str, optional): Target CRS; defaults to ``equal_area_crs`` in the yaml.
        where (str, optional): OGR SQL ``WHERE`` clause applied while reading, so
            rows that don't match are never loaded.

    Returns:
        GeoDataFrame: Reprojected to the equal-area CRS with optional area column.
    """
    equal_area_crs = resolve(equal_area_crs, "equal_area_crs")
    gdf = gpd.read_file(shp_path) if where is None else gpd.read_file(shp_path, where=where)
    gdf = gdf.set_crs(projection)
    gdf = gdf.to_crs(equal_area_crs)

    # Compute area in km² if requested and not already present
    if area_col is not None and area_col not in gdf.columns:
        gdf[area_col] = gdf.geometry.area / 1e6

    return gdf


def _resolve_mtbs_region(region):
    region = str(resolve(region, "mtbs_region")).lower()
    if region not in MTBS_REGIONS:
        raise ValueError(f"mtbs region must be one of {MTBS_REGIONS}, got {region!r}")
    return region


def mtbs_region_where(region=None, id_col="FIRE_ID"):
    """OGR SQL ``WHERE`` clause selecting MTBS fires for ``region`` by FIRE_ID state prefix.

    ``"ak"`` keeps Alaska only; ``"conus"`` drops AK, HI and PR; ``"all"`` drops HI and PR.
    """
    region = _resolve_mtbs_region(region)
    if region == "ak":
        return f"{id_col} LIKE 'AK%'"
    excluded = MTBS_EXCLUDED_STATES + (("AK",) if region == "conus" else ())
    return " AND ".join(f"{id_col} NOT LIKE '{s}%'" for s in excluded)


def mtbs_tag(region=None):
    """Filename prefix for MTBS outputs, e.g. ``mtbs_conus`` / ``mtbs_ak`` / ``mtbs_all``."""
    return f"mtbs_{_resolve_mtbs_region(region)}"


def minfire_tag(min_fire=None):
    """Filename tag for the minimum fire size, e.g. ``minfire6p45`` / ``minfire4``.

    ``min_fire`` defaults to the current ``min_fire`` (override or yaml). The
    value is formatted with ``:g`` and ``.`` becomes ``p``, so ``4`` and ``4.0``
    give the same tag and shapefile names carry no extra dot.
    """
    value = float(resolve(min_fire, "min_fire"))
    return "minfire" + f"{value:g}".replace(".", "p")


def load_mtbs(path, area_col="AREA_SQKM", crs=None, max_year=None, year_col="YEAR", region=None):
    """Load MTBS burn-area boundaries in the equal-area CRS.

    The file's CRS is overridden with ``mtbs_crs`` before reprojecting. With the
    default ``area_col="AREA_SQKM"`` the vendor's area column is kept; pass
    ``area_col="area_km2"`` to compute polygon areas instead (notebook 1).

    ``region`` (default ``mtbs_region`` in the yaml) is ``"conus"``, ``"ak"`` or
    ``"all"`` (CONUS + Alaska). Hawai'i and Puerto Rico are always excluded. The
    filter runs at read time, so fires outside the region are never loaded. The
    region is recorded in ``gdf.attrs["mtbs_region"]``.

    Fires after ``mtbs_max_year`` (default from the yaml) are dropped because
    MTBS is still mapping/reviewing them; pass ``max_year`` to override.
    """
    region = _resolve_mtbs_region(region)
    gdf = load_shapefile(
        path, resolve(crs, "mtbs_crs"), area_col=area_col, where=mtbs_region_where(region)
    )
    gdf = cap_mtbs_years(gdf, year_col=year_col, max_year=max_year)
    gdf.attrs["mtbs_region"] = region
    return gdf


def load_idaho(path, layer="Fires1950_Present", equal_area_crs=None):
    """Load the Idaho Historical Fires Database layer, reproject, and add ``area_km2``."""
    gdf = gpd.read_file(path, layer=layer)
    gdf = gdf.to_crs(resolve(equal_area_crs, "equal_area_crs"))
    gdf["area_km2"] = gdf.geometry.area / 1e6
    return gdf


def load_gfa(
    folder,
    years=None,
    bbox=None,
    to_equal_area=True,
    verbose=True,
    crs=None,
    equal_area_crs=None,
):
    """Load and concatenate the yearly Global Fire Atlas perimeter shapefiles.

    Args:
        folder: Directory containing ``Global_fire_atlas_V1_perimeter_YYYY.shp`` files
            (or, in the older layout, one ``Global_fire_atlas_V1_perimeter_YYYY/`` folder per year).
        years: Optional inclusive ``(start, end)`` year range.
        bbox: Optional lon/lat box ``{"minx", "maxx", "miny", "maxy"}`` (e.g. ``CONUS_BBOX``).
        to_equal_area: Reproject to the equal-area CRS and add ``area_km2`` / ``perim_km``.
            If False, geometries stay in the GFA CRS and no areas are computed.
        verbose: Print each file as it is read.

    Returns:
        GeoDataFrame with a ``YEAR`` column.
    """
    crs = resolve(crs, "gfa_crs")
    equal_area_crs = resolve(equal_area_crs, "equal_area_crs")

    gfa_all_data = []
    for gfa_year, gfa_shp_path in _find_gfa_shapefiles(folder).items():
        if years is not None and not (years[0] <= gfa_year <= years[1]):
            continue

        if verbose:
            print(f"Processing {gfa_shp_path} ...")
        gfa_gdf = gpd.read_file(gfa_shp_path).set_crs(crs)
        # bbox is lon/lat, so clip in the native GFA CRS before any reprojection
        if bbox is not None:
            gfa_gdf = gfa_gdf.cx[
                bbox["minx"] : bbox["maxx"], bbox["miny"] : bbox["maxy"]
            ]
        if to_equal_area:
            gfa_gdf = gfa_gdf.to_crs(equal_area_crs)
            gfa_gdf["area_km2"] = gfa_gdf.geometry.area / 1e6
            gfa_gdf["perim_km"] = gfa_gdf.geometry.length / 1000
        gfa_gdf["YEAR"] = gfa_year
        gfa_all_data.append(gfa_gdf)

    out_crs = equal_area_crs if to_equal_area else crs
    if not gfa_all_data:
        return gpd.GeoDataFrame(
            columns=["fire_ID", "geometry", "YEAR", "landcover_s"], crs=out_crs
        )
    return gpd.GeoDataFrame(pd.concat(gfa_all_data, ignore_index=True), crs=out_crs)


def filter_years(df, year_col, years):
    """Rows with ``years[0] <= df[year_col] <= years[1]``."""
    return df[(df[year_col] >= years[0]) & (df[year_col] <= years[1])]


def cap_mtbs_years(df, year_col="YEAR", max_year=None):
    """Drop MTBS fires after ``mtbs_max_year`` (provisional years still under MTBS review).

    See https://www.mtbs.gov/data-availability.
    """
    return df[df[year_col] <= resolve(max_year, "mtbs_max_year")]


def overlap_with(gdf, other, dedup_col):
    """Fires in ``gdf`` that intersect any geometry in ``other``, one row per ``dedup_col``."""
    out = gpd.sjoin(gdf, other[["geometry"]], predicate="intersects")
    return out.drop_duplicates(subset=dedup_col)


def restore_truncated_columns(gdf, mapping=None):
    """Undo shapefile 10-character column truncation (missing keys are ignored)."""
    return gdf.rename(columns=SHAPEFILE_RENAMES if mapping is None else mapping)


def load_evt_attributes(path):
    """Read the LANDFIRE EVT attribute table CSV."""
    return pd.read_csv(path)


def attach_evt_attributes(gdf, attrs, code_col="EVT_class", key_col="VALUE"):
    """Left-merge EVT attributes (EVT_NAME, EVT_LF, EVT_PHYS, EVT_GP_N, ...) onto fires."""
    return gdf.merge(attrs, how="left", left_on=code_col, right_on=key_col)


def save_pickle(obj, path, verbose=True):
    """Pickle ``obj`` to ``path`` (creating parent directories)."""
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    if verbose:
        print(f"Saved to: {path}")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_vector(gdf, path, verbose=True):
    """Write a GeoDataFrame with ``to_file`` (creating parent directories)."""
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    gdf.to_file(path)
    if verbose:
        print(f"Saved GeoDataFrame to {path}")
