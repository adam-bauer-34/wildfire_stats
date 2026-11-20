"""Every package module imports, and no code uses sys.path hacks or the old wfpl name."""

import importlib
import pkgutil
import re

import pytest

import wildfire_stats
from conftest import REPO

MODULES = sorted(
    m.name for m in pkgutil.walk_packages(wildfire_stats.__path__, "wildfire_stats.")
    if ".old" not in m.name
)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_expected_modules_present():
    for name in [
        "wildfire_stats.config",
        "wildfire_stats.data.io",
        "wildfire_stats.data.rasters",
        "wildfire_stats.data.classification",
        "wildfire_stats.fitting.fits",
        "wildfire_stats.fitting.powerlaw",
        "wildfire_stats.fitting.stats_helpers.ccdf",
        "wildfire_stats.fitting.stats_helpers.qq",
        "wildfire_stats.fitting.stats_helpers.kuiper",
        "wildfire_stats.fitting.stats_helpers.logpdf",
        "wildfire_stats.plotting.dataset_comp",
        "wildfire_stats.plotting.distributions",
        "wildfire_stats.plotting.landcover",
        "wildfire_stats.plotting.temporal",
        "wildfire_stats.reporting.summaries",
        "wildfire_stats.reporting.tables",
    ]:
        assert name in MODULES


def _sources():
    for p in (REPO / "src").rglob("*.py"):
        yield p, p.read_text()
    for p in (REPO / "dev").glob("*.ipynb"):  # wrappers only, not dev/original_nbs
        yield p, p.read_text()


@pytest.mark.parametrize("pattern", [r"sys\.path\.(append|insert)", r"^\W*import wildfire_powerlaw", r"^\W*from utils\b"])
def test_no_legacy_imports(pattern):
    # ^\W* also catches statements inside notebook JSON strings ('    "import ...')
    hits = [str(p) for p, text in _sources() if re.search(pattern, text, flags=re.M)]
    assert not hits, hits


def test_config_resolve(isolated_config):
    from wildfire_stats import config as cfg

    assert cfg.resolve(None, "min_fire") == 6.45
    assert cfg.resolve(7, "min_fire") == 7
    assert isinstance(cfg.load_assumptions()["prior_weight"], float)
    de = cfg.load_dist_bounds()["de_bounds"]
    assert all(isinstance(v, float) for b in de.values() for pair in b for v in pair)
    p = cfg.load_paths()
    assert p.mtbs.name == "mtbs" and p.logged_output.name == "logged_output"


def test_load_mtbs_caps_max_year(tmp_path, isolated_config):
    import geopandas as gpd

    from wildfire_stats.data.io import load_mtbs
    from shapely.geometry import box

    g = gpd.GeoDataFrame(
        {"YEAR": [2000, 2022, 2023, 2024], "FIRE_ID": [f"CA{i}" for i in range(4)]},
        geometry=[box(-100 + i, 40, -99.5 + i, 40.5) for i in range(4)],
    )
    p = tmp_path / "mtbs.shp"
    g.to_file(p)

    assert load_mtbs(p)["YEAR"].tolist() == [2000, 2022]
    assert load_mtbs(p, max_year=2024)["YEAR"].tolist() == [2000, 2022, 2023, 2024]

    cfg_file = isolated_config / "default_assumptions.yaml"
    cfg_file.write_text(cfg_file.read_text().replace("mtbs_max_year: 2022", "mtbs_max_year: 2023"))
    assert load_mtbs(p)["YEAR"].tolist() == [2000, 2022, 2023]


def test_load_mtbs_region(tmp_path, isolated_config):
    import geopandas as gpd
    import pytest

    from wildfire_stats.data.io import load_mtbs, mtbs_tag
    from shapely.geometry import box

    ids = ["CA001", "AK001", "HI001", "PR001", "OR001", "AK002"]
    g = gpd.GeoDataFrame(
        {"YEAR": [2000] * len(ids), "FIRE_ID": ids},
        geometry=[box(-100 + i, 40, -99.5 + i, 40.5) for i in range(len(ids))],
    )
    p = tmp_path / "mtbs.shp"
    g.to_file(p)

    def got(**kw):
        gdf = load_mtbs(p, **kw)
        return sorted(gdf["FIRE_ID"]), gdf.attrs["mtbs_region"]

    assert got(region="conus") == (["CA001", "OR001"], "conus")
    assert got(region="ak") == (["AK001", "AK002"], "ak")
    assert got(region="all") == (["AK001", "AK002", "CA001", "OR001"], "all")
    assert got(region="AK")[1] == "ak"
    assert got() == (["AK001", "AK002", "CA001", "OR001"], "all")  # yaml default
    assert mtbs_tag() == "mtbs_all"

    cfg_file = isolated_config / "default_assumptions.yaml"
    cfg_file.write_text(cfg_file.read_text().replace("mtbs_region: all", "mtbs_region: ak"))
    assert got() == (["AK001", "AK002"], "ak")
    assert mtbs_tag() == "mtbs_ak"

    with pytest.raises(ValueError):
        load_mtbs(p, region="hawaii")


def test_config_overrides(isolated_config):
    import pytest

    from wildfire_stats import config as cfg

    assert cfg.resolve(None, "min_fire") == 6.45
    cfg.set_overrides(min_fire=4)
    assert cfg.resolve(None, "min_fire") == 4
    assert cfg.get("min_fire") == 4
    assert cfg.resolve(12, "min_fire") == 12  # explicit argument still wins
    assert cfg.load_assumptions()["min_fire"] == 4
    assert cfg.load_assumptions()["min_fires"] == 400  # other keys untouched
    with pytest.raises(KeyError):
        cfg.set_overrides(min_fyre=4)
    cfg.set_overrides(min_fire=None)
    assert cfg.resolve(None, "min_fire") == 6.45
    cfg.set_overrides(min_fire=4)
    cfg.clear_overrides()
    assert cfg.load_assumptions()["min_fire"] == 6.45


def test_minfire_tag(isolated_config):
    from wildfire_stats import config as cfg
    from wildfire_stats.data.io import minfire_tag

    assert minfire_tag() == "minfire6p45"  # yaml default
    assert minfire_tag(6.45) == "minfire6p45"
    assert minfire_tag(4) == minfire_tag(4.0) == "minfire4"
    assert minfire_tag(4.5) == "minfire4p5"
    cfg.set_overrides(min_fire=4)
    assert minfire_tag() == "minfire4"
