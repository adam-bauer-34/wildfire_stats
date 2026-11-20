"""Load the yaml files in the repo-level ``config/`` directory.

Nothing is cached, so edits to a yaml file are picked up on the next call.
Set ``WILDFIRE_STATS_CONFIG_DIR`` to read config from somewhere else.

``set_overrides(min_fire=4)`` overrides assumptions for the current process
(e.g. a notebook kernel): ``resolve`` and ``load_assumptions`` return the
override instead of the yaml value until ``clear_overrides()`` is called.
Explicit function arguments still take precedence over overrides.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Per-process overrides of default_assumptions.yaml (see set_overrides).
_OVERRIDES: dict = {}


def config_dir() -> Path:
    """Return the directory holding the config yaml files."""
    return Path(os.environ.get("WILDFIRE_STATS_CONFIG_DIR", REPO_ROOT / "config"))


def load_yaml(name: str) -> dict:
    """Read ``config/<name>`` and return its contents as a dict."""
    with open(config_dir() / name) as f:
        return yaml.safe_load(f) or {}


def load_assumptions() -> dict:
    """default_assumptions.yaml merged with any ``set_overrides`` values."""
    return {**load_yaml("default_assumptions.yaml"), **_OVERRIDES}


def set_overrides(**kw) -> dict:
    """Override assumptions for this process, e.g. ``set_overrides(min_fire=4)``.

    Keys must exist in default_assumptions.yaml (``KeyError`` otherwise).
    Passing ``key=None`` removes that override. Returns the active overrides.
    """
    known = load_yaml("default_assumptions.yaml")
    unknown = [k for k in kw if k not in known]
    if unknown:
        raise KeyError(
            f"unknown assumption(s) {unknown}; not in default_assumptions.yaml"
        )
    for k, v in kw.items():
        if v is None:
            _OVERRIDES.pop(k, None)
        else:
            _OVERRIDES[k] = v
    return dict(_OVERRIDES)


def clear_overrides() -> None:
    """Remove every ``set_overrides`` value."""
    _OVERRIDES.clear()


def get(key: str):
    """Current value of an assumption (override if set, else the yaml)."""
    return resolve(None, key)


def load_dist_bounds() -> dict:
    return load_yaml("dist_bounds_param_clips.yaml")


def load_diff_evo() -> dict:
    return load_yaml("diff_evo.yaml")


def load_poisson() -> dict:
    return load_yaml("poisson.yaml")


def resolve(value, key: str):
    """Return ``value``; if None, the ``set_overrides`` value for ``key``, else the yaml entry."""
    if value is not None:
        return value
    if key in _OVERRIDES:
        return _OVERRIDES[key]
    return load_yaml("default_assumptions.yaml")[key]


@dataclass(frozen=True)
class Paths:
    data_head: Path
    modis: Path
    mtbs: Path
    gfa: Path
    idaho: Path
    evt: Path
    logged_output: Path
    output: Path
    figs: Path


def load_paths() -> Paths:
    """Build data/output directories from ``config/paths.yaml``.

    ``*_SUBDIR`` entries are relative to ``DATA_HEAD``; ``FIGS_DIR`` is
    relative to the repo root (unless absolute).
    """
    p = load_yaml("paths.yaml")
    head = Path(p["DATA_HEAD"])
    return Paths(
        data_head=head,
        modis=head / p["MODIS_SUBDIR"],
        mtbs=head / p["MTBS_SUBDIR"],
        gfa=head / p["GFA_SUBDIR"],
        idaho=head / p["IDAHO_SUBDIR"],
        evt=head / p["EVT_SUBDIR"],
        logged_output=head / p["LOGGED_OUTPUT_SUBDIR"],
        output=head / p["OUTPUT_SUBDIR"],
        figs=REPO_ROOT / p["FIGS_DIR"],
    )
