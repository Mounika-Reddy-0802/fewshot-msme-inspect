"""Config loading.

Every experiment is described by a YAML file under ``configs/``. A config may
inherit from another one with ``extends:`` so the shared block lives in
``base.yaml`` and each experiment file only states what it changes.

Two conveniences on top of plain YAML:

* placeholders  ``${data_root}``, ``${results_root}``, ``${repo_root}`` and
  ``${env:VAR}`` are substituted at load time, so no config carries a
  machine-specific path.
* overrides     dotted ``key.path=value`` strings, for command line use.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from src.utils.paths import data_root, repo_root, results_root

CONFIG_DIR = repo_root() / "configs"

_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _substitute(value: Any) -> Any:
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda m: _resolve(m.group(1)), value)
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v) for v in value]
    return value


def _resolve(name: str) -> str:
    if name.startswith("env:"):
        var = name[len("env:") :]
        if var not in os.environ:
            raise KeyError(f"config references ${{env:{var}}} but it is not set")
        return os.environ[var]
    known = {
        "repo_root": repo_root,
        "data_root": data_root,
        "results_root": results_root,
    }
    if name not in known:
        raise KeyError(f"unknown config placeholder ${{{name}}}")
    return known[name]().as_posix()


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _apply_override(config: dict, item: str) -> None:
    if "=" not in item:
        raise ValueError(f"override must look like key.path=value, got {item!r}")
    dotted, raw = item.split("=", 1)
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override into non-mapping key {dotted!r}")
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
    resolve: bool = True,
) -> dict:
    """Load a YAML config, following ``extends`` and applying ``overrides``."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")

    config = _read(path)
    parent = config.pop("extends", None)
    if parent is not None:
        base = load_config(Path(parent) if Path(parent).is_absolute() else path.parent / parent, resolve=False)
        config = _deep_merge(base, config)

    for item in overrides or []:
        _apply_override(config, item)

    return _substitute(config) if resolve else config
