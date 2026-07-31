from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from .compatibility import CompatibilityState


def load_compatibility_matrix() -> dict[str, Any]:
    resource = files("pysbagen").joinpath("data/sbagen_compatibility_matrix.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    validate_compatibility_matrix(payload)
    return payload


def validate_compatibility_matrix(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "pysbagen.sbagen-compatibility-matrix.v1":
        raise ValueError("Unknown compatibility matrix schema")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Compatibility matrix has no rows")
    valid_states = {state.value for state in CompatibilityState}
    seen: set[str] = set()
    required = {"id", "construct", "parser", "execution", "render", "round_trip", "state", "deviation", "fixtures", "provenance"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Matrix row is missing fields: {sorted(missing)}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate matrix row id: {row['id']}")
        seen.add(row["id"])
        if row["state"] not in valid_states:
            raise ValueError(f"Invalid matrix state for {row['id']}: {row['state']}")
        if not row["fixtures"]:
            raise ValueError(f"Matrix row has no fixture ids: {row['id']}")


def matrix_rows() -> list[dict[str, Any]]:
    return list(load_compatibility_matrix()["rows"])


def matrix_row(row_id: str) -> dict[str, Any]:
    for row in matrix_rows():
        if row["id"] == row_id:
            return row
    raise KeyError(row_id)
