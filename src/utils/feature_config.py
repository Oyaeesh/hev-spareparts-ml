"""Utilities for feature configuration shared by notebooks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

# Baseline blocklist entries known to leak target information or duplicate labels.
BASE_BLOCKLIST = {
    "price tier",
    "price_tier",
    "price category",
    "price_category",
    "price bin",
    "price_bin",
    "repair_to_price",
    "predicted_price",
    "target",
}


def _normalize(label: str) -> str:
    """Normalize column names for comparisons."""
    simplified = label.strip().lower()
    while "\\\\" in simplified:
        simplified = simplified.replace("\\\\", "\\")
    return " ".join(simplified.split())


def sanitize_feature_lists(
    df: pd.DataFrame,
    *,
    cat_cols_raw: Sequence[str],
    num_cols_raw: Sequence[str],
    blocklist: Optional[Iterable[str]] = None,
    drop_part: bool = False,
) -> Tuple[List[str], List[str], set]:
    """Return sanitized categorical/numeric feature lists and the resolved blocklist."""

    blocklist_resolved = set(blocklist or set())
    blocklist_resolved.update(BASE_BLOCKLIST)

    df_lookup: Dict[str, str] = {_normalize(col): col for col in df.columns}
    blocklist_norm = set()

    def _add_blocklisted(name: str) -> None:
        if name not in blocklist_resolved:
            blocklist_resolved.add(name)
        blocklist_norm.add(_normalize(name))

    # Resolve blocklist names to actual dataframe columns when possible.
    for entry in list(blocklist_resolved):
        resolved = df_lookup.get(_normalize(entry))
        if resolved is not None:
            blocklist_resolved.discard(entry)
            blocklist_resolved.add(resolved)
            blocklist_norm.add(_normalize(resolved))
        else:
            blocklist_norm.add(_normalize(entry))

    def _prepare(columns: Sequence[str]) -> List[str]:
        prepared: List[str] = []
        for col in columns:
            resolved = df_lookup.get(_normalize(col), col if col in df.columns else None)
            if resolved is None:
                continue
            if drop_part and _normalize(resolved) == "part":
                continue
            if _normalize(resolved) in blocklist_norm:
                _add_blocklisted(resolved)
                continue
            if resolved not in prepared:
                prepared.append(resolved)
        return prepared

    cat_cols = _prepare(cat_cols_raw)
    num_cols = _prepare(num_cols_raw)

    # Auto-include additional numeric columns created upstream (e.g., ratios).
    cat_norm = {_normalize(c) for c in cat_cols}

    for candidate in df.select_dtypes(include=["number", "bool"]).columns:
        candidate_norm = _normalize(candidate)
        if drop_part and candidate_norm == "part":
            continue
        if candidate_norm in cat_norm:
            continue
        if candidate_norm == "price":
            continue
        if candidate_norm in blocklist_norm:
            _add_blocklisted(candidate)
            continue
        if candidate not in num_cols:
            num_cols.append(candidate)

    return cat_cols, num_cols, blocklist_resolved


def save_feature_metadata(
    path: Path,
    *,
    categorical: Sequence[str],
    numerical: Sequence[str],
    blocklist: Iterable[str],
    thresholds: Optional[MutableMapping[str, float]] = None,
    enabled: bool = True,
) -> Optional[Dict[str, object]]:
    """Persist feature schema metadata for reuse across notebook sessions."""
    if not enabled:
        return None

    payload = {
        "feature_schema": {
            "categorical": list(categorical),
            "numerical": list(numerical),
        },
        "blocklist": sorted(set(blocklist)),
    }
    if thresholds is not None:
        payload["thresholds"] = dict(thresholds)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload


def load_feature_metadata(path: Path) -> Dict[str, object]:
    """Load persisted metadata, raising FileNotFoundError if absent."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    metadata.setdefault("feature_schema", {})
    schema = metadata["feature_schema"]
    schema.setdefault("categorical", [])
    schema.setdefault("numerical", [])
    metadata.setdefault("blocklist", [])
    return metadata
