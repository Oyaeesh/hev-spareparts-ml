import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

BASE_BLOCKLIST = {
    "on line price",
    "online price",
    "maintenance_to_price",
}

def sanitize_feature_lists(
    df,
    cat_cols_raw: Sequence[str],
    num_cols_raw: Sequence[str],
    blocklist: Optional[Iterable[str]] = None,
    drop_part: bool = True,
) -> Tuple[list, list, set]:
    """Return sanitised categorical and numerical feature lists."""
    blocklist = set(BASE_BLOCKLIST if blocklist is None else blocklist)

    cat_cols = [c for c in cat_cols_raw if c not in blocklist]
    if drop_part and 'part' in cat_cols:
        cat_cols.remove('part')

    num_cols = [c for c in num_cols_raw if c not in blocklist]

    required_cols = set(cat_cols + num_cols + ['price'])
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return cat_cols, num_cols, blocklist

def save_feature_metadata(
    path: Path,
    categorical: Sequence[str],
    numerical: Sequence[str],
    blocklist: Iterable[str],
    thresholds: Optional[dict] = None,
    enabled: bool = True,
) -> None:
    """Persist feature metadata to JSON when enabled."""
    if not enabled:
        return

    path = Path(path)
    data = {
        "feature_schema": {
            "categorical": list(categorical),
            "numerical": list(numerical),
        },
        "blocklist": sorted(set(blocklist)),
    }
    if thresholds is not None:
        data["thresholds"] = {
            "low": float(thresholds.get("low")),
            "high": float(thresholds.get("high")),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2)

def load_feature_metadata(path: Path) -> Optional[dict]:
    """Load persisted feature metadata if it exists."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)
