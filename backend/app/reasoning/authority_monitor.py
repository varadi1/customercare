"""
Authority weight drift monitoring.

Tracks how learned authority adjustments change over time:
  - save_authority_snapshot() — saves weekly JSON snapshot
  - compute_authority_drift_report() — compares current vs previous week
  - format_drift_report() — Markdown for Discord/Obsidian
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _snapshot_dir() -> Path:
    from ..config import settings
    d = Path(settings.authority_snapshot_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_authority_snapshot(adjustments: dict[str, dict[str, float]]) -> Path | None:
    """Save a timestamped authority adjustment snapshot."""
    if not adjustments:
        return None

    ts = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
    path = _snapshot_dir() / f"authority_{ts}.json"
    path.write_text(json.dumps(adjustments, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved authority snapshot: %s (%d categories)", path.name, len(adjustments))
    return path


def _load_latest_snapshots(n: int = 2) -> list[tuple[str, dict]]:
    """Load the N most recent snapshots."""
    d = _snapshot_dir()
    files = sorted(d.glob("authority_*.json"), reverse=True)[:n]
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append((f.stem, data))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def compute_authority_drift_report() -> dict[str, Any]:
    """Compare the two most recent authority snapshots.

    Returns:
        {
            "has_drift": bool,
            "current_snapshot": str,
            "previous_snapshot": str,
            "drifts": [
                {"category": str, "chunk_type": str,
                 "current": float, "previous": float, "delta": float}
            ],
            "new_categories": list[str],
            "removed_categories": list[str],
        }
    """
    snapshots = _load_latest_snapshots(2)

    if len(snapshots) < 2:
        return {
            "has_drift": False,
            "message": "Nem eleg snapshot a drift szamitashoz (min. 2 kell)",
            "snapshot_count": len(snapshots),
        }

    curr_name, curr = snapshots[0]
    prev_name, prev = snapshots[1]

    drifts = []
    all_cats = set(curr.keys()) | set(prev.keys())

    for cat in sorted(all_cats):
        curr_types = curr.get(cat, {})
        prev_types = prev.get(cat, {})
        all_types = set(curr_types.keys()) | set(prev_types.keys())

        for ct in sorted(all_types):
            c_val = curr_types.get(ct, 0)
            p_val = prev_types.get(ct, 0)
            delta = round(c_val - p_val, 4)
            if abs(delta) > 0.005:
                drifts.append({
                    "category": cat,
                    "chunk_type": ct,
                    "current": c_val,
                    "previous": p_val,
                    "delta": delta,
                })

    new_cats = sorted(set(curr.keys()) - set(prev.keys()))
    removed_cats = sorted(set(prev.keys()) - set(curr.keys()))

    return {
        "has_drift": len(drifts) > 0 or len(new_cats) > 0 or len(removed_cats) > 0,
        "current_snapshot": curr_name,
        "previous_snapshot": prev_name,
        "drifts": sorted(drifts, key=lambda x: abs(x["delta"]), reverse=True),
        "new_categories": new_cats,
        "removed_categories": removed_cats,
    }


def format_drift_report(report: dict[str, Any]) -> str:
    """Format drift report for Discord or Obsidian."""
    if not report.get("has_drift"):
        return "Authority drift: nincs valtozas."

    lines = [
        f"**Authority Drift** ({report['previous_snapshot']} -> {report['current_snapshot']})",
    ]

    if report.get("new_categories"):
        lines.append(f"  uj: {', '.join(report['new_categories'])}")
    if report.get("removed_categories"):
        lines.append(f"  torolve: {', '.join(report['removed_categories'])}")

    for d in report.get("drifts", [])[:10]:
        sign = "+" if d["delta"] > 0 else ""
        arrow = "↑" if d["delta"] > 0 else "↓"
        lines.append(
            f"  {arrow} {d['category']}/{d['chunk_type']}: "
            f"{d['previous']:+.3f} -> {d['current']:+.3f} ({sign}{d['delta']:.3f})"
        )

    return "\n".join(lines)
