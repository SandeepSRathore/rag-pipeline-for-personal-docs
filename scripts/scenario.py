"""Apply / revert the incremental-ingestion scenario (see data/scenarios/README.md).

uv run python scripts/scenario.py apply
uv run python scripts/scenario.py revert
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
SCEN = ROOT / "data" / "scenarios"
BACKUP = ROOT / "var" / "scenario_backup"

MODIFIED = ["hr/pto-policy-2025.md"]
DELETED = ["hr/travel-policy.md"]
ADDED = ["hr/learning-budget.md"]


def apply() -> None:
    if BACKUP.exists():
        sys.exit("scenario already applied (run `revert` first)")
    BACKUP.mkdir(parents=True)
    for rel in MODIFIED + DELETED:
        (BACKUP / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CORPUS / rel, BACKUP / rel)
    for rel in MODIFIED:
        shutil.copy2(SCEN / "modified" / rel, CORPUS / rel)
    for rel in DELETED:
        (CORPUS / rel).unlink()
    for rel in ADDED:
        shutil.copy2(SCEN / "added" / rel, CORPUS / rel)
    print("applied: 1 modified, 1 deleted, 1 added")


def revert() -> None:
    if not BACKUP.exists():
        sys.exit("nothing to revert")
    for rel in MODIFIED + DELETED:
        shutil.copy2(BACKUP / rel, CORPUS / rel)
    for rel in ADDED:
        (CORPUS / rel).unlink(missing_ok=True)
    shutil.rmtree(BACKUP)
    print("reverted corpus to original state")


if __name__ == "__main__":
    {"apply": apply, "revert": revert}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__)
    )()
