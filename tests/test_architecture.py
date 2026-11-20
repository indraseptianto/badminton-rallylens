"""Layer-boundary guards (see CLAUDE.md → Architecture).

These parse imports with ``ast`` so they run without importing heavy deps.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "rallylens"

# cv2 outside vision/ and viz/ is existing debt; the allowlist stops it growing.
CV2_ALLOWED_OUTSIDE_VISION = {
    "common.py",
    "pipeline/court.py",
    "pipeline/shuttle.py",
    "analysis/metrics.py",
}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def test_analysis_never_imports_viz() -> None:
    offenders = [
        _rel(p)
        for p in (SRC / "analysis").rglob("*.py")
        if any(m == "rallylens.viz" or m.startswith("rallylens.viz.") for m in _imports(p))
    ]
    assert not offenders, f"analysis must not depend on viz: {offenders}"


def test_vision_never_imports_viz_or_analysis() -> None:
    bad = ("rallylens.viz", "rallylens.analysis", "rallylens.pipeline")
    offenders = [
        _rel(p)
        for p in (SRC / "vision").rglob("*.py")
        if any(m == b or m.startswith(b + ".") for m in _imports(p) for b in bad)
    ]
    assert not offenders, f"vision must be a leaf layer: {offenders}"


def test_cv2_stays_in_allowed_modules() -> None:
    offenders = [
        _rel(p)
        for p in SRC.rglob("*.py")
        if not _rel(p).startswith(("vision/", "viz/"))
        and _rel(p) not in CV2_ALLOWED_OUTSIDE_VISION
        and "cv2" in _imports(p)
    ]
    assert not offenders, f"new cv2 imports outside vision/viz: {offenders}"
