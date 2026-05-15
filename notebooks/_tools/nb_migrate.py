"""Notebook migration helper for the `notebooks/` folder.

Two subcommands:

    upgrade   bump nbformat to >=4.5, set kernel to `rl_signaling`, assign
              stable cell IDs.
    audit     report every cell that still references the legacy
              `rl_signaling` API (NetMultiAgentEnv / TempNetMultiAgentEnv /
              simulation_function / temp_simulation_function) or the
              Colab-only scaffolding (`!git clone`, `%cd`,
              `google.colab.drive`).

Both commands accept either a single `.ipynb` path or a directory; in
the directory case every `*.ipynb` under it is processed.

Source-level rewrites (legacy -> canonical API) are intentionally NOT
performed here — they should land as small, reviewable diffs via
`NotebookEdit` with `cell_id=...`. See NOTEBOOK_REFACTOR_PLAN.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

try:
    import nbformat
except ImportError:
    sys.exit(
        "nbformat is required. Install via `pip install -e \".[dev]\"` from the repo root."
    )


TARGET_NBFORMAT_MAJOR = 4
TARGET_NBFORMAT_MINOR = 5
TARGET_KERNEL_NAME = "rl_signaling"
TARGET_KERNEL_DISPLAY = "Python (rl_signaling)"

LEGACY_API_PATTERNS = {
    "NetMultiAgentEnv":         r"\bNetMultiAgentEnv\b",
    "TempNetMultiAgentEnv":     r"\bTempNetMultiAgentEnv\b",
    "simulation_function":      r"\bsimulation_function\b",
    "temp_simulation_function": r"\btemp_simulation_function\b",
}

COLAB_PATTERNS = {
    "!git clone":               r"^\s*!\s*git\s+clone\b",
    "%cd":                      r"^\s*%cd\b",
    "google.colab.drive":       r"from\s+google\.colab\s+import\s+drive|google\.colab\.drive",
    "colab.userdata":           r"google\.colab\.userdata",
    "runtime.unassign":         r"google\.colab\.runtime|runtime\.unassign\(\)",
}


def iter_notebooks(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".ipynb":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.ipynb"))
    raise SystemExit(f"Not a notebook or directory: {path}")


def upgrade(path: Path) -> dict:
    """Bump nbformat, normalize the kernel, assign stable cell IDs.

    Returns a small dict describing the changes made.
    """
    nb = nbformat.read(str(path), as_version=4)
    changes: dict[str, object] = {}

    if nb.nbformat != TARGET_NBFORMAT_MAJOR or nb.nbformat_minor < TARGET_NBFORMAT_MINOR:
        changes["nbformat"] = (
            f"{nb.nbformat}.{nb.nbformat_minor} -> "
            f"{TARGET_NBFORMAT_MAJOR}.{TARGET_NBFORMAT_MINOR}"
        )
        nb.nbformat = TARGET_NBFORMAT_MAJOR
        nb.nbformat_minor = TARGET_NBFORMAT_MINOR

    kernelspec = nb.metadata.setdefault("kernelspec", {})
    if kernelspec.get("name") != TARGET_KERNEL_NAME:
        changes["kernel_name"] = f"{kernelspec.get('name')!r} -> {TARGET_KERNEL_NAME!r}"
        kernelspec["name"] = TARGET_KERNEL_NAME
    if kernelspec.get("display_name") != TARGET_KERNEL_DISPLAY:
        changes["kernel_display"] = (
            f"{kernelspec.get('display_name')!r} -> {TARGET_KERNEL_DISPLAY!r}"
        )
        kernelspec["display_name"] = TARGET_KERNEL_DISPLAY
    kernelspec.setdefault("language", "python")

    assigned_ids = 0
    for cell in nb.cells:
        if not cell.get("id"):
            cell["id"] = uuid.uuid4().hex[:8]
            assigned_ids += 1
    if assigned_ids:
        changes["assigned_ids"] = assigned_ids

    if changes:
        nbformat.validate(nb)
        nbformat.write(nb, str(path))
        # Round-trip JSON sanity check per the KB notebook skill.
        with open(path) as f:
            json.load(f)
    return changes


def audit(path: Path) -> dict:
    """Report legacy-API / Colab-only patterns in a notebook.

    Returns a dict ``{pattern_label: [(cell_id, line_number, line_text), ...]}``
    plus metadata about kernel/nbformat compliance.
    """
    nb = nbformat.read(str(path), as_version=4)
    hits: dict[str, list[tuple[str, int, str]]] = {}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cid = cell.get("id", "<no-id>")
        source = "".join(cell.source) if isinstance(cell.source, list) else cell.source
        for line_no, line in enumerate(source.splitlines(), start=1):
            for label, pattern in {**LEGACY_API_PATTERNS, **COLAB_PATTERNS}.items():
                if re.search(pattern, line):
                    hits.setdefault(label, []).append((cid, line_no, line.strip()))

    report = {
        "nbformat": f"{nb.nbformat}.{nb.nbformat_minor}",
        "kernel_name": nb.metadata.get("kernelspec", {}).get("name"),
        "hits": hits,
    }
    return report


def format_audit_report(path: Path, report: dict) -> str:
    out = [f"== {path}"]
    nb_ok = report["nbformat"] == f"{TARGET_NBFORMAT_MAJOR}.{TARGET_NBFORMAT_MINOR}"
    kernel_ok = report["kernel_name"] == TARGET_KERNEL_NAME
    out.append(
        f"  nbformat={report['nbformat']} {'OK' if nb_ok else 'NEEDS UPGRADE'}; "
        f"kernel={report['kernel_name']!r} {'OK' if kernel_ok else 'NEEDS UPGRADE'}"
    )
    if not report["hits"]:
        out.append("  legacy-API hits: none")
    else:
        out.append("  legacy-API hits:")
        for label, instances in report["hits"].items():
            out.append(f"    {label} (x{len(instances)}):")
            for cid, ln, line in instances:
                out.append(f"      cell {cid} L{ln}: {line[:100]}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("upgrade", help="bump nbformat / kernel / cell IDs")
    p_up.add_argument("path", type=Path)

    p_au = sub.add_parser("audit", help="report legacy-API and Colab patterns")
    p_au.add_argument("path", type=Path)

    args = parser.parse_args(argv)
    nbs = iter_notebooks(args.path)
    if not nbs:
        print(f"No notebooks found under {args.path}", file=sys.stderr)
        return 1

    if args.cmd == "upgrade":
        for nb_path in nbs:
            changes = upgrade(nb_path)
            if changes:
                print(f"== {nb_path}")
                for k, v in changes.items():
                    print(f"  {k}: {v}")
            else:
                print(f"== {nb_path}: already current")
    elif args.cmd == "audit":
        for nb_path in nbs:
            report = audit(nb_path)
            print(format_audit_report(nb_path, report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
