#!/usr/bin/env python3
"""
bib_unused.py — report bib-entry / cite-key drift between .bib and .tex.

Two checks:
  A. Bib entries defined in the .bib but never \\cite'd anywhere in the .tex.
  B. (with --reverse) Cite keys used in .tex that have no entry in the .bib —
     typos or missing-citation cases.

CLI:
    python scripts/bib_unused.py manuscript/References.bib manuscript/main_v2.tex
    python scripts/bib_unused.py manuscript/References.bib manuscript/main_v2.tex --reverse
    python scripts/bib_unused.py manuscript/References.bib manuscript/main_v2.tex --reverse --strict

Exit 0 if no unused entries (or no missing keys when --reverse). Exit 1 when
drift exists in any checked direction. Use --soft to force exit 0.
The classic intended use is a pre-submission sanity check: trim never-cited
entries from the bibliography to keep the reference list tight.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


# @type{key,            — captures the key on each entry's opening line.
RE_BIB_ENTRY = re.compile(
    r"@(?P<kind>[A-Za-z]+)\s*\{\s*(?P<key>[^\s,{}]+)\s*,",
)

# \cite{key}, \citep{a, b, c}, \citet{key}, \citep*{key}, etc.
# Tolerates whitespace and optional starred form.  Multiple keys split on ','.
RE_CITE = re.compile(
    r"\\cite[pt]?\*?\s*(?:\[[^\]]*\])?\s*(?:\[[^\]]*\])?\s*\{(?P<keys>[^{}]+)\}",
)


@dataclass
class BibAuditReport:
    bib_path: str
    tex_paths: list[str] = field(default_factory=list)
    defined_keys: list[str] = field(default_factory=list)
    used_keys: list[str] = field(default_factory=list)
    unused_entries: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    duplicate_definitions: list[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return bool(self.unused_entries or self.missing_keys)


def parse_bib_keys(bib_text: str) -> tuple[list[str], list[str]]:
    """Return (ordered_keys, duplicates). Order = first-occurrence order."""
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for m in RE_BIB_ENTRY.finditer(bib_text):
        if m["kind"].lower() in {"comment", "preamble", "string"}:
            continue
        key = m["key"]
        if key in seen:
            duplicates.append(key)
        else:
            seen[key] = m.start()
    return list(seen.keys()), duplicates


def parse_cite_keys(tex_text: str) -> list[str]:
    keys: list[str] = []
    for m in RE_CITE.finditer(tex_text):
        for raw in m["keys"].split(","):
            k = raw.strip()
            if k:
                keys.append(k)
    return keys


def audit(bib_path: Path, tex_paths: list[Path]) -> BibAuditReport:
    bib_text = bib_path.read_text(encoding="utf-8", errors="replace")
    defined, duplicates = parse_bib_keys(bib_text)
    defined_set = set(defined)

    used_keys: list[str] = []
    seen_used: set[str] = set()
    for p in tex_paths:
        for k in parse_cite_keys(p.read_text(encoding="utf-8", errors="replace")):
            if k not in seen_used:
                seen_used.add(k)
                used_keys.append(k)
    used_set = set(used_keys)

    unused = [k for k in defined if k not in used_set]
    missing = [k for k in used_keys if k not in defined_set]

    return BibAuditReport(
        bib_path=str(bib_path),
        tex_paths=[str(p) for p in tex_paths],
        defined_keys=defined,
        used_keys=used_keys,
        unused_entries=unused,
        missing_keys=missing,
        duplicate_definitions=duplicates,
    )


def render_text(report: BibAuditReport, show_reverse: bool) -> str:
    lines: list[str] = []
    lines.append("bib_unused.py — bib / cite-key drift report")
    lines.append("=" * 60)
    lines.append(f"  .bib                : {report.bib_path}")
    lines.append(f"  .tex sources        : {', '.join(report.tex_paths)}")
    lines.append(
        f"  defined entries     : {len(report.defined_keys)}  "
        f"(unique cite keys used: {len(report.used_keys)})"
    )
    lines.append("")
    lines.append(f"[A. Defined but never cited] ({len(report.unused_entries)})")
    if not report.unused_entries:
        lines.append("  none")
    else:
        for k in report.unused_entries:
            lines.append(f"  - {k}")

    if show_reverse:
        lines.append("")
        lines.append(f"[B. Cited but not defined in .bib] ({len(report.missing_keys)})")
        if not report.missing_keys:
            lines.append("  none")
        else:
            for k in report.missing_keys:
                lines.append(f"  - {k}")

    if report.duplicate_definitions:
        lines.append("")
        lines.append(
            f"[note: duplicate bib definitions] ({len(report.duplicate_definitions)})"
        )
        for k in report.duplicate_definitions:
            lines.append(f"  - {k}")

    return "\n".join(lines)


def render_json(report: BibAuditReport) -> str:
    payload = asdict(report)
    payload["has_drift"] = report.has_drift
    return json.dumps(payload, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report bib entries that are defined but never cited (and "
            "optionally cite keys with no bib entry). Useful as a pre-submission "
            "bibliography hygiene check."
        )
    )
    parser.add_argument("bib_file", type=Path, help="Path to .bib file.")
    parser.add_argument(
        "tex_files", type=Path, nargs="+",
        help="One or more .tex source files to scan for \\cite calls.",
    )
    parser.add_argument(
        "--reverse", action="store_true",
        help=(
            "Also list cite keys used in .tex that have no entry in .bib "
            "(typos or missing-citation cases)."
        ),
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=(
            "Exit non-zero if either direction shows drift (default: exit "
            "non-zero only on unused-bib-entries drift)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--soft", action="store_true",
        help="Always exit 0, even when drift is detected.",
    )
    args = parser.parse_args()

    if not args.bib_file.exists():
        print(f"error: bib file not found: {args.bib_file}", file=sys.stderr)
        return 2
    missing_tex = [p for p in args.tex_files if not p.exists()]
    if missing_tex:
        print(
            f"error: tex file(s) not found: {', '.join(str(p) for p in missing_tex)}",
            file=sys.stderr,
        )
        return 2

    report = audit(args.bib_file, args.tex_files)
    print(
        render_json(report)
        if args.json
        else render_text(report, show_reverse=args.reverse or args.strict)
    )

    if args.soft:
        return 0
    if args.strict and (report.unused_entries or report.missing_keys):
        return 1
    if (not args.strict) and report.unused_entries:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
