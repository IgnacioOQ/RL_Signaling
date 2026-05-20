#!/usr/bin/env python3
"""
dash_audit.py — report em-dash occurrences in .tex / .md files.

Distinguishes LaTeX em-dash ``---`` (three hyphens) from Unicode em-dash ``—``
and from the LaTeX en-dash ``--`` used for ranges (Lewis--Skyrms, pages 12--14);
en-dashes are NOT reported.

CLI:
    python scripts/dash_audit.py manuscript/main_v2.tex
    python scripts/dash_audit.py manuscript/main_v2.tex --count-only
    python scripts/dash_audit.py manuscript/main_v2.tex --max 30
    python scripts/dash_audit.py manuscript/main_v2.tex --exclude-pattern '\\citep\\{[^}]*--[^}]*\\}'

Exit 0 by default (audit is informational). Pass --max N to fail with exit 1
when the count exceeds N.  Use --soft to force exit 0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Exactly three hyphens — not two, not four. (Lookarounds prevent eating
# longer runs like "----" as a "---" match.)
RE_LATEX_EM = re.compile(r"(?<!-)-{3}(?!-)")
RE_UNICODE_EM = re.compile(r"—")


@dataclass
class DashHit:
    path: str
    line: int
    col: int
    kind: str  # "latex" | "unicode"
    context: str


@dataclass
class FileReport:
    path: str
    latex_count: int = 0
    unicode_count: int = 0
    hits: list[DashHit] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.latex_count + self.unicode_count


@dataclass
class DashReport:
    files: list[FileReport] = field(default_factory=list)
    excluded_count: int = 0

    @property
    def total_latex(self) -> int:
        return sum(f.latex_count for f in self.files)

    @property
    def total_unicode(self) -> int:
        return sum(f.unicode_count for f in self.files)

    @property
    def total(self) -> int:
        return self.total_latex + self.total_unicode


def _strip_tex_comment(line: str) -> str:
    """Return the non-comment portion of a .tex line (everything before an
    unescaped ``%``). Lines whose dash sits in a comment are still scanned —
    we just clip the line for context-display purposes."""
    out: list[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(c)
            out.append(line[i + 1])
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def scan_file(
    path: Path,
    context_chars: int,
    exclude_patterns: list[re.Pattern[str]],
    skip_tex_comments: bool,
) -> FileReport:
    report = FileReport(path=str(path))
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        scan_line = (
            _strip_tex_comment(raw_line)
            if (skip_tex_comments and path.suffix == ".tex")
            else raw_line
        )
        for kind, regex in (("latex", RE_LATEX_EM), ("unicode", RE_UNICODE_EM)):
            for m in regex.finditer(scan_line):
                col = m.start()
                left = max(0, col - context_chars)
                right = min(len(scan_line), m.end() + context_chars)
                context = scan_line[left:right].replace("\t", " ")
                if any(p.search(context) for p in exclude_patterns):
                    continue
                if kind == "latex":
                    report.latex_count += 1
                else:
                    report.unicode_count += 1
                report.hits.append(
                    DashHit(
                        path=str(path),
                        line=lineno,
                        col=col + 1,
                        kind=kind,
                        context=context,
                    )
                )
    return report


def render_text(report: DashReport, count_only: bool, max_count: int | None) -> str:
    lines: list[str] = []
    lines.append("dash_audit.py — em-dash occurrences")
    lines.append("=" * 60)
    for fr in report.files:
        header = (
            f"{fr.path}: latex {fr.latex_count} + unicode {fr.unicode_count} = "
            f"{fr.total} em-dash(es)"
        )
        lines.append(header)
        if not count_only:
            for h in fr.hits:
                lines.append(
                    f"  {fr.path}:{h.line}:{h.col} [{h.kind}]  …{h.context}…"
                )
    lines.append("")
    lines.append(
        f"Total: latex {report.total_latex} + unicode {report.total_unicode} = "
        f"{report.total} em-dash(es)"
    )
    if max_count is not None:
        verdict = "OVER --max" if report.total > max_count else "OK"
        lines.append(f"  --max: {max_count}  →  {verdict}")
    return "\n".join(lines)


def render_json(report: DashReport, max_count: int | None) -> str:
    payload = {
        "files": [
            {
                "path": fr.path,
                "latex_count": fr.latex_count,
                "unicode_count": fr.unicode_count,
                "total": fr.total,
                "hits": [asdict(h) for h in fr.hits],
            }
            for fr in report.files
        ],
        "total_latex": report.total_latex,
        "total_unicode": report.total_unicode,
        "total": report.total,
        "max": max_count,
        "over_max": (max_count is not None and report.total > max_count),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def expand_inputs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.exists() and p.is_file():
            out.append(p)
            continue
        # Glob relative to cwd.
        matched = sorted(Path().glob(pat))
        if matched:
            out.extend(matched)
        else:
            print(f"warning: no files matched {pat!r}", file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan .tex / .md files for em-dash occurrences (LaTeX ``---`` and "
            "Unicode ``—``); en-dashes ``--`` are intentionally ignored. Used by "
            "todo.dash_sweep and during voice-revision passes."
        )
    )
    parser.add_argument(
        "files", nargs="+",
        help="One or more file paths or globs (e.g. 'manuscript/*.tex').",
    )
    parser.add_argument(
        "--exclude-pattern", "-x", action="append", default=[],
        help=(
            "Regex; an occurrence whose surrounding context matches will be "
            "skipped. Repeatable. Default: none. Example: "
            r"'\\citep\{[^}]*--[^}]*\}' to skip dashes inside bibkeys."
        ),
    )
    parser.add_argument(
        "--context", type=int, default=50,
        help="Characters of surrounding context per occurrence (default 50).",
    )
    parser.add_argument(
        "--count-only", action="store_true",
        help="Emit per-file counts only; no individual occurrences.",
    )
    parser.add_argument(
        "--max", type=int, default=None, metavar="N",
        help="Fail (exit 1) when total dash count exceeds N. Default: no maximum.",
    )
    parser.add_argument(
        "--no-skip-comments", action="store_true",
        help="Do not strip %% comments from .tex files when scanning (default: strip).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--soft", action="store_true",
        help="Always exit 0, even when --max is exceeded.",
    )
    args = parser.parse_args()

    files = expand_inputs(args.files)
    if not files:
        print("error: no input files found", file=sys.stderr)
        return 2

    exclude_patterns = [re.compile(p) for p in args.exclude_pattern]
    report = DashReport()
    for path in files:
        report.files.append(
            scan_file(
                path,
                context_chars=args.context,
                exclude_patterns=exclude_patterns,
                skip_tex_comments=not args.no_skip_comments,
            )
        )

    if args.json:
        print(render_json(report, args.max))
    else:
        print(render_text(report, count_only=args.count_only, max_count=args.max))

    if args.soft:
        return 0
    if args.max is not None and report.total > args.max:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
