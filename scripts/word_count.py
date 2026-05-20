#!/usr/bin/env python3
"""
word_count.py — texcount wrapper for paper revisions.

Reports body / abstract / headers / captions word counts, optionally
broken down per section, with an optional target comparison.

CLI:
    python scripts/word_count.py manuscript/main_v2.tex
    python scripts/word_count.py manuscript/main_v2.tex --target 9000 --per-section
    python scripts/word_count.py manuscript/main_v2.tex --json

Exit 0 if at/under target (or no target supplied); exit 1 if over target.
Use --soft to force exit 0.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Subcount line example:
#   1202+2+0 (1/0/30/0) Subsection: Signaling Games
RE_SUBCOUNT = re.compile(
    r"^\s*(?P<text>\d+)\+(?P<headers>\d+)\+(?P<captions>\d+)\s+"
    r"\((?P<n_headers>\d+)/(?P<n_floats>\d+)/(?P<n_inlines>\d+)/(?P<n_displayed>\d+)\)\s+"
    r"(?P<kind>[A-Za-z_]+):\s*(?P<title>.*?)\s*$"
)


@dataclass
class SectionCount:
    kind: str
    title: str
    text: int
    headers: int
    captions: int

    @property
    def total(self) -> int:
        return self.text + self.headers + self.captions


@dataclass
class WordCountReport:
    tex_path: str
    sum_count: int = 0
    body: int = 0
    headers: int = 0
    captions: int = 0
    n_headers: int = 0
    n_floats: int = 0
    n_math_inline: int = 0
    n_math_displayed: int = 0
    target: int | None = None
    per_section: list[SectionCount] = field(default_factory=list)

    @property
    def over_target(self) -> bool:
        return self.target is not None and self.body > self.target

    @property
    def margin(self) -> int | None:
        if self.target is None:
            return None
        return self.body - self.target


def run_texcount(tex_path: Path, include_subcounts: bool) -> str:
    if shutil.which("texcount") is None:
        raise RuntimeError(
            "texcount not found on PATH. Install via your TeX distribution "
            "(e.g. `tlmgr install texcount` or your package manager)."
        )
    cmd = ["texcount", "-sum"]
    if include_subcounts:
        cmd.append("-sub")
    cmd.append(str(tex_path))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"texcount failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def parse_texcount(output: str) -> tuple[dict, list[SectionCount]]:
    summary: dict = {}
    sections: list[SectionCount] = []
    for line in output.splitlines():
        m = re.match(r"^Sum count:\s+(\d+)", line)
        if m:
            summary["sum_count"] = int(m.group(1))
            continue
        m = re.match(r"^Words in text:\s+(\d+)", line)
        if m:
            summary["body"] = int(m.group(1))
            continue
        m = re.match(r"^Words in headers:\s+(\d+)", line)
        if m:
            summary["headers"] = int(m.group(1))
            continue
        m = re.match(r"^Words outside text \(captions, etc\.\):\s+(\d+)", line)
        if m:
            summary["captions"] = int(m.group(1))
            continue
        m = re.match(r"^Number of headers:\s+(\d+)", line)
        if m:
            summary["n_headers"] = int(m.group(1))
            continue
        m = re.match(r"^Number of floats/tables/figures:\s+(\d+)", line)
        if m:
            summary["n_floats"] = int(m.group(1))
            continue
        m = re.match(r"^Number of math inlines:\s+(\d+)", line)
        if m:
            summary["n_math_inline"] = int(m.group(1))
            continue
        m = re.match(r"^Number of math displayed:\s+(\d+)", line)
        if m:
            summary["n_math_displayed"] = int(m.group(1))
            continue
        m = RE_SUBCOUNT.match(line)
        if m:
            sections.append(
                SectionCount(
                    kind=m["kind"],
                    title=m["title"],
                    text=int(m["text"]),
                    headers=int(m["headers"]),
                    captions=int(m["captions"]),
                )
            )
    return summary, sections


def render_text(report: WordCountReport) -> str:
    lines: list[str] = []
    lines.append(f"word_count.py — {report.tex_path}")
    lines.append("=" * 60)
    lines.append(f"  sum count           : {report.sum_count:>7,}")
    lines.append(f"  body                : {report.body:>7,}  (Words in text)")
    lines.append(f"  headers             : {report.headers:>7,}")
    lines.append(f"  captions / outside  : {report.captions:>7,}")
    lines.append(
        f"  headers/floats/$inline/$display : "
        f"{report.n_headers} / {report.n_floats} / "
        f"{report.n_math_inline} / {report.n_math_displayed}"
    )

    if report.target is not None:
        sign = "+" if report.margin and report.margin > 0 else ""
        verdict = "OVER target" if report.over_target else "OK"
        lines.append("")
        lines.append(
            f"  target              : {report.target:>7,}  "
            f"(body {sign}{report.margin:,} → {verdict})"
        )

    if report.per_section:
        lines.append("")
        lines.append("  per-section (text + headers + captions):")
        for sec in report.per_section:
            lines.append(
                f"    {sec.text:>5}+{sec.headers}+{sec.captions:<3} "
                f" {sec.kind}: {sec.title}"
            )

    return "\n".join(lines)


def render_json(report: WordCountReport) -> str:
    payload = asdict(report)
    payload["over_target"] = report.over_target
    payload["margin"] = report.margin
    return json.dumps(payload, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report LaTeX manuscript word counts via texcount, with optional "
            "target comparison and per-section breakdown. Useful during a "
            "word-count-reduction pass against a journal limit."
        )
    )
    parser.add_argument("tex_file", type=Path, help=".tex source to count.")
    parser.add_argument(
        "--target", type=int, default=None,
        help="Journal word limit. If set, report body vs target and exit 1 when over.",
    )
    parser.add_argument(
        "--per-section", action="store_true",
        help="Include per-section breakdown (one row per \\section / \\subsection).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--soft", action="store_true",
        help="Always exit 0, even when body exceeds --target (default: exit 1 on overshoot).",
    )
    args = parser.parse_args()

    if not args.tex_file.exists():
        print(f"error: file not found: {args.tex_file}", file=sys.stderr)
        return 2

    output = run_texcount(args.tex_file, include_subcounts=args.per_section)
    summary, sections = parse_texcount(output)

    report = WordCountReport(
        tex_path=str(args.tex_file),
        sum_count=summary.get("sum_count", 0),
        body=summary.get("body", 0),
        headers=summary.get("headers", 0),
        captions=summary.get("captions", 0),
        n_headers=summary.get("n_headers", 0),
        n_floats=summary.get("n_floats", 0),
        n_math_inline=summary.get("n_math_inline", 0),
        n_math_displayed=summary.get("n_math_displayed", 0),
        target=args.target,
        per_section=sections if args.per_section else [],
    )

    print(render_json(report) if args.json else render_text(report))

    if args.soft:
        return 0
    return 1 if report.over_target else 0


if __name__ == "__main__":
    sys.exit(main())
