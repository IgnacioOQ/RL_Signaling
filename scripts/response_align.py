#!/usr/bin/env python3
"""
response_align.py — drift checker for paper-revision artifacts.

Catches drift between the three coupled documents of a revision arc:

  1. Operational checklist  (e.g. "responses_checklist.md")
  2. Formal response document  (e.g. "responses_to_reviewers.md")
  3. Manuscript LaTeX source  (one or more .tex files)

Four checks:

  A. Checklist [x] top-level items without a matching response-narrative entry.
  B. Response-narrative entries without a matching checklist top-level item.
  C. Italicized verbatim quotes  *"..."*  in the response narrative that no
     longer appear anywhere in the .tex source(s).
  D. Sub-bullet [x] items whose [main_v2.tex:NNN] anchor no longer points at
     content matching the bullet's quoted "now reads" / "has been replaced
     with" fragment.

Exit 0 if no drift; exit 1 otherwise.  Use --soft to force exit 0.

CLI:
    python scripts/response_align.py \\
        --checklist  "manuscript/reviewers/responses_checklist.md" \\
        --responses  "manuscript/reviewers/responses_to_reviewers.md" \\
        --manuscript manuscript/main_v2.tex manuscript/Appendix.tex
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Checklist: "### [R2 · C1] What exactly is the paper showing?"
# Tolerate optional spaces around "·" and arbitrary unicode middle-dot variants.
RE_CHECKLIST_COMMENT = re.compile(
    r"^###\s+\[\s*R\s*(?P<rev>\d+)\s*[·•]\s*C\s*(?P<com>\d+)\s*\]\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)

# Top-level status box on the immediately-following table row, e.g.
#   | R2·C1 | What exactly... | `[x]` done (2026-05-19) |
RE_TOP_TABLE_ROW = re.compile(
    r"^\|\s*R\s*(?P<rev>\d+)\s*[·•]\s*C\s*(?P<com>\d+)\s*\|.*?\|\s*"
    r"`?\[(?P<box>[ x~])\]`?.*?\|",
    re.MULTILINE,
)

# Sub-bullet: "- [x] §2.3: ..." (possibly multi-line; we capture to next bullet)
RE_SUBBULLET = re.compile(
    r"^-\s+\[(?P<box>[ x~])\]\s+(?P<body>.+?)(?=\n-\s+\[|\n###|\n##|\n---|\Z)",
    re.MULTILINE | re.DOTALL,
)

# Response narrative reviewer block: optional leading "\\" allowed before =====
RE_RESP_REVIEWER = re.compile(
    r"^\\?={3,}\s*Reviewer\s*\\?#?\s*(?P<rev>\d+)\s*\\?={3,}\s*$",
    re.MULTILINE,
)

# Response narrative comment block, e.g. "\\--- Comment 1: title \\---"
RE_RESP_COMMENT = re.compile(
    r"^\\?-{3,}\s*Comment\s+(?P<com>\d+)\s*:\s*(?P<title>.+?)\s*\\?-{3,}\s*$",
    re.MULTILINE,
)

# Italicized double-quoted string used as verbatim manuscript quote.
# Accept straight or curly quotes; permit either order of asterisk and quote.
RE_VERBATIM_QUOTE = re.compile(
    r'\*[\"“](?P<text>.+?)[\"”]\*',
    re.DOTALL,
)

# Markdown-style file anchor: [main_v2.tex:184] or [main_v2.tex:184](../main_v2.tex#L184)
RE_ANCHOR = re.compile(r"\[(?P<file>[\w./-]+\.tex):(?P<line>\d+)\]")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChecklistComment:
    rev: int
    com: int
    title: str
    box: str  # 'x' | '~' | ' '
    body: str
    subbullets: list["SubBullet"] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"R{self.rev}·C{self.com}"


@dataclass
class SubBullet:
    box: str
    body: str
    anchors: list[tuple[str, int]]  # (file, line)
    quotes: list[str]  # italic verbatim quotes inside the bullet note


@dataclass
class ResponseComment:
    rev: int
    com: int
    title: str
    body: str
    quotes: list[str]
    # Per-quote classification — only "post" quotes are checked against the .tex.
    # A "pre" quote is the manuscript wording the revision removed, which is
    # expected to be absent from the current .tex.
    quote_classes: list[str] = field(default_factory=list)  # "pre" | "post"

    @property
    def id(self) -> str:
        return f"R{self.rev}·C{self.com}"


@dataclass
class DriftReport:
    checklist_path: str
    responses_path: str
    manuscript_paths: list[str]
    checklist_without_narrative: list[str] = field(default_factory=list)
    narrative_without_checklist: list[str] = field(default_factory=list)
    missing_verbatim_quotes: list[dict] = field(default_factory=list)
    stale_anchors: list[dict] = field(default_factory=list)

    @property
    def total_drift(self) -> int:
        return (
            len(self.checklist_without_narrative)
            + len(self.narrative_without_checklist)
            + len(self.missing_verbatim_quotes)
            + len(self.stale_anchors)
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_checklist(text: str) -> list[ChecklistComment]:
    """Extract top-level comment blocks from the operational checklist."""
    # Locate top-table boxes for the high-level status.
    top_status: dict[str, str] = {}
    for m in RE_TOP_TABLE_ROW.finditer(text):
        cid = f"R{int(m['rev'])}·C{int(m['com'])}"
        top_status[cid] = m["box"]

    # Find all "### [R<n> · C<m>] title" headings; each spans up to the next
    # ### or ## or end of file.
    headings = list(RE_CHECKLIST_COMMENT.finditer(text))
    comments: list[ChecklistComment] = []
    for i, m in enumerate(headings):
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        # Stop at the next top-level section divider ("## " or "---" followed
        # by a "## " header) if it comes before the next ###.
        next_h2 = re.search(r"^##\s+", text[start:end], re.MULTILINE)
        if next_h2:
            end = start + next_h2.start()
        body = text[start:end].strip()

        rev, com = int(m["rev"]), int(m["com"])
        cid = f"R{rev}·C{com}"
        box = top_status.get(cid, " ")
        cc = ChecklistComment(rev=rev, com=com, title=m["title"], box=box, body=body)
        cc.subbullets = _parse_subbullets(body)
        comments.append(cc)
    return comments


def _parse_subbullets(body: str) -> list[SubBullet]:
    out: list[SubBullet] = []
    for m in RE_SUBBULLET.finditer(body):
        sb_body = m["body"].strip()
        anchors = [(a["file"], int(a["line"])) for a in RE_ANCHOR.finditer(sb_body)]
        quotes = [q["text"].strip() for q in RE_VERBATIM_QUOTE.finditer(sb_body)]
        out.append(SubBullet(box=m["box"], body=sb_body, anchors=anchors, quotes=quotes))
    return out


# Cues that mark a verbatim italic quote as PRE-revision (deleted manuscript
# wording) — these appear either before or after the quote in the surrounding
# narrative prose.  Pre-revision quotes are intentionally absent from the
# current .tex and should not be flagged as drift.
PRE_REV_CUES_PRECEDING = (
    "previous", "previously", "originally", "the original",
    "the old", "formerly", "earlier", "the dropped", "tightened from",
    "softened from", "was rewritten", "was replaced", "the deleted",
    "the redundant",
)
PRE_REV_CUES_FOLLOWING = (
    "has been replaced", "have been replaced", "was replaced", "were replaced",
    "has been dropped", "have been dropped", "was dropped",
    "has been deleted", "have been deleted",
    "has been expanded", "has been tightened", "has been rewritten",
    "is now", "now reads", "now opens", "now closes",
)


def classify_quote(narrative: str, q_start: int, q_end: int) -> str:
    """Return 'pre' if the quote is the deleted-manuscript half of a before /
    after pair, otherwise 'post'.  Quotes that don't sit inside a before / after
    pair are classified 'post' (checked against the .tex)."""
    pre_ctx = narrative[max(0, q_start - 240): q_start].lower()
    post_ctx = narrative[q_end: q_end + 160].lower()
    if any(cue in pre_ctx for cue in PRE_REV_CUES_PRECEDING):
        return "pre"
    if any(cue in post_ctx for cue in PRE_REV_CUES_FOLLOWING):
        return "pre"
    return "post"


def parse_responses(text: str) -> list[ResponseComment]:
    """Extract comment-narrative blocks from the formal response document."""
    # Build a sequence of (kind, pos, payload) tokens, sorted by file position.
    tokens: list[tuple[str, int, dict]] = []
    for m in RE_RESP_REVIEWER.finditer(text):
        tokens.append(("reviewer", m.start(), {"rev": int(m["rev"]), "end": m.end()}))
    for m in RE_RESP_COMMENT.finditer(text):
        tokens.append(
            (
                "comment",
                m.start(),
                {"com": int(m["com"]), "title": m["title"], "end": m.end()},
            )
        )
    tokens.sort(key=lambda t: t[1])

    out: list[ResponseComment] = []
    current_rev: int | None = None
    for i, (kind, _pos, payload) in enumerate(tokens):
        if kind == "reviewer":
            current_rev = payload["rev"]
            continue
        if kind == "comment":
            if current_rev is None:
                continue  # "Comment N" without a Reviewer header → skip
            start = payload["end"]
            end = tokens[i + 1][1] if i + 1 < len(tokens) else len(text)
            body = text[start:end].strip()
            quotes: list[str] = []
            classes: list[str] = []
            for m in RE_VERBATIM_QUOTE.finditer(body):
                quotes.append(m["text"].strip())
                classes.append(classify_quote(body, m.start(), m.end()))
            out.append(
                ResponseComment(
                    rev=current_rev,
                    com=payload["com"],
                    title=payload["title"].strip(),
                    body=body,
                    quotes=quotes,
                    quote_classes=classes,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Quote normalisation and matching
# ---------------------------------------------------------------------------

# Strip LaTeX commands of the form \cmd or \cmd{...} (non-greedy, single-level).
RE_LATEX_CMD_BRACES = re.compile(r"\\[a-zA-Z]+\*?\s*\{([^{}]*)\}")
RE_LATEX_CMD_BARE = re.compile(r"\\[a-zA-Z]+\*?")
# Order matters: handle $$...$$ display math first so the $...$ inline-math
# pass doesn't mispair its bracketing dollars and eat unrelated body text.
RE_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$", re.DOTALL)
RE_BRACKET_MATH = re.compile(r"\\\[.*?\\\]", re.DOTALL)
RE_PAREN_MATH = re.compile(r"\\\(.*?\\\)", re.DOTALL)
RE_INLINE_MATH = re.compile(r"\$[^$]+\$")
RE_WS = re.compile(r"\s+")


def normalize_for_match(s: str) -> str:
    """Aggressive normalisation for fuzzy quote-matching across .md and .tex."""
    # Drop \cite{...} entirely (the body is bib-key noise, not prose).
    s = re.sub(r"\\cite[pt]?\*?\{[^{}]*\}", " ", s)
    # Strip math first (before LaTeX-command passes) so display-math $$...$$
    # doesn't desynchronize the $...$ inline-math regex.
    s = RE_DISPLAY_MATH.sub(" ", s)
    s = RE_BRACKET_MATH.sub(" ", s)
    s = RE_PAREN_MATH.sub(" ", s)
    s = RE_INLINE_MATH.sub(" ", s)
    # Repeated passes for nested braces (\emph{\textit{x}} → x).
    for _ in range(3):
        s = RE_LATEX_CMD_BRACES.sub(r"\1", s)
    s = RE_LATEX_CMD_BARE.sub(" ", s)
    # Markdown escaped specials in the response narrative.
    s = s.replace("\\---", "---").replace("\\--", "--")
    s = s.replace("\\[", "[").replace("\\]", "]")
    s = s.replace("\\_", "_").replace("\\#", "#").replace("\\&", "&")
    # LaTeX-style double quotes ``...'' → "..."  (must run before single ').
    s = s.replace("``", '"').replace("''", '"')
    s = s.replace("`", "'")
    # Smart quotes / typographic dashes / ellipsis.
    s = (
        s.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
        .replace("—", "---").replace("–", "--")
        .replace("…", "...")
    )
    # Quote characters carry no semantic weight for prose matching — collapse them.
    s = s.replace('"', " ").replace("'", " ")
    # Strip lingering braces, then collapse whitespace and lowercase.
    s = s.replace("{", " ").replace("}", " ")
    s = RE_WS.sub(" ", s).strip().lower()
    return s


def _split_on_ellipsis(quote: str) -> list[str]:
    """Quotes with ``[...]`` elision break into independently-matched fragments."""
    parts = re.split(r"\\?\[\s*(?:\.\.\.|…)\s*\\?\]|\.{3,}|…", quote)
    return [p.strip() for p in parts if p.strip()]


def _split_into_sentences(fragment: str) -> list[str]:
    """Coarse sentence split — keeps the verbatim-quote check robust against
    a single divergent punctuation char anywhere in a multi-sentence quote."""
    # Split on .  !  ? followed by whitespace + capital, plus ';' before capital.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])|;\s+(?=[A-Z(])", fragment)
    return [p.strip() for p in parts if p.strip()]


def quote_appears_in_manuscript(
    quote: str,
    normalized_tex: str,
    min_fragment_len: int = 25,
    coverage_threshold: float = 0.66,
) -> tuple[bool, str | None]:
    """
    Quote matching with two levels of granularity:
      1. Split on '[...]' elision markers — each elision-fragment is independent.
      2. For each elision-fragment, split into sentences and require at least
         `coverage_threshold` of sentences (of length >= min_fragment_len after
         normalisation) to appear as substrings of the manuscript.

    A quote with three sentences and one minor wording drift in one sentence
    still counts as present (2/3 ≥ 0.66). A quote whose entire content is
    absent reports the first missing sentence as the diagnostic.
    """
    elision_fragments = _split_on_ellipsis(quote)
    for frag in elision_fragments:
        sentences = _split_into_sentences(frag) or [frag]
        scorable = []
        for sent in sentences:
            nsent = normalize_for_match(sent)
            if len(nsent) < min_fragment_len:
                continue
            scorable.append((sent, nsent in normalized_tex, nsent))
        if not scorable:
            continue  # nothing substantive to match in this elision-fragment
        hits = sum(1 for _, ok, _ in scorable if ok)
        coverage = hits / len(scorable)
        if coverage < coverage_threshold:
            # Return the first sentence we couldn't find as the diagnostic.
            for sent, ok, _ in scorable:
                if not ok:
                    return False, sent
            return False, frag
    return True, None


# ---------------------------------------------------------------------------
# Drift checks
# ---------------------------------------------------------------------------


def check_drift(
    checklist: list[ChecklistComment],
    responses: list[ResponseComment],
    manuscripts: dict[str, str],
    strict_quotes: bool = False,
) -> DriftReport:
    report = DriftReport(
        checklist_path="",  # filled by caller
        responses_path="",
        manuscript_paths=list(manuscripts.keys()),
    )

    # Map IDs across both documents.
    checklist_ids = {c.id: c for c in checklist}
    response_ids = {r.id: r for r in responses}

    # Check A — checklist [x] without narrative.
    for cid, cc in checklist_ids.items():
        if cc.box == "x" and cid not in response_ids:
            report.checklist_without_narrative.append(cid)

    # Check B — narrative without checklist.
    for cid in response_ids:
        if cid not in checklist_ids:
            report.narrative_without_checklist.append(cid)

    # Check C — italic verbatim quotes in narrative not found in any manuscript.
    normalized_tex_all = "\n".join(
        normalize_for_match(text) for text in manuscripts.values()
    )
    for rc in responses:
        for quote, qclass in zip(rc.quotes, rc.quote_classes):
            if qclass == "pre" and not strict_quotes:
                continue  # pre-revision quote: legitimately absent from .tex
            ok, missing_frag = quote_appears_in_manuscript(quote, normalized_tex_all)
            if not ok:
                report.missing_verbatim_quotes.append(
                    {
                        "comment_id": rc.id,
                        "class": qclass,
                        "quote": quote if len(quote) <= 240 else quote[:240] + "…",
                        "missing_fragment": missing_frag,
                    }
                )

    # Check D — stale anchors in checklist sub-bullets.
    tex_lines: dict[str, list[str]] = {
        Path(p).name: text.splitlines() for p, text in manuscripts.items()
    }
    for cc in checklist:
        for sb in cc.subbullets:
            if sb.box != "x":
                continue
            for fname, lineno in sb.anchors:
                key = Path(fname).name
                if key not in tex_lines:
                    continue  # anchor points at a file not provided; skip
                lines = tex_lines[key]
                if not (1 <= lineno <= len(lines)):
                    report.stale_anchors.append(
                        {
                            "comment_id": cc.id,
                            "anchor": f"{fname}:{lineno}",
                            "reason": "line out of range",
                        }
                    )
                    continue
                # Scan ±5 lines around the anchor.
                window = "\n".join(
                    lines[max(0, lineno - 6) : min(len(lines), lineno + 5)]
                )
                normalized_window = normalize_for_match(window)
                # Only validate when the bullet carries a quoted fragment.
                # Bullets with no quoted "now reads" / "replaced with" string
                # are not actionable for fuzzy match.
                for q in sb.quotes:
                    ok, missing_frag = quote_appears_in_manuscript(
                        q, normalized_window
                    )
                    if not ok:
                        report.stale_anchors.append(
                            {
                                "comment_id": cc.id,
                                "anchor": f"{fname}:{lineno}",
                                "missing_fragment": missing_frag,
                                "quote": q if len(q) <= 200 else q[:200] + "…",
                            }
                        )
                        break  # one report per anchor is enough

    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_text(report: DriftReport) -> str:
    lines: list[str] = []
    lines.append("response_align.py — drift report")
    lines.append("=" * 60)
    lines.append(f"checklist  : {report.checklist_path}")
    lines.append(f"responses  : {report.responses_path}")
    lines.append(f"manuscript : {', '.join(report.manuscript_paths)}")
    lines.append("")

    def section(name: str, items: list, fmt) -> None:
        lines.append(f"[{name}] ({len(items)})")
        if not items:
            lines.append("  none")
        else:
            for it in items:
                lines.append(f"  - {fmt(it)}")
        lines.append("")

    section(
        "A. Checklist [x] items without a response-narrative entry",
        report.checklist_without_narrative,
        lambda cid: cid,
    )
    section(
        "B. Narrative entries without a checklist item",
        report.narrative_without_checklist,
        lambda cid: cid,
    )
    section(
        "C. Verbatim quotes in narrative not found in manuscript",
        report.missing_verbatim_quotes,
        lambda it: (
            f"{it['comment_id']}: missing fragment "
            f"{it['missing_fragment']!r} — quote: {it['quote']!r}"
        ),
    )
    section(
        "D. Stale checklist anchors (line no longer matches documented edit)",
        report.stale_anchors,
        lambda it: (
            f"{it['comment_id']} @ {it['anchor']}: "
            + (
                f"missing fragment {it['missing_fragment']!r}"
                if "missing_fragment" in it
                else it.get("reason", "(unknown)")
            )
        ),
    )

    lines.append(
        f"Total drift items: {report.total_drift}"
        + ("  — OK" if report.total_drift == 0 else "")
    )
    return "\n".join(lines)


def render_json(report: DriftReport) -> str:
    return json.dumps(asdict(report), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drift checker for paper-revision artifacts: catches inconsistencies"
            " across the operational checklist, the formal response narrative,"
            " and the manuscript LaTeX source."
        )
    )
    parser.add_argument(
        "--checklist", required=True, type=Path,
        help="Path to the operational checklist (markdown).",
    )
    parser.add_argument(
        "--responses", required=True, type=Path,
        help="Path to the formal response narrative (markdown).",
    )
    parser.add_argument(
        "--manuscript", required=True, type=Path, nargs="+",
        help="One or more .tex source files to validate quotes and anchors against.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--soft", action="store_true",
        help="Always exit 0, even when drift is detected (default: exit 1 on drift).",
    )
    parser.add_argument(
        "--strict-quotes", action="store_true",
        help=(
            "Check every italic verbatim quote in the response narrative, "
            "including before-quotes that the revision deleted from the .tex. "
            "Default: skip quotes detected as the pre-revision side of a "
            "before/after pair (those are expected to be absent from the .tex)."
        ),
    )
    args = parser.parse_args()

    checklist = parse_checklist(args.checklist.read_text(encoding="utf-8"))
    responses = parse_responses(args.responses.read_text(encoding="utf-8"))
    manuscripts = {
        str(p): p.read_text(encoding="utf-8") for p in args.manuscript
    }

    report = check_drift(
        checklist, responses, manuscripts, strict_quotes=args.strict_quotes
    )
    report.checklist_path = str(args.checklist)
    report.responses_path = str(args.responses)

    output = render_json(report) if args.json else render_text(report)
    print(output)

    if args.soft:
        return 0
    return 1 if report.total_drift else 0


if __name__ == "__main__":
    sys.exit(main())
