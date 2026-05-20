"""Smoke tests for scripts/{response_align,word_count,dash_audit,bib_unused}.py.

Each script gets at least one positive case (clean input, expected exit-0
behaviour) and one negative case (deliberate drift, expected exit-1).

Tests invoke the scripts as subprocesses to exercise the real CLI surface,
including the exit-code policy promised in --help.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a script via the current Python interpreter, return CompletedProcess."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# response_align.py
# ---------------------------------------------------------------------------


def _write_aligned_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Three documents that are internally consistent — no drift expected."""
    tex = tmp_path / "main.tex"
    tex.write_text(
        r"""\documentclass{article}
\begin{document}
\section{Intro}
Sender chooses signal $s$ and sends it to receiver.
The new wording: a quantitative dependence on initialization.
\end{document}
""",
        encoding="utf-8",
    )
    checklist = tmp_path / "checklist.md"
    checklist.write_text(
        """# Checklist

| Comment | Title | Status |
|---|---|---|
| R1·C1 | Test comment | `[x]` done (2026-01-01) |

### [R1 · C1] Test comment

**Reviewer's concern:** placeholder.

**Actions:**

- [x] Rewrite the intro. — *Now reads as the new wording at [main.tex:4](../main.tex#L4).* (done 2026-01-01)
""",
        encoding="utf-8",
    )
    responses = tmp_path / "responses.md"
    responses.write_text(
        """Response to Reviewers

\\===== Reviewer \\#1 \\=====

\\--- Comment 1: Test comment \\---

Reviewer: placeholder.

Response: the intro now reads: *"a quantitative dependence on initialization."*
""",
        encoding="utf-8",
    )
    return checklist, responses, tex


def test_response_align_clean(tmp_path):
    checklist, responses, tex = _write_aligned_fixture(tmp_path)
    r = run(
        "response_align.py",
        "--checklist", str(checklist),
        "--responses", str(responses),
        "--manuscript", str(tex),
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "Total drift items: 0" in r.stdout


def test_response_align_detects_checklist_without_narrative(tmp_path):
    """An [x] checklist item with no corresponding narrative entry is drift."""
    checklist, responses, tex = _write_aligned_fixture(tmp_path)
    # Add a second [x] checklist item with no matching response narrative.
    extended = checklist.read_text() + (
        "\n### [R1 · C2] Orphan comment\n\n"
        "**Actions:**\n- [x] Did the thing.\n"
    )
    checklist.write_text(extended)
    # Also update the top-line table so the box is detected.
    text = checklist.read_text().replace(
        "| R1·C1 | Test comment | `[x]` done (2026-01-01) |",
        "| R1·C1 | Test comment | `[x]` done (2026-01-01) |\n"
        "| R1·C2 | Orphan comment | `[x]` done (2026-01-02) |",
    )
    checklist.write_text(text)
    r = run(
        "response_align.py",
        "--checklist", str(checklist),
        "--responses", str(responses),
        "--manuscript", str(tex),
    )
    assert r.returncode == 1, f"stdout:\n{r.stdout}"
    assert "R1·C2" in r.stdout


def test_response_align_soft_flag_forces_exit_zero(tmp_path):
    """--soft suppresses exit 1 even when drift is present."""
    checklist, responses, tex = _write_aligned_fixture(tmp_path)
    checklist.write_text(
        checklist.read_text()
        + "\n### [R1 · C9] Orphan\n\n**Actions:**\n- [x] x\n"
    )
    checklist.write_text(
        checklist.read_text().replace(
            "| R1·C1 | Test comment | `[x]` done (2026-01-01) |",
            "| R1·C1 | Test comment | `[x]` done (2026-01-01) |\n"
            "| R1·C9 | Orphan | `[x]` done (2026-01-01) |",
        )
    )
    r = run(
        "response_align.py",
        "--checklist", str(checklist),
        "--responses", str(responses),
        "--manuscript", str(tex),
        "--soft",
    )
    assert r.returncode == 0, r.stdout


def test_response_align_json_output(tmp_path):
    checklist, responses, tex = _write_aligned_fixture(tmp_path)
    r = run(
        "response_align.py",
        "--checklist", str(checklist),
        "--responses", str(responses),
        "--manuscript", str(tex),
        "--json",
    )
    assert r.returncode == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["checklist_without_narrative"] == []
    assert payload["narrative_without_checklist"] == []


# ---------------------------------------------------------------------------
# word_count.py
# ---------------------------------------------------------------------------

texcount_available = shutil.which("texcount") is not None


@pytest.fixture
def trivial_tex(tmp_path: Path) -> Path:
    p = tmp_path / "doc.tex"
    p.write_text(
        r"""\documentclass{article}
\begin{document}
\section{Introduction}
One two three four five six seven eight nine ten.
Eleven twelve thirteen fourteen fifteen.
\end{document}
""",
        encoding="utf-8",
    )
    return p


@pytest.mark.skipif(not texcount_available, reason="texcount not installed")
def test_word_count_under_target(trivial_tex):
    r = run("word_count.py", str(trivial_tex), "--target", "1000")
    assert r.returncode == 0, r.stdout
    assert "OK" in r.stdout


@pytest.mark.skipif(not texcount_available, reason="texcount not installed")
def test_word_count_over_target(trivial_tex):
    r = run("word_count.py", str(trivial_tex), "--target", "5")
    assert r.returncode == 1, r.stdout
    assert "OVER target" in r.stdout


@pytest.mark.skipif(not texcount_available, reason="texcount not installed")
def test_word_count_soft_flag(trivial_tex):
    r = run("word_count.py", str(trivial_tex), "--target", "5", "--soft")
    assert r.returncode == 0


@pytest.mark.skipif(not texcount_available, reason="texcount not installed")
def test_word_count_json(trivial_tex):
    r = run("word_count.py", str(trivial_tex), "--target", "1000", "--json")
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["target"] == 1000
    assert payload["over_target"] is False


# ---------------------------------------------------------------------------
# dash_audit.py
# ---------------------------------------------------------------------------


@pytest.fixture
def dash_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "doc.tex"
    p.write_text(
        "First line --- with one LaTeX em-dash.\n"
        "Second line — with a Unicode em-dash.\n"
        "Third line with en-dash for ranges -- like Lewis--Skyrms (should not count).\n"
        "Fourth -- and -- and -- still en-dashes.\n",
        encoding="utf-8",
    )
    return p


def test_dash_audit_counts(dash_fixture):
    r = run("dash_audit.py", str(dash_fixture))
    assert r.returncode == 0
    assert "latex 1" in r.stdout
    assert "unicode 1" in r.stdout


def test_dash_audit_max_triggers_fail(dash_fixture):
    r = run("dash_audit.py", str(dash_fixture), "--max", "0")
    assert r.returncode == 1
    assert "OVER --max" in r.stdout


def test_dash_audit_soft_suppresses_fail(dash_fixture):
    r = run("dash_audit.py", str(dash_fixture), "--max", "0", "--soft")
    assert r.returncode == 0


def test_dash_audit_count_only(dash_fixture):
    r = run("dash_audit.py", str(dash_fixture), "--count-only")
    assert r.returncode == 0
    # No per-occurrence line:column listing in count-only mode.
    assert ":1:" not in r.stdout and ":2:" not in r.stdout


def test_dash_audit_json(dash_fixture):
    r = run("dash_audit.py", str(dash_fixture), "--json")
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["total_latex"] == 1
    assert payload["total_unicode"] == 1


# ---------------------------------------------------------------------------
# bib_unused.py
# ---------------------------------------------------------------------------


@pytest.fixture
def bib_fixture(tmp_path: Path) -> tuple[Path, Path]:
    bib = tmp_path / "refs.bib"
    bib.write_text(
        "@article{cited_a, title={A}, author={X}, year=2020}\n"
        "@article{cited_b, title={B}, author={Y}, year=2021}\n"
        "@book{never_cited, title={C}, author={Z}, year=2022}\n",
        encoding="utf-8",
    )
    tex = tmp_path / "doc.tex"
    tex.write_text(
        r"""\documentclass{article}
\begin{document}
Per \citep{cited_a}, the result holds. See also \citet{cited_b}.
\end{document}
""",
        encoding="utf-8",
    )
    return bib, tex


def test_bib_unused_reports_unused(bib_fixture):
    bib, tex = bib_fixture
    r = run("bib_unused.py", str(bib), str(tex))
    assert r.returncode == 1, r.stdout
    assert "never_cited" in r.stdout
    assert "cited_a" not in r.stdout.split("[A.")[1].split("[")[0]


def test_bib_unused_clean_case_exits_zero(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text("@article{only_one, title={X}, year=2020}\n", encoding="utf-8")
    tex = tmp_path / "doc.tex"
    tex.write_text(r"See \citep{only_one}.", encoding="utf-8")
    r = run("bib_unused.py", str(bib), str(tex))
    assert r.returncode == 0


def test_bib_unused_reverse_detects_typos(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text("@article{key_one, title={A}, year=2020}\n", encoding="utf-8")
    tex = tmp_path / "doc.tex"
    tex.write_text(
        r"See \citep{key_one} and also \citep{key_typo}.",
        encoding="utf-8",
    )
    # Without --strict, exit 0 because A check passes (no unused).
    r = run("bib_unused.py", str(bib), str(tex), "--reverse")
    assert r.returncode == 0
    assert "key_typo" in r.stdout
    # --strict promotes B-direction drift to a non-zero exit.
    r = run("bib_unused.py", str(bib), str(tex), "--reverse", "--strict")
    assert r.returncode == 1


def test_bib_unused_json(bib_fixture):
    bib, tex = bib_fixture
    r = run("bib_unused.py", str(bib), str(tex), "--json")
    payload = json.loads(r.stdout)
    assert "never_cited" in payload["unused_entries"]
    assert payload["missing_keys"] == []
    assert payload["has_drift"] is True
