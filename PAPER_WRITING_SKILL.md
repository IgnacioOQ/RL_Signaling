---
status: active
type: how-to
id: paper_writing_skill
description: Writing-style and editorial conventions for this author's philosophy papers — philosophical-essay voice, concision calibrated to section function, no meta-reflexivity, a confident non-defensive tone, and strict separation of manuscript prose from reviewer-response prose.
label: [skill, agent]
repository: [RL_Signaling]
injection: procedural
volatility: evolving
scope: general
last_checked: '2026-05-20'
---

# Paper Writing Skill: Philosophical-Essay Voice

This skill encodes the writing style and editorial taste of one author: a philosopher who writes philosophy papers that engage formal and computational material (signaling games, reinforcement learning) without becoming technical reports. It was distilled from a revision pass on the manuscript *Signaling Games with Distributed Rewards* (PHOS-17993) and from the author's own margin comments. It is descriptive of one author's calibrated taste, not generic writing advice. Reach for it when drafting prose in this author's voice, when editing existing prose toward it, or when an editorial decision turns on tone, concision, or structure. When a choice is genuinely open, prefer the option this document points to; when the document is silent, ask rather than guess.

## When to use this skill

Load this skill when drafting or revising a paper for this author, when proposing prose edits, or when an editorial decision turns on tone, concision, or structure. It pairs with — but is distinct from — any project-specific LaTeX or revision-workflow reference: this document governs *how the prose reads*, not how the build works.

## The voice

The target register is the **philosophical essay**: an argument made by a present, first-person author, in plain and precise English, to a reader who is intelligent but not a specialist in every tool the paper uses.

- **First person, and committed.** The author writes "I argue", "I do not hold this view", "I suspend judgment on this". The author takes positions and owns them. Do not retreat into "it is argued that" or "one might conclude".
- **Conversational but exact.** Sentences can open inquiry with a question ("What is the genesis of such a covenant?") and can address the reader directly. This never licenses vagueness: every sentence still has to be true and precise.
- **Plain words first.** Prefer Anglo-Saxon, concrete diction over abstract Latinate diction where both carry the meaning. "Use", not "utilize". "Show", not "demonstrate", unless "demonstrate" is doing real work.
- **Scholarly, but the author's thread leads.** Claims are situated against the literature (here: Grice, Lewis, Skyrms, Gilbert, Huttegger, Millikan). Citations support the argument; they do not replace it. The author's own line of thought is always the spine.
- **A paper may end warmly.** A serious argument can close on a human image or a recurring motif. Do not sand a deliberate literary closing into a flat summary.

## Principles

### Concision is calibrated to a section's job

Length and detail follow function. A section that exists to *set up* an idea is not a section that exists to *report results*, and it should not borrow the other's granularity.

- A proof-of-concept or setup section makes **qualitative** claims. Per-parameter numbers, score tables, and slope-by-slope readings belong in a results section, where reporting them is the job. Author's own words: *"This is a proof of concept, not the simulation results section."*
- When you must cut, cut by **function**, not by uniform percentage. Trim framing and scaffolding to the bone; trim substantive argument gently.

### The paper is not about itself

The manuscript advances an argument. It should not narrate its own structure or classify its own claims more than minimally.

- One lean signpost is enough ("The remainder of this introduction sets up the background"). A second clause explaining what the signpost does *not* do is meta-clutter — cut it.
- Never write a paragraph whose job is to taxonomize the paper's own claims ("A note on the kind of claim being made", "the discussion distinguishes three layers"). If two such paragraphs appear back to back, that is the clearest possible signal to merge and shrink.
- A reader should be able to follow the argument without being told, repeatedly, how to read it.

### Confident, never defensive

State limitations — that is honest scholarship — but state them as facts, not as confessions.

- No apology, no self-deprecation. The sentence *"For this I apologize, but the mathematics of dynamical systems is not my strength"* was cut for exactly this reason. An honest limitation reads: "A proof of convergence is not attempted here; that is left open."
- Downgrade self-flagellating intensifiers: "severely limited" becomes "limited".
- No hedge stacks. One clear caveat, not "this suggests ... which may indicate ... such that it appears". A single honest "this is a conjecture" is stronger than three layers of qualification.

### Say each thing once

Redundancy is the most common avoidable bloat.

- A figure caption states what the reader is looking at. It does **not** re-explain what the body already explains. If caption and body say the same thing, the caption loses.
- If a section largely restates another, it should be absorbed or cut. In this manuscript an entire "Informal Proof of Concept" subsection was removed because it duplicated the formal proof-of-concept section.
- Two paragraphs that make the same positioning move ("our approach is like the evolutionary one in that...") are one paragraph.

### Concrete anchors and narrative threads

Abstraction is carried by concrete cases, and the concrete cases recur.

- Thought-experiment scenarios (here: Sue and Jack walking; the Strangers' Game) are introduced once and **returned to**. They thread the paper and give the reader a fixed point. Preserve those threads when editing.
- Worked examples are didactic, not decorative. When the proof-of-concept paragraphs were compressed, the concrete extremes (the *n = 1* and *n = 100* cases, with the "≈ 99%" first-episode figure) were deliberately kept. Concreteness is not flab.
- When in doubt, cut the framing around an example, not the example.

### Punctuation, emphasis, and rhythm

- **Avoid the em-dash as rhythmic punctuation.** Where a sentence reaches for a dash pair or a dash-aside, use a colon, a semicolon, a comma, or a new sentence. (En-dashes inside proper names — Roth–Erev, Lewis–Skyrms — are correct typography and stay.)
- **Do not overuse italic emphasis.** Reserve italics for introducing a technical term on first use, or for a genuine contrast the reader would otherwise miss. Italic-for-importance, scattered, deadens the page.
- Vary sentence length deliberately; do not let every sentence settle into the same clause-comma-clause shape.

### Every paragraph has a job

You should be able to name what each paragraph does in a few words. If you cannot, the paragraph is unfocused or unnecessary.

- Sections flow into one another. End a section by motivating the next ("That is the quantitative question the following section takes up"), rather than stopping flat.
- Enumerate — "First ... Second ... Third", "(a)/(b)/(c)" — **only when the content is genuinely an N-fold list of distinct items.** Three real, separable observations may be numbered. A single flowing thought dressed up as three points is an LLM tell; write it as prose.
- Do not reach for a parenthetical list of three by reflex.

### Name things precisely

- Prefer the concrete, correctly named object over a vague one. When a payoff structure *is* a matrix, call it the matrix **M**, not "the game **G**". When a formal object has a standard tuple form, give the tuple.
- Keep the precise reason, not the gesture at one. If there is a real mechanism behind a claim (a named theorem, a specific result, an identified obstruction), state it compactly. "No general convergence theorem is available, because the cooperative-case proof relies on a single shared potential that distributed rewards remove" beats "this is hard to prove in general".

## Manuscript prose versus reviewer-response prose

This is the distinction the author flags most sharply: **the manuscript and the response-to-reviewers are two different documents with two different jobs, and their content must not mix.**

- The **manuscript** states the claim, makes the argument, and reads as though written for a reader who never saw the reviews. It does not defend its own choices, does not say "what this is and what this is not" for the reviewers' benefit, and does not carry caveats that exist only to pre-empt an objection.
- The **response-to-reviewers** defends the choices, explains what was changed and why, and quotes the manuscript. Scaffolding of the form "we have addressed this by...", point-by-point self-classification, and apologetic framing all live here.
- When a passage in the manuscript reads as if it is answering a reviewer rather than informing a reader, move its substance into the response document and delete it from the manuscript.
- Consequence for revision work: when manuscript prose that a response document quotes is edited, the response's quotations must be re-synced in lockstep. Substance the reviewer asked for must survive the edit, even when the wording around it changes.

## The keep / cut calculus

When trimming for length or tightening for voice, the asymmetry below is the decision aid. Cut the left column without hesitation; protect the right.

| Cut hard | Protect |
|:--|:--|
| Meta-paragraphs about the paper's structure or its own claim-types | The substantive argument and its steps |
| Reviewer-facing defense and "what this is / is not" scaffolding | Citations, and the precise named mechanism behind a claim |
| Redundancy: caption restating body, section restating section | Concrete worked examples and recurring thought experiments |
| Hedge stacks, apology, self-deprecation | A confident first-person stance |
| Decorative "First/Second/Third" over a single thought | Genuine N-fold enumeration of distinct items |
| Em-dash used for rhythm | En-dashes inside proper names |
| Per-parameter numbers in a setup or proof-of-concept section | Numbers in a section whose job is to report them |
|  | A literary closing image that has earned its place |

## LLM tells to avoid

Prose drafted by a language model tends to carry these markers. Treat each as a defect to remove:

- The em-dash used as a connective rhythm device.
- Meta openers: "It is worth noting that", "A note on", "Importantly,".
- Claim-taxonomy paragraphs that classify the paper's own contributions.
- "First, ... Second, ... Third, ..." imposed on a single continuous idea.
- Parenthetical lists of exactly three, used reflexively.
- Hedge stacks: "suggests ... may indicate ... appears to ... could potentially".
- "In other words" followed by a near-identical restatement.
- Abstract-Latinate diction where a plain word carries the same meaning.
- Italic emphasis sprayed across a paragraph.
- Captions that re-explain what the body already says.
- Apologetic or self-deprecating framing of a limitation.

## Quick checklist

Run this before declaring a draft or an edit finished:

1. Does every paragraph have a job you can name in a few words?
2. Is the level of detail right for *this* section's role (setup vs. results)?
3. Is anything here meta-commentary about the paper itself? Cut or shrink it.
4. Does anything read as a reply to a reviewer rather than as text for a reader? Move it.
5. Are limitations stated as facts, with no apology and no hedge stack?
6. Any em-dash used for rhythm? Replace with a colon, semicolon, comma, or full stop.
7. Does any caption duplicate the body? Does any section duplicate another?
8. Are enumerated lists genuinely N-fold, or decoration on one idea?
9. Is the first-person, committed authorial voice intact?
10. Do the recurring concrete examples and the closing image survive the edit?

## Notes

This document is descriptive and evolving. It was distilled on 2026-05-20 from a revision pass on PHOS-17993 (*Signaling Games with Distributed Rewards*) and from the author's `writing_comments.md`. It captures one author's taste at one point in time; refine it as further preferences are articulated, and do not treat it as a universal style authority. When it conflicts with an explicit instruction from the author in the moment, the instruction wins.
