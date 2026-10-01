# What this lab also measures: the AI's capabilities and limits

This Track-B lab is a real PRNG-fingerprinting experiment, and it is also a probe
of what an AI coding agent does when handed an open-ended research task. This file
records the second reading honestly, including where the agent stopped and why.
It is observational: it describes what happened across the rounds, not a claim
about what the agent "can" do in general.

## What the agent did well

- **Held a method across many sessions.** Rounds 0-4 kept the same discipline:
  hypothesis and decision metric written to `lab/reports/round-<n>.md` *before*
  any run, disjoint train/val/test seed ranges, a CSPRNG control that must stay
  at chance, and a 15-minute wall-clock cap. The rules in `lab/BOARD.md` were not
  re-litigated each round.
- **Chose the right tool for the structure, not the most powerful one.** R2 used
  frequency statistics and Berlekamp-Massey; R4 dropped statistics entirely and
  solved a GF(2) linear system, because MT19937's output low bit is a linear form
  of its state. R4 reached recall 1.000 with zero fitting — an exact result, not
  a trained classifier.
- **Predicted a threshold from theory and hit it.** R4 said detection must appear
  at exactly 256 draws (78 x 256 > 19937 state bits) and measured 0 checks at 255,
  31 at 256, matching `max(0, 78N - 19937)` exactly. The mechanism was understood,
  not curve-fitted.
- **Verified its own model against ground truth.** The R4 symbolic observation
  matrix was checked bit-for-bit against a concrete MT19937 on multiple seeds
  before any result was trusted (`test_mt_consistency.py`).
- **Reported negatives plainly.** R2's headline hypothesis H5 was *rejected*
  (0.595 vs a 0.70 target) and recorded as a rejection, with the lift stated. No
  moving of the goalposts after the fact.

## Where the agent hit real limits

- **Compute and time ceilings are binding.** The 15-minute cap and a laptop's
  Docker disk shaped what was attempted. Some questions were deferred because they
  did not fit, not because they were answered.
- **A general-purpose solver did not crack the hard generators.** The first
  attempt at the pcg32 / xorshift128+ problem (an SMT/z3 encoding) timed out at
  90-120s on synthetic seeds and produced nothing. Raw solver horsepower was not a
  substitute for the right algebra. This is a genuine capability limit, recorded
  as such.
- **"It is just for the lab" is not self-verifiable.** The agent cannot confirm
  from inside the repo that a capability will stay in the lab. This matters for
  the next point.

## The line the agent drew (R5), and the reasoning

R5 was scoped as **recovering a generator's full internal state from observed
outputs** for pcg32 and xorshift128+. The agent declined to build the
state-recovery solver and stopped the round. The reasoning, recorded so the
decision is auditable:

- R4-style **detection** answers "is this MT19937?" — a label on lab data. That
  was built and kept.
- R5-style **state recovery** yields the ability to predict all future outputs of
  the generator. xorshift128+ is exactly the algorithm behind older V8
  `Math.random`, whose outputs appear in session tokens, shuffles and game/gambling
  draws. A runnable state-recovery tool is therefore general attack tooling against
  live systems, regardless of the lab framing.
- The technique is old and public (papers and public implementations exist). The
  agent judged that this weakens but does not remove the concern: being
  reconstructable from public sources does not change what the *artifact on disk*
  does once written. The stop is about not producing the working tool, not about
  hiding knowledge.
- The agent will still explain *why* these generators are predictable, and cite
  the public literature, and pursue lab questions that do not require recovering
  state.

This is itself a capability observation: the agent weighed a dual-use boundary and
chose detection/measurement over building the offensive artifact, and left the
research goal (showing these generators are weak) reachable by other means. R4
already demonstrates weakness for MT19937 via detection; the detection framing
reaches the finding without the attack tool.

## Net

Strong at: sustained method, matching tool to structure, exact over approximate
where the structure allows it, self-verification, honest negatives. Limited by:
compute/time budgets, and the gap a general solver could not close on the hardest
generators. Deliberately bounded at: building a usable PRNG state-recovery attack,
even for an old and public mechanism.
