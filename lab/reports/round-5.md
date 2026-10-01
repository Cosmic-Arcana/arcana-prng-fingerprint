# Round 5 — not run: scope and the boundary

Status: **deliberately not implemented.** This file explains what R5 would have
been, what was and was not done, and why the round was stopped. It contains no
solver, no attack recipe, and no reconstruction-enabling detail. See
`lab/CAPABILITIES.md` for the standing boundary this round respects.

## What R4 settled, and what it left open

R4 (`lab/reports/round-4.md`) solved MT19937 as a **detection** problem: given a
run of draws, decide whether it is consistent with *some* MT19937 state. That
works because MT19937's output is an XOR-linear function of its state, so the
question reduces to whether a linear system has a solution — a yes/no label,
answered exactly.

Two generators stayed at chance through every statistical round: **PCG32** and
**xorshift128+**. Their output is not an XOR-linear function of their state
(PCG32 applies a state-dependent rotation; xorshift128+ ends in an addition that
carries), so the R4 approach does not reach them. The round-2 note floated a
follow-up aimed at these two.

## Why R5 was stopped

The only follow-up that could reach PCG32 / xorshift128+ is a different *kind* of
task from R4. R4 labels a sample. The follow-up would **reconstruct a generator's
internal state from its observed output** — and a recovered state is, by
definition, the ability to reproduce the generator's entire future (and past)
output stream.

That capability is dual-use in a way detection is not:

- xorshift128+ is the algorithm behind older V8 `Math.random`. Output from such
  generators appears in session identifiers, shuffles, and game / gambling draws
  in real systems.
- A working state-reconstruction tool is therefore offensive tooling against
  those systems, regardless of the lab framing around it.

The technique is old and documented in public research, which is why R5 was
considered at all. But being reconstructable from public sources does not change
what a runnable tool *does* once it exists. The lab's standing rule
(`lab/CAPABILITIES.md`) is to build detection and measurement, not a usable
state-reconstruction attack. R5 fell on the wrong side of that line, so it was
not built.

## What was actually done, and then discarded

A short exploratory probe was written against **synthetic seeds only** (lab data
the lab itself generated, with the answer already known) to gauge whether a
general-purpose constraint solver could even approach the problem within the
lab's 15-minute compute cap.

- Outcome: it could not. The attempts did not converge within the cap and
  produced no usable result.
- The probe lived only in a scratch directory, was never committed to any repo,
  and is not published. No part of it is in this repository.

This is itself a finding worth recording: a general solver pointed at the problem
was **not** a shortcut. The hard generators resist the off-the-shelf approach,
and closing the gap would require dedicated effort specifically aimed at building
the attack — which is exactly what the boundary says not to do.

## What remains legitimately open (detection / measurement only)

The lab can still advance on these two generators **without** reconstructing
state, for example:

- Measuring whether any *detection* signal (a yes/no "is this PCG32?" style test)
  exists at all for them, and if not, demonstrating that absence — a negative
  result is still a result.
- Characterising how dataset parameters (draw count, orientation, shuffle
  budget) change detectability, as R3 and R4 did for the linear generators.

None of these require recovering state or predicting output, and all of them stay
inside the boundary.

## Net

R5 as originally sketched — recover PCG32 / xorshift128+ state — is intentionally
left unbuilt. The one probe that was attempted stayed on synthetic data, did not
work, and was discarded. The research goal that motivated the lab (showing these
generators are weak, and characterising that weakness) is pursued through
detection and measurement, which R0–R4 and R3 already demonstrate for the
generators where it is achievable.
