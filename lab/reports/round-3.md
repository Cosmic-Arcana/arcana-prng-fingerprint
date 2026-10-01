# Round 3 — `derived` orientation: does reusing shuffle words change detectability?

Status: **hypothesis registered before any run.** Results are appended below only
after the run completes. Nothing in the hypothesis section is edited afterwards.

Every round so far used `independent-bit` orientation: each card's reversed flag
is the low bit of a fresh `next_u32()`, so a draw's 78 orientation bits are 78
consecutive generator words. `derived` orientation models a real implementation
bug — one random value decides both a card's position and its orientation. Here
the flag is **bit 31 of a word the shuffle already consumed**:
`reversed[i] = bit31(trace[i % 77])`, where `trace` is the 77 words Fisher-Yates
drew (one `next_u32()` per swap, i = 77..1).

This round asks, as a **detection/measurement** question (no state recovery):
how does derived orientation change what a detector sees, relative to
independent-bit? It has an algebraic part (does the R4 MT19937 detector transfer?)
and a statistical part (does the round-2 classifier's shuffle/generator accuracy
move?).

## What changes in the observation model

1. **No extra words consumed.** Independent-bit spends 78 words per draw on
   orientation on top of the shuffle's 77 (155 words/draw for Fisher-Yates).
   Derived spends 0 extra: a draw is 77 words total. So N draws span far fewer
   generator words, and the orientation bits are *not* a contiguous low-bit run.
2. **The exposed bit is tempered bit 31, not bit 0.** For MT19937 that is still a
   GF(2)-linear form of state (it reads state-word bits 16, 24, 27, 31), so an
   R4-style consistency detector is still well defined.
3. **One dependent bit per draw.** With 78 cards and 77 trace words,
   `reversed[77] = reversed[0]` exactly (both are `bit31(trace[0])`). So each
   draw yields at most **77 independent** orientation observations, and any model
   that treats all 78 as independent will see one perfectly redundant column.
4. **Position of each observed bit is known.** Fisher-Yates word i (i = 77..1)
   sits at a fixed stream offset, so the observation matrix is still exactly
   constructible for a fixed-budget shuffle.

## Hypotheses

H12 (primary, algebraic). An R4-style MT19937 consistency detector built for
derived orientation reaches **recall 1.000 on mt19937 test samples with 0
non-MT test samples flagged**, once enough independent observations are
collected. Because derived exposes ~77 independent bits per draw (vs 78) and the
same 19937-bit state, the threshold is **N >= ceil(19937 / 77) = 259 draws**,
not 256.

H13 (statistical, measurement). Re-running the round-2 feature pipeline on a
derived-orientation dataset, the headline 6-class weak-generator balanced
accuracy and the 11-class shuffle accuracy each land **within 0.05** of their
independent-bit values (0.595 and 0.765 at N=128), i.e. derived orientation does
not materially help or hurt a statistical fingerprint. Direction, if any, is
reported.

H14 (canary). The ChaCha20 seed-parity control stays inside its chance interval
in both the algebraic and statistical parts. Any excursion is a bug.

## The metric that decides it

Primary: **mt19937 test recall and non-MT flag count** from the derived
detector at N = 260, plus the measured threshold location (the N at which checks
first appear). H12 holds iff recall 1.000, 0 non-MT flagged, and the first
checks appear at N = 259 +/- 1.

Secondary: the round-2 heads re-measured on derived data, reported as deltas
against the independent-bit round-2 numbers.

## Method, fixed in advance

- Data: generate a derived dataset from a committed config in `arcana-rng-lab`.
  For the algebraic part it must carry enough draws (>= 300) on the fixed-budget
  Fisher-Yates shuffle for the generators in the R4 set. For the statistical
  part, the round-2 grid shape (128 draws, the round-2 shuffle list) with
  `orientation: derived`.
- Algebraic detector: the R4 code generalised to read bit 31 (tempering taps for
  bit 31) at the derived positions, with the dependent 78th bit dropped per draw.
  Nothing fitted. Null space by the same Four-Russians elimination.
- Statistical part: the round-2 extraction and `HistGradientBoostingClassifier`
  unchanged, class_weight balanced, max_iter chosen on validation only, test
  scored once.
- Splits disjoint by seed (train 1e6+, val 2e6+, test 3e6+). Determinism: fixed
  random_state, config JSON per run under `lab/runs/`, git SHA recorded.
- No new dependencies beyond numpy / sklearn. Cap: 15 minutes wall clock.

## Scope note

This round stays on the detection/measurement side of the R5 boundary recorded
in `lab/CAPABILITIES.md`: it labels and measures samples, and (for MT19937)
tests state *consistency*. It does not recover generator state for pcg32 or
xorshift128+ and builds no predictor of future output.

## What would falsify the mechanism, not just the metric

- mt19937 checks appearing before N = 259 would mean derived exposes more than
  ~77 independent bits per draw — i.e. the 78th bit is not actually redundant, so
  the word-reuse model is wrong.
- An mt19937 sample failing a check means the bit-31 tempering functional or the
  derived positions are wrong, voiding the round.
- A large swing in the statistical heads (H13 failing badly) would mean
  orientation source, not just the shuffle, carries generator signal — worth its
  own round.

---

# Results
