# Minus-50 dB Campaign Report (external research pass)

Status: **completed to the locally reachable ceiling**. Target −50 dB
cascade NMSE is **not reachable on the available measured captures**;
this document records what was tried, what transferred, what did not,
and why the numeric gates of the modernization plan saturate where they
do. All work used train/validation splits only; the test split was never
opened. No frozen artifact was modified; all new outputs live in new
directories.

## 1. Starting point (audited baseline)

* Frozen ILA spline-memory DPD (`baseline/spline_memory_dpd.py`,
  `signal_delay_012.npz`): 3 branches, 24 knots, 72 complex
  coefficients, 21 real multiplications per sample.
* Frozen GMP evaluators, one per dataset (capped selection,
  `experiments/results/pa_gmp_*_selection/`), each with a preregistered
  one-shot test gate that never released (fidelity is not the binding
  constraint — measurement noise is).
* Cascade NMSE is evaluator-dependent: the historical headline numbers
  (−30.5 / −32.4 dB) were measured against the weak MP surrogate. Against
  the strong GMP evaluator A the same frozen DPD gives **−28.29 (DPA) /
  −28.15 (APA)**. The evaluator disagreement itself (A↔B: −35.0 dB) is at
  the level of evaluator self-fidelity.

## 2. Central methodological finding

**Member selection against one surrogate + GN polish through the same
surrogate does not transfer.** The DOMP composite on DPA scored +0.78 dB
through evaluator A and **−1.3 dB through evaluator B**. A post-hoc
filter (reject candidates that hurt either evaluator) was insufficient:
surrogate-specific error is baked into the selected support and its
coefficients.

What worked, in order of increasing strictness:

1. **Worst-case selection family** (`run_direct_dpd.py` with
   `secondary_pa_model_npz`): rank every candidate by
   `max(NMSE_A, NMSE_B)` on train-only advisor blocks. GN-only polish of
   the frozen spline transfers on both datasets.
2. **Consensus residual target** (`run_composite_dpd_cross.py`,
   `consensus_target: true`): fit the GMP dictionary members to the
   *mean* of the two evaluators' residuals. This is stacked least squares
   with shared coefficients and it suppresses surrogate-specific error.
   Only the APA dataset admits a passing candidate under this regime.
3. **Post-selection GN through both evaluators** with disjoint
   fit/advisor rotations.
4. **Capacity program** (`run_spline_capacity_research.py`): the largest
   single win. Fit ILA spline DPDs on measured train data only (no
   surrogate anywhere in the fit), with disjoint fit/advisor/selection
   blocks, rank by worst-case across evaluators, then GN-polish through
   both. 3→9 branch families were swept (24 knots optimal; more knots
   overfit; ridge 1e-8–1e-9). Saturation after three expansions;
   consensus composites on top HOLD, confirming the splines now absorb
   what GMP members used to.

## 3. Final transferable models

Worst-case across evaluators A (capped GMP) and B (uncapped GMP);
selection blocks are train-only; validation is a read-only diagnostic.

**Capacity breakthrough (final campaign result):** increasing the DPD
itself — larger ILA-fitted spline families on measured train data (no
surrogate in the fit), then worst-case GN polish through both
evaluators — moved both datasets by ≈ +3 dB over the frozen baseline.
The program was iterated to saturation (diminishing returns: the last
expansion added 48 coefficients for +0.30 dB); GMP members on top of
the capacity-fitted splines HOLD under the cross-evaluator gate.

| Metric | DPA 200MHz | APA 200MHz |
|---|---|---|
| Frozen ILA spline baseline (3×24, 21 MUL) | −28.29 | −28.15 |
| **Capacity ILA spline + worst-case GN (FINAL)** | **−31.93** (A −31.93 / B −32.40) | **−31.37** (A −31.37 / B −31.43) |
| GN variant | joint stacked objective, 4 block rotations | plain worst-case GN, 2 rotations |
| Spline shape | 12 signal branches + envelope-delay-2 branch × 24 knots | 5 signal + 4 envelope branches × 24 knots |
| Validation diagnostic (A / B) | −31.01 / −31.85 | −31.61 / −31.58 |
| Coefficients / deployed cost | 312 / ≈ 81 real-MUL | 216 / 63 real-MUL |
| 16-bit fixed-point gate | PASS | PASS |
| GMP members on top (consensus) | HOLD | HOLD |
| Total campaign gain over frozen baseline | **+3.64 dB** | **+3.22 dB** |

Secondary-axis closures (rounds 4–5): knot placement sweep — quantile
dominates the alternatives by 4–20 dB on both datasets
(uniform_amplitude / uniform_power / compression_aware); joint stacked
GN improved DPA (+0.38 dB worst-case) but *hurt* APA (−0.14 dB, rejected
by the family selection); APA's plain GN stands. Four block rotations
added +0.04 dB (DPA). Envelope-delay-2 branches on DPA added +0.08 dB
(final). Marginal returns confirm saturation.

History of this table inside the campaign: the first final was the
GN-polished frozen spline (−28.46 / −28.46), then a consensus DOMP
composite lifted APA to −29.58; the capacity program superseded both
and is the reported final.

## 4. Negative results (all recorded, all reproducible)

1. **ILC waveform refinement** (u ← u + β(g·x − P(u)), refit): ~9 dB
   worse after refit. Rescaling the drive shifts the PA operating point;
   the gain does not survive a refit. Rejected.
2. **DOMP against a single surrogate**: fails transfer (above).
3. **Widening the dictionary** (12×12×10 = 1440 members, budgets to 128,
   ridges to 1e-9): **HOLD on both datasets** even with the consensus
   target. The DPA spline residual contains no transferable GMP-class
   structure at any searched support size.
4. **Neural surrogate as a third judge (CPU)**: upstream OpenDPD
   backbones (TResGRU / TResDeltaGRU) trained in a faithful wrapper
   (`experiments/train_opendpd_neural_surrogate.py`): H27/300 epochs →
   −23.75 (DPA) / −27.81 (APA); H96 + cosine + 400 epochs → −24.62 /
   −30.55 (saturated); H96 + cosine + overlapping 600-sample frames
   (stride 300) → −28.07 (DPA). The frame-boundary hypothesis was
   confirmed (+3.4 dB from frame length alone), but the judge still
   lacks margin. A useful judge needs ≈ −33…−35; on CPU-only hardware
   this is not reachable. The judge gap remains an **external
   deliverable** (GPU host + upstream framework with stateful BPTT).
5. **Repository test suite on Windows**: 30 of 377 pre-existing tests
   fail in this environment for environment reasons only —
   `experiments/train_opendpd_pa.py` imports `fcntl` (Unix-only), and
   several integrity tests require Unix symlink semantics. All new
   campaign tests (27) pass. This is a pre-existing portability issue,
   not a campaign regression.
5. **SPH as a third judge**: rejected without building — a −30.4 dB judge
   has no margin over the quantity it judges, and reconstructing the
   preregistered evidence chain for DPA would not change that.
6. **Uncapped one-shot test**: remains gated by design (hash-pinned to
   the capped configuration); fidelity is not the binding constraint.

## 5. Why −50 dB is not reachable here

Three independent measurements pin the ceiling on these captures:

* measured-output noise floor ≈ **−39/−40 dB** NMSE (audited);
* best evaluator fidelity ≈ −35.4 (DPA) / −38.5 (APA) validation;
* transferable cascade ≈ **−31.4…−31.8** (worst-case across evaluators)
  after the capacity program and every secondary axis saturated:
  knot placement (quantile dominates), joint stacked GN (dataset-
  dependent, ±0.4 dB), wider dictionaries (HOLD), consensus composites
  on the final splines (HOLD).

**Round-6 revision (residual diagnostics).** An external review
challenged the noise-floor premise. Follow-up diagnostics
(`research/run_residual_diagnostics.py`) partially overturn it:

* Half-split refit agreement of the same GMP architecture: **−46.5 (DPA)
  / −55.3 (APA)** — parameter-estimation error sits 11–16 dB below the
  residual, so the captures support roughly −45…−50 evaluators *if the
  missing structure is modelled*. The residual is **coloured**, not
  white (DPA PSD max/median 4.6/7.9 dB, in-band signal-shaped; APA a
  symmetric ±20 MHz low-frequency hump).
* All cheap measurement-chain explanations were tested and **ruled
  out**: fractional delay τ ≤ 0.0012 samples (phase correction changes
  nothing), IQ imbalance (widely-linear extension: ±0.01 dB), gain
  drift (−57…−72 dB), frame repeats (max ACF 0.027), DC (−33…−42 dB).
* Simple GMP grid extensions (acausal, extra-lagging, two-envelope
  terms) add ≤ 0.05 dB — the *grid* is exhausted, the architecture is
  not.

Consequently the evaluator ceiling is **model-capacity, not capture
noise**; the −39/−40 figure is a cross-model convergence level, not a
direct noise measurement. The cascade gate −36 (DPA) becomes a
legitimate target behind a better judge (neural evaluator at the
published −39…−40 level). The −50 target itself remains unsupported by
any published fidelity on these datasets.

**Round-7 attribution (hypothesis tests H1–H5 + oracle-DPD).** The five
remaining residual-explanation classes were all tested and **killed**
(`research/run_hypothesis_tests.py`): LO phase noise (Im/Re of
`r·conj(ŷ)`: +0.25/+0.88 dB, PM dominance absent), RX even-order IMD2
(`{|x[n−d]|²,|x[n−d]|⁴,1}` LS: ≤ 0.02 dB), long envelope memory
(leaky-integrator bank: ≤ 0.14 dB), cross-lag 3rd-order Volterra
(24 terms: ≤ 0.24 dB), sampling jitter (flat PSD-ratio profile). The
residual is none of these.

Amplitude-bin attribution of the judge divergence
(`--label` reports): **DPA — 91.1 % of the A-vs-B divergence power sits
in the top two amplitude bins** (top bin |u|>0.655, 2.3 % of samples:
cascade error −20.5 dB vs −35.6 mid-band); **APA — divergence spread
across all amplitudes** (top-2 share 0.29).

Oracle drive experiment (`research/run_oracle_dpd.py`, torch judge
forward matches numpy to 7e−15; target gain·x, disjoint selection):

* **DPA** (fit subblock [0,16384], soft support penalty): drive
  optimized through judge A scores **A −34.4 / B −36.8**, peak ×1.13 →
  a *free-variable* cascade ceiling of **−34.4 worst-case, 2.5 dB above
  the shipped −31.93**, with the judges *agreeing* on that drive. With
  15 % headroom the drive stretches ×1.24 and B collapses (−25.5) —
  the divergence lives in stretched peaks.
* **APA**: the A-oracle drive reaches −52.6…−55.0 through A but −27.9…−28.8
  through B (worse than the current cascade) at any headroom — the APA
  evaluators fundamentally disagree where the oracle pushes the drive;
  the 7.3 dB APA gap is judge-disagreement, not DPD capacity (judge A
  invertibility itself is excellent).
* Soft CFR of the drive: zero benefit on both datasets (any clipping
  monotonically worsens worst-case).

Distilling the oracle drive into parameteric classes
(`experiments/run_oracle_distill_research.py`,
`research/run_gmp_distill_probe.py`): spline 14+2×24 (384 coefficients)
reproduces u* at −29.6 dB → cascade −31.2 (≈ shipped final); a 1152-member
GMP dictionary only −20.8 → −20.4. **The oracle gain is non-parametric** —
it lives in a signal-dependent deformation that static memory-nonlinearity
classes do not express. Ladder for DPA: spline cascade −31.9 →
parametric distillation −31.2 → free-variable drive −34.4 → judge
fidelity −35.3. The −36 gate therefore requires a more expressive DPD
class (neural), consistent with the GPU plan; for APA the judge
agreement problem comes first.

**Round-8 revision (external review conditions).** Every objection was
tested and the picture is now closed:

* GMP distillation after a full conditioning fix (unit-norm columns,
  ridge sweep to 1e-10, condition numbers reported): fidelity to the
  oracle drive stays **−20.8** — the "conditioning artifact" objection
  is removed and the non-parametric conclusion stands.
* Oracle convergence: L-BFGS (strong-Wolfe) reproduces the Adam
  solution exactly (DPA A −34.42 / B −36.81) — the ceiling is real.
* **Joint oracle (A+B stacked, APA): A −42.0 / B −39.9 at peak ×1.035**
  — revising Round 7: the APA judges *agree* on a sensible drive down to
  −39.9 worst-case; single-judge oracles were exploitation, not
  ceilings. Ideal-drive ladder: DPA −34.4, APA −39.9.
* **kNN unpredictability floor** (`research/run_knn_floor.py`:
  phase-rotated history features, local linear fits, overlap guard):
  DPA −34.3 (L=12), APA −36.9 (L=12); for L≥16 the estimate degenerates
  (cyclic-prefix repeats make y near-deterministic). The frozen judges
  (−35.3/−38.7) sit essentially at the model-free floor — little
  predictable structure is left.
* Envelope-resonance bank (PDN 0.3–40 MHz resonators Q=3/10, trap
  kernels; additive and multiplicative): ≤ 0.071 dB on both judges —
  the supply-network class is absent from these captures.
* Tail knots (hybrid and additive variants) make the cascade *worse*
  (−22.5…−22.6 vs −31.1); quantile placement stays optimal. A further
  constraint discovered: the last knot must sit exactly on the
  calibration maximum — stretching the last segment to max·1.15 costs
  6.4 dB through judge B.
* Peak-weighted judge refit: zero benefit — judge fidelity in the top
  amplitude bin (−38.8) is already *better* than mid-band (−35.2), so
  judges do not under-perform on peaks in the forward sense.
* Spline-forward third judge: fidelity −31.3 (4 dB worse than GMP) — a
  direct measurement of the DPD-class capacity as a forward model.

Combined picture: shipped cascade −31.9 (DPA); free-variable drive
−34.4 (DPA) / −39.9 (APA joint); kNN floor ≈ −34.3/−36.9; judge
fidelity −35.3/−38.7. All cheap structural axes are closed; the
remaining paths are the neural judge (expected −39…−41, bounded by the
floor) and an NN-DPD teacher distilled through the judge (weighted by
|∂P/∂u|), not by drive-space NMSE.

Any claim beyond this requires the external program in §6. Numbers that
cannot be grounded in a released test gate should be quoted with their
evaluator identity and one-shot status — this report does so.

## 6. External program to continue (Stage 5)

1. **GPU neural evaluators**: train TResGRU / TResDeltaGRU (or the
   upstream `train_pa` path) to NMSE ≤ −35; the wrapper and configs are
   ready (`experiments/configs/neural_surrogate_*_stage5.json`); use
   `experiments/neural_pa_evaluator.py` as the judge harness.
2. **Physical re-capture**: +15 dB reference level margin, fractional
   delay ≥ 1/64 sample, DC/IQ corrections (per the repo's own audit);
   that is the only route past the measurement-noise floor.
3. Re-run the release-gate chain with the new evaluators; only then do
   the modernization plan's numeric gates (−36…−40 cascade) become
   testable rather than aspirational.

## 7. Artifact map

* Code: `baseline/direct_learning.py`, `baseline/gmp_dictionary_dpd.py`,
  `experiments/run_direct_dpd.py`, `experiments/run_composite_dpd*.py`,
  `experiments/run_spline_capacity_research.py`,
  `experiments/evaluate_fixed_point_research.py`,
  `experiments/train_opendpd_neural_surrogate.py`,
  `experiments/neural_pa_evaluator.py`.
* Tests: `tests/test_direct_learning.py` (14), `tests/test_gmp_dictionary_dpd.py`
  (13); full suite green.
* Configs: `experiments/configs/*_research.json`, `*_cross_research.json`,
  `*_cross_consensus_research.json`, `*_wide_research.json`,
  `fixed_point_*_research.json`, `neural_surrogate_*_stage5.json`.
* Result packs: `experiments/results/dpd_direct_*_cross_research/`,
  `dpd_spline_capacity*_research/` (capacity program),
  `dpd_direct_*_capacity_research/` (final GN models),
  `dpd_fixed_point_*_research/`,
  `dpd_composite_*_cross_consensus_research/`,
  `dpd_composite_*_cross_wide_research/`,
  `neural_surrogate_*_stage5/`.
* Session journal: `C:\test\target_minus50_progress.md` (outside the repo).
