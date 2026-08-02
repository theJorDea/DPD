# Implementation notes

Дата среза: 2026-08-03.

## 1. Repository layout and dependency policy

Project code is NumPy-first and deterministic. Required runtime is pinned in
`requirements-baseline.txt` (`numpy==2.5.1`); SciPy/pandas/scikit-learn and
matplotlib are optional audit/plot dependencies.

Upstream sources are read-only submodules:

```text
vendor/OpenDPD
vendor/DPD_for_PA
vendor/chaotic_library
```

Local implementation does not patch upstream code. Audit findings live under
`research/`; reproducible models/evaluators live in `baseline/`; runnable
workflows/configs/results are under `experiments/`; invariants are in `tests/`.

## 2. Core APIs

### 2.1 Metrics, alignment and protocol

- `baseline/alignment.py`
  - integer cross-correlation delay;
  - fractional parabolic diagnostic;
  - complex gain utilities.
- `baseline/fractional_alignment.py`
  - frame-safe windowed-sinc shift;
  - no circular convolution;
  - deterministic coefficient digest/support mask.
- `baseline/metrics.py`
  - pooled complex NMSE;
  - OpenDPD-compatible segment metric;
  - sample RMS EVM, PSD, ACLR and PAPR helpers.
- `baseline/pa_benchmark.py`
  - `PAEvaluationProtocol`;
  - `freeze_pa_evaluation_protocol(...)` from train only;
  - `prepare_pa_split(...)`;
  - `evaluate_pa_predictor(...)` for a frozen forward model.

Evaluator direction is hard-coded conceptually as
`x_split -> model -> y_hat -> measured y`. Post-prediction gain/delay fitting
is not part of the API.

### 2.2 Memory Polynomial

`baseline/pa_models.py` provides:

- segmented causal design matrices;
- `MemoryPolynomialPA.predict(...)`;
- `predict_segments(...)` with reset per frame;
- NPZ `save/load`;
- column-scaled complex ridge fit and diagnostics.

Historical MP selected models remain valid. Their old JSON operation records
have stale `state_real_values=0`; current counter correctly adds 46 DPA and 58
APA real delay-state values. NMSE/coefficients are unaffected.

### 2.3 Generalized Memory Polynomial

`baseline/gmp_pa.py` provides:

```python
GMPConfig(...)
gmp_terms(config)
gmp_segmented_design_matrix(...)
fit_gmp_pa(...)
GeneralizedMemoryPolynomialPA.predict(...)
GeneralizedMemoryPolynomialPA.predict_segments(...)
GeneralizedMemoryPolynomialPA.predict_streaming_chunk(...)
GeneralizedMemoryPolynomialPA.save(...) / load(...)
```

Supported branch families:

- aligned;
- lagging envelope;
- causal-leading envelope (`zero lookahead`);
- explicit `opendpd_exact` lookahead mode, not eligible for the current causal
  frontier unless latency/cooldown is declared.

Fit modes:

- `ridge_lstsq`: augmented, column-scaled complex least squares;
- `truncated_svd`: rank-controlled solve with explicit `svd_rcond`.

Normal equations are avoided. Diagnostics retain rank, singular values,
condition, coefficient norm, boundary requirements and train NMSE.

### 2.4 Complexity

`baseline/complexity.py` exposes `OperationCount` and counters for:

- memoryless complex spline;
- spline-memory branch lower bounds;
- MP;
- factorized GMP;
- widely-linear residual correction;
- proper-complex FIR residual correction with incremental enclosing-state
  accounting;
- EnhancedESN_FAN scalar and I/Q pair.

Fields are not collapsed into FLOPs:

```text
real_multiplications, real_additions, real_divisions
nonlinear_operations, comparisons, lookups
real_memory_reads, real_memory_writes
stored_real_coefficients, stored_real_constants, state_real_values
```

Operation counting uses `4M+2A` per complex multiply. Historical PA selectors
used a strict `<1000 real MUL/sample` search bound, but this is not a Huawei
PA-model gate. Deployment DPD passes only by measured target time
`T_DPD/sample <= T_reference(1000 real MUL)`, including all operations and
memory effects.

### 2.5 Residual analysis

`baseline/residual_analysis.py` implements boundary-safe:

- lagged complex correlations;
- radial/tangential error;
- |x|, |x|² and nonlinear envelope diagnostics;
- slow one-pole envelope probes;
- amplitude-quantile/compression regions;
- segment-position/reset diagnostics;
- AM/AM, AM/PM and residual PSD.

`experiments/analyze_pa_residuals.py` is a generic MP/GMP workflow:

- schema-2 portable config;
- hash-bound selection/model/source/data;
- coefficient leave-one-frame-out fit;
- full-train frozen-model reproduction check;
- train OOF + validation only, no test;
- atomic bundle publication and single-writer lock;
- complete artifact hashes and evidence scope.

### 2.6 Causal widely-linear residual diagnostic

`baseline/widely_linear_pa.py` implements the deliberately narrow correction

```text
delta_y[n] = sum_d b[d] * conj(x[n-d])
```

with non-negative unique delays, segmented zero-state inference, explicit
continuous-streaming state, complex ridge fit, diagnostics and NPZ save/load.
It is intentionally not phase-equivariant and therefore is an IQ/measurement
asymmetry diagnostic, not a generic PA inductive bias or proof of PA physics.
`widely_linear_residual_correction_cost(...)` counts each complex tap and can
separate reused versus standalone delay-state storage.

### 2.7 Causal proper-complex FIR residual diagnostic

`baseline/complex_fir_pa.py` implements

```text
delta_y[n] = sum_d b[d] * x[n-d]
```

with the same deterministic complex ridge, segmented reset, continuous state,
NPZ and diagnostics contract. Unlike the conjugate branch it is naturally
phase-equivariant. `complex_fir_residual_correction_cost(...)` counts either
standalone delay state or only the incremental state beyond an enclosing
model's maximum raw-input delay. For the selected APA GMP that boundary is
29 samples; extending to lag 49 adds 40 real state values, not zero and not a
second full 98-real buffer.

### 2.8 Standalone spline-Hammerstein PA (SPH)

`baseline/spline_hammerstein_pa.py` provides the phase-equivariant forward PA
family used in the APA selection:

```text
v[n] = x[n] * C(|x[n]|)
y_hat[n] = v[n] + sum(l=1..L-1) h[l] * v[n-l], h[0] = 1+0j
```

The implementation exposes `SplineHammersteinPA`,
`SplineHammersteinState`, `fit_spline_hammerstein_pa`, local-support knot
design, deterministic complex ALS, `predict_segments`, continuous
`predict_chunk`, NPZ `save/load` and an exact `operation_count()`.
`experiments/select_pa_sph.py` owns recipe enumeration/ranking and
`experiments/run_pa_sph.py` owns hash verification, staged train OOF, frozen
full-train refit, descriptive validation and atomic publication.

The selected APA model is `K=32,L=8`, 37 MUL/36 ADD, one amplitude `sqrt`,
5 comparisons, 4 LUT accesses, 78 stored real coefficients, 63 constants and
14 persistent state reals. Streaming/reset equivalence is exact in complex128.
The model is deliberately retained as a negative low-cost control point:
train-OOF NMSE is −30.402374 dB, so it is not promoted to the evaluator and
does not unlock DPD.

## 3. Experiment runners

### 3.1 Model selection

- `experiments/select_pa_mp.py` — staged MP architecture/ridge selection.
- `experiments/select_pa_gmp.py` — strict-budget GMP enumeration, architecture
  selection and solver/ridge refinement.

GMP config contract freezes:

- dataset and A0 protocol;
- architecture grid/topology names;
- exclusive MUL ceiling;
- primary full-record validation metric;
- secondary common-interior metric;
- architecture-stage solver and regularization refinement.

Output:

```text
selected_gmp_pa.npz
selected_validation_evaluation.json
validation_trials.json
selection_manifest.json
execution_record.json
```

### 3.2 Alignment sensitivity

`experiments/evaluate_fractional_alignment_sensitivity.py` compares equal-
support A0/A1 with fixed MP/GMP recipes using train OOF + validation. It cannot
read test. `experiments/results/pa_alignment_protocol_decision.json` separately
freezes the chosen protocol; runner recommendation alone cannot mutate formal
configs.

### 3.3 Residual release gate

`experiments/decide_gmp_test_release.py` consumes a preregistered residual
config and verifies before test:

- selection/config/model/source/data/artifact hashes;
- matched MP/GMP OOF and validation metrics;
- rank/condition/coefficient norm/support thresholds;
- boundary gaps and diagnostic completeness;
- exact operation budget;
- full/reset/streaming equivalence.

It never reads or hashes test files. A PASS grants one workflow-specific
frozen test invocation; it is intentionally not Gate A→B.

### 3.4 Frozen test

`experiments/evaluate_frozen_pa.py`:

1. validates manifest task/model/split contract;
2. verifies config, model, source and train/validation hashes;
3. validates topology and operation count against loaded NPZ;
4. validates frozen boundary policy;
5. only then opens test CSV;
6. writes evaluation, prediction and manifest without refit.

Public preflight:

```python
verify_selection_before_test_access(selection_manifest)
```

One-shot outputs record `refit_performed=false`,
`post_prediction_gain_fit=false` and `post_prediction_delay_fit=false`.

### 3.5 Linear residual audits

`experiments/audit_widely_linear_pa.py` refits the frozen GMP recipe in each
leave-one-frame-out fold, then fits only the selected residual feature mode to
that fold's fit residual. It dispatches explicitly between `conjugate` and
`proper`; `experiments/audit_complex_fir_pa.py` is the proper-FIR CLI.
Candidate selection uses OOF full/common gains; reused validation is
descriptive and test access is absent. The runner checks source/data hashes,
exact cost, full rank, frame reset and arbitrary-chunk streaming equivalence
before publishing an atomic result bundle.

### 3.6 SPH staged selection runner

`experiments/run_pa_sph.py` is the complete forward-identification workflow for
the standalone spline-Hammerstein family. It enforces this order:

1. verify preregistered config, source, evidence and train/validation hashes;
2. load train frames and run deterministic staged OOF selection;
3. freeze the recipe and full-train parameter hashes;
4. load validation only for reused descriptive metrics/residuals;
5. verify model/cost/streaming invariants and atomically rename the bundle.

The runner rejects rank-deficient control/FIR designs, non-finite values,
objective increases and support violations. The APA publication
`experiments/results/pa_sph_apa200_selection/` records 60 unique recipes,
180 completed OOF fits, 4 cache hits, exact reset/chunk equivalence and
`test_split_accessed=false`. The selected result is intentionally not wired
into the DPD evaluator because its OOF quality gate failed.

### 3.7 Non-factorized sparse spline-memory PA runner

The sparse PA implementation is split into three independently reviewable
layers:

- `baseline/sparse_spline_memory_pa.py`: forward phase-equivariant model,
  segmented complex ridge fit, support/condition diagnostics and exact cost;
- `experiments/select_pa_sparse_spline_memory.py`: canonical recipe hashes,
  train-only OOF folds, S0/S1/S2 staged search, cache and gate ranking;
- `experiments/sparse_pa_benchmark_support.py` plus
  `experiments/run_pa_sparse_spline_memory.py`: frozen evidence checks,
  frame-aware metrics, streaming checks and atomic bundle publication.

The production command was:

```bash
.venv/bin/python -m experiments.run_pa_sparse_spline_memory \
  --config experiments/configs/pa_sparse_spline_memory_apa200.json
```

The runner verified hashes before train load, selected only on three explicit
train frames, froze recipe and full-train coefficients, and loaded validation
afterward. It contains no test loader. The selected recipe has six branches
and `K=12`; OOF NMSE is `−32.030011 dB` full / `−32.088250 dB` common, cost is
`54 MUL / 58 ADD`, and the result is rejected versus both MP and GMP. The
immutable output is
`experiments/results/pa_sparse_spline_memory_apa200_selection/` (14 unique
recipes, 42 OOF fits, 33.5888 s before publication). Residual analysis pointed
to lag 9, so the follow-up family was separately preregistered around that lag.

The generic selector now supports an optional
`reference_models.incremental_control_oof` contract. When present,
`_annotate_research_gates(...)` compares pooled and per-fold NMSE against the
frozen parent and records explicit minimum-gain checks; the production runner
serializes those fields without changing legacy config behavior. The lag-9
config requires `+0.25 dB` pooled full/common and `+0.10 dB` in every fold,
in addition to the existing MP/GMP gates.

The lag-9 production command was:

```bash
.venv/bin/python -m experiments.run_pa_sparse_spline_memory \
  --config experiments/configs/pa_sparse_spline_memory_lag9_apa200.json
```

It selected
`parent_plus_signal_lag8_10_current_envelope_K12_r1e-08_b0:0,1:1,2:2,22:22,23:23,24:24,8:0,9:0,10:0`.
Train OOF full/common NMSE was `−37.792478/−37.852832 dB`, reused validation
was `−37.860728/−37.898605 dB`, and the exact schedule was `72 MUL/82 ADD`
with 216 coefficient reals and 48 state reals. Incremental and cheap-Pareto
gates passed; evaluator and Gate A→B remained closed. The immutable bundle is
`experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/`, with
`62.5693 s` runtime and test access false.

### 3.8 APA capture-transfer runner and release

The transfer implementation is deliberately separate from DPD:

- `experiments/transfer_pa_apa200_to_b.py` loads only source/target train and
  validation, verifies byte-identical excitation inputs, freezes source
  topology, and fits coefficients only on chronological target-train
  prefixes;
- `experiments/release_pa_transfer_apa200_to_b.py` requires explicit
  `--release-test`, verifies the pre-test manifest, selected `N`, coefficient
  hashes and source hashes before opening target test, then publishes an
  immutable bundle;
- `experiments/verify_pa_transfer_bundle.py` replays the 20 pre-test metric
  records without opening test;
- `experiments/verify_pa_transfer_release.py` replays all 4 held-out records,
  checks prediction/artifact/data hashes and verifies that test was not used for
  selection, coefficient fit, delay, gain or bin fitting.

The frozen commands are:

```bash
.venv/bin/python -m experiments.transfer_pa_apa200_to_b \
  --config experiments/configs/pa_transfer_apa200_to_b.json
.venv/bin/python -m experiments.verify_pa_transfer_bundle \
  --bundle experiments/results/pa_transfer_apa200_to_b_pretest
.venv/bin/python -m experiments.release_pa_transfer_apa200_to_b \
  --config experiments/configs/pa_transfer_apa200_to_b_release.json \
  --release-test
.venv/bin/python -m experiments.verify_pa_transfer_release \
  --bundle experiments/results/pa_transfer_apa200_to_b_test_release \
  --output experiments/results/pa_transfer_apa200_to_b_test_release_verification.json
```

The pre-test manifest SHA is
`570c3f98af77961f23d30eaa71f38f35c80745a523656042a2dfee1d7e8ddd00`; the
held-out release manifest SHA is
`067a00e66032ae3b0dfde35437a3116ea45931f65ef5bf833aca3ebafe635d07`.
The source-code hash map is embedded in the release manifest, including
`experiments/transfer_pa_apa200_to_b.py =
0b8c4f7ea43d43924516b48db086d639e8beb502741a3d156daa2b9f623f7c82`.

The first held-out attempt exposed a guard bug: the code required three complete
`19662` target-train frames, while the frozen framing is
`19662,19662,19656`. It loaded the target pair but stopped before model
inference and before any metric. The incident is immutable at
`experiments/results/pa_transfer_apa200_to_b_release_incident_001.json`
(SHA `d03217f7ec74f49fbcd3f8619c528d7737b907e281d609b90363339ecacb2a34`).
Only the frame-length guard was corrected; a retry used unchanged models,
coefficients, selected `N` and metric protocol. The final bundle therefore
records access count `2` and `strict_single_open_execution=false`; this is
reported, not concealed.

## 4. Artifact integrity and publication rules

Canonical numerical outputs are immutable:

- no `--overwrite` in current selection/residual/release/test commands;
- existing output directory/report is an error;
- temporary files are atomically replaced only after complete write;
- lock ownership token prevents concurrent writers;
- SHA-256 covers configs, model, source, input files and result payloads;
- execution record stores command, source commit, wall time, environment,
  accessed splits and evidence scope.

Historical MP/test workflows already used dataset test, so current GMP test is
called workflow-specific sealed, not globally pristine.

Absolute checkout paths remain in some historical and frozen test JSON. New
schema-2 configs resolve repository-relative paths; portability should be
improved in future manifests without rewriting existing hash-bound evidence.

The long-FIR config was intentionally committed before its new counter. Its
result manifest therefore records a preregistered/current source-hash mismatch
for `baseline/complexity.py`; numerical discovery artifacts and unchanged GMP/
residual sources match exactly. This is preserved provenance, not repaired by
rewriting the preregistration.

The SPH and sparse PA bundles were published after all source/config/data hashes and model
parameter hashes were verified. Their payload hashes reverify exactly, the
completion manifest was written last inside a temporary directory, and the
directory was atomically renamed. Validation was loaded only after recipe and
full-train model freeze; no test file was opened, hashed or named.

The APA transfer release has a distinct disclosed access audit: its first
process loaded the target test pair before a frame-length guard failed, but
did not infer or compute a metric. The corrected retry used the unchanged
frozen protocol. Consequently it is not described as a pristine single-open
run; `release_access_count=2` and the incident hash are part of the manifest.

### 3.9 Fixed-point PA arithmetic runner

The fixed-point implementation is deliberately separate from DPD code:

- `baseline/fixed_point_pa.py` defines signed formats, nearest-even shifts,
  integer square root, checked products, saturation counters and the vectorized
  causal GMP evaluator;
- `baseline/fixed_point_sparse_spline_pa.py` adds integer knot addressing,
  two-point complex interpolation, delayed branches and explicit sparse
  operation counting;
- `experiments/evaluate_fixed_point_pa.py` verifies hashes, opens only train
  and validation, freezes scales before validation and writes a machine-readable
  report;
- `experiments/configs/pa_fixed_point_apa200.json` preregisters 16/14/12-bit
  widths, 48-bit power/address codes, 56-bit accumulators, rounding and
  saturation policy;
- `tests/test_fixed_point_pa_contract.py`,
  `tests/test_fixed_point_gmp_pa.py`,
  `tests/test_fixed_point_sparse_spline_pa.py` and
  `tests/test_evaluate_fixed_point_pa.py` cover arithmetic boundaries,
  streaming/reset equivalence, knot collisions, overflow reporting and
  train/validation-only split access.

The APA report is
`experiments/results/pa_fixed_point_apa200/fixed_point_report.json`; the DPA GMP
report is
`experiments/results/pa_fixed_point_dpa200/fixed_point_report.json`.  Both
report zero saturation in all evaluated rows.  APA 16-bit GMP/sparse
fixed-vs-float validation degradation is below −64 dB, APA 12-bit GMP
degrades to about −36.4 dB, and DPA 12-bit GMP remains about −52.5 dB.  The
sparse schedule reports six integer divisions explicitly; reciprocal-multiply
hardware must account for them. Host timings are not target latency. The separate
fixed-point spline-memory DPD cascade is implemented as described below; it does
not convert PA-evaluator arithmetic into physical-PA evidence.

### 3.10 Fixed-point spline-memory DPD and one-command surrogate demo

`baseline/fixed_point_spline_memory_dpd.py` implements the selected
three-branch phase-equivariant DPD as deterministic signed-integer arithmetic.
`experiments/evaluate_fixed_point_dpd.py` freezes formats from train-only desired
input, floating DPD train drive and coefficients, then evaluates 16/14/12-bit
validation without adapting scales, topology or precision. It writes float/fixed
waveforms and a completion manifest; the frozen PA surrogate remains floating so
the report isolates DPD arithmetic degradation.

`experiments/run_surrogate_demo.py` is the presentation/reproduction wrapper
added in `a385635` and hardened in `b982266`. Run it with a new output path:

```bash
DPD_DEMO_OUTPUT=/tmp/dpd-surrogate-demo-manual
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.run_surrogate_demo --output-root "$DPD_DEMO_OUTPUT"
```

The wrapper performs no fit or model selection and opens no measured output or
test split. Its only evaluation direction is
`desired validation x -> frozen DPD -> frozen legacy PA surrogate`. It verifies
the frozen config/source/artifact hashes, metric tolerances, operation contracts,
zero fixed-point saturation/knot collisions, exact configured chunked-streaming
equivalence and bit-exact 90-degree rotation on the evaluated signals. It
requires exactly 12 child completion manifests, publishes `summary.json`
atomically and publishes
`completion_manifest.json` last. A missing final manifest marks an incomplete
run; an existing output root is never overwritten.

Compact numerical regression contract:

| Dataset | Float NMSE no DPD -> DPD | Float configured adjacent relative L/R | 12-bit cascade NMSE | 12-bit drive vs float NMSE |
|---|---:|---:|---:|---:|
| DPA_200MHz validation | -20.3381 -> -30.5324 dB | +4.7494 / +7.7372 dB | -30.5148 dB | -54.8714 dB |
| APA_200MHz validation | -19.9688 -> -32.3800 dB | +16.4797 / +13.8639 dB | -32.3790 dB | -53.5984 dB |

The float operation vector is `21 MUL, 24 ADD, 0 DIV, 1 magnitude nonlinear,
6 LUT, 18 reads, 2 writes, 4 state reals` per complex sample, with `5/3`
comparisons and `144/48` stored coefficient reals for DPA/APA. The integer
reference vector is `20 MUL, 25 ADD, 1 DIV, 1 integer sqrt, 8 LUT, 28 reads,
2 writes, 4 state reals`, again with `5/3` comparisons. These are analytical
software schedules, not measured customer-equivalent timing.

Targeted sealing command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest tests.test_run_surrogate_demo -v
```

All **10/10 tests passed** on the current environment. They cover the sealed
end-to-end run, frozen hashes/claims/tolerances, symlink-safe reads, exact child
manifest set, output-directory identity/no-overwrite handling, failure marking
and fixed integrity checks.

The validation data had already participated in historical floating-model
selection. Therefore this is explicitly `validation_replay_surrogate_only`, not
untouched final evidence. It does not prove physical-PA or independent-evaluator
linearization, customer RF harmonic/spur attenuation, OpenDPD superiority,
Huawei acceptance, target latency/resources/power or that 12 bit is sufficient.
No precision was selected; Gate A->B remains closed.

## 5. Tests and invariants

The targeted one-command-demo sealing suite is **10/10 passed** on
the environment in this document, including the real end-to-end demo. The
repository-wide count changes as the independent OpenDPD resume work lands and
is not inferred from this targeted run. The full discovery command below remains
the acceptance command after any code change.
Test modules cover:

- gain/delay and frame-safe fractional alignment;
- spline partition of unity/continuity and complex ridge;
- correct DPD desired-input direction;
- MP/GMP basis, fit, rank truncation, save/load and streaming;
- exact complexity/state counts;
- PA evaluator test guards and no post-hoc fitting;
- deterministic selector and historical PA-search `<1000` exclusion (not a
  deployment-DPD acceptance test);
- residual OOF/test isolation/atomic publication;
- causal widely-linear fit/save/load/streaming, cost and deterministic audit
  selection/fallback;
- phase-equivariant long-FIR fit/save/load/streaming, incremental state cost
  and shared audit dispatch;
- release predicates and streaming checks;
- optional incremental-control gate and bounded lag-9 preregistration contract;
- memoryless and selected spline-memory DPD fixed-point arithmetic,
  16/14/12-bit format freeze, saturation/collision, phase rotation and
  configured chunked-streaming equivalence;
- fixed-point GMP/sparse PA arithmetic, format freeze and no-test runner.
- one-command surrogate-demo hashes, claim scope, metric tolerances, exact child
  manifests, output identity/failure semantics and sealed end-to-end execution.

Run after any code change:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest discover -s tests -v
```

The previous 120-test 0.396 s timing is obsolete.

## 6. Key implementation commits

| Commit | Task |
|---|---|
| `a4a3312` | causal factorized GMP model and tests |
| `f73314e` | rank-controlled GMP calibration |
| `0647d95` | validation-only GMP selector |
| `fc85e44` / `79089f8` | frame-safe fractional transform and gated runner |
| `754a069` | preregister alignment sensitivity configs |
| `443a0f6` / `895a08e` | DPA/APA sensitivity results |
| `2ab7e0f` | freeze A0 protocol |
| `575b746` / `e6cff9d` | DPA/APA formal GMP selections |
| `79822f8` | generic hash-bound MP/GMP residual workflow |
| `07d98f6` | preregister residual release thresholds |
| `6f60bcb` / `622f88c` | DPA/APA residual results |
| `434cbbd` / `51ae3a8` | release-gate code and decisions |
| `8ae235d` / `0e56add` | one-shot frozen GMP tests |
| `ab8a374` | preregister APA widely-linear residual audit |
| `f8f75de` / `340c923` | exact correction cost and causal model/tests |
| `9400e3d` | hash-bound widely-linear audit runner/tests |
| `5d2e273` | APA widely-linear negative result bundle |
| `37beab9` | preregister APA proper long-FIR residual audit |
| `6529522` / `dae3ae6` | exact FIR cost and causal model/tests |
| `d9c897e` | shared hash-bound proper-FIR audit runner/tests |
| `31bf9a6` | APA proper long-FIR negative result bundle |
| `4602591` | preregister APA standalone SPH search |
| `b89de01` / `bf417e4` / `f148def` | SPH inference, local-support designs and deterministic ALS fit |
| `b72b69b` / `5a27c54` / `8790330` | SPH recipe validation, OOF evaluator and staged orchestration |
| `64b19e5` / `1b6ebde` / `252ff2a` | SPH input integrity, atomic runner/publication and progress record |
| `516afaa` | immutable APA SPH result bundle |
| `1452437` / `6be677c` / `49a2714` | preregistered sparse PA, forward model and support gate |
| `f0d77c5` | staged sparse PA OOF search |
| `841a381` / `4be0921` | sparse PA integrity support and production runner |
| `5b804f3` | immutable APA sparse PA result bundle |
| `5d21002` | incremental parent-control gate and fold-wise gain checks |
| `7275ffd` | explicit frozen-implementation preregistration status |
| `db41e16` | preregistered bounded APA lag-9 config |
| `c023e7c` | lag-9 config contract tests |
| `aa9bd38` | immutable APA lag-9 sparse PA result bundle |
| `790f744`–`503c48a` | APA transfer preregistration, guarded release, incident and verification |
| `27e15a3`–`48be898` | synchronized PA/benchmark/robustness/gap/roadmap reports |
| `161837e` | exact APA transfer reproduction protocol and hashes |
| `bd59d05` / `4aa5480` / `a2fa84b` | fixed-point arithmetic contract, GMP reference and vectorized kernel |
| `3beddb5` | bit-accurate sparse spline-memory PA and tests |
| `869312c` | sealed train/validation-only fixed-point PA evaluator |
| `63a3dcf` | immutable APA fixed-point PA result report |
| `c109eaf` | synchronized roadmap, hardware, PA benchmark and experiment plan |
| `9561ea0` | immutable DPA fixed-point GMP config and result report |
| `904e179`–`15d6482` | supervisor requirement clarification and synchronized DPD-only timing/spectral contracts |
| `da993cb` | hash-bound target-calibrated coefficient override support and tests |
| `619f680` | preregistered target PA fixed-point config |
| `62ea00c` | immutable target PA fixed-point result report |
| `706c0e5` | target quantization benchmark documentation |
| `91aedae` / `16b736a` / `d649aa1` | sealed spline-memory DPD integer runner and immutable DPA/APA validation bundles |
| `c88b394` / `d7b72d5` | immutable DPA/APA float plus 16/14/12-bit spectral replay bundles |
| `a385635` / `b982266` | one-command surrogate demo and hardened artifact sealing |

Each numerical dataset task was committed and pushed separately from code and
documentation.

## 7. Public result locations

```text
experiments/results/pa_gmp_dpa200_selection/
experiments/results/pa_gmp_apa200_selection/
experiments/results/pa_gmp_dpa200_residuals/
experiments/results/pa_gmp_apa200_residuals/
experiments/results/pa_gmp_dpa200_test/
experiments/results/pa_gmp_apa200_test/
experiments/results/pa_widely_linear_residual_apa200/
experiments/results/pa_long_fir_residual_apa200/
experiments/results/pa_sph_apa200_selection/
experiments/results/pa_sparse_spline_memory_apa200_selection/
experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/
experiments/results/pa_transfer_apa200_to_b_pretest/
experiments/results/pa_transfer_apa200_to_b_test_release/
experiments/results/pa_transfer_apa200_to_b_test_release_verification.json
experiments/results/pa_fixed_point_apa200/fixed_point_report.json
experiments/results/pa_fixed_point_dpa200/fixed_point_report.json
experiments/results/pa_fixed_point_apa200_b/fixed_point_report.json
experiments/results/dpd_fixed_point_dpa200_validation/
experiments/results/dpd_fixed_point_apa200_validation/
experiments/results/dpd_fixed_point_{dpa200,apa200}_spectrum_{float,16bit,14bit,12bit}_validation/
```

The one-command wrapper writes its sealed bundle to the caller-supplied fresh
output root; no machine-specific demo output path is a normative repository
artifact.

Normative documents:

- `REQUIREMENTS.md`;
- `EXPERIMENT_PLAN.md`;
- `ROADMAP.md`;
- `PA_MODEL_BENCHMARK.md`;
- `DPD_BENCHMARK.md`;
- `ROBUSTNESS_AND_ADAPTATION.md`;
- `HARDWARE_COST.md`;
- `FINAL_GAP_ANALYSIS.md`.

## 8. Known issues

1. Scientific-supervisor clarification freezes DPD-only equivalent-time scope,
   but exact harmonic/spur bands/threshold and 1000-real-MUL target timing
   kernel remain unknown.
2. Feedback frequency response, DC and IQ imbalance are not independently
   calibrated; A0/A1 is only delay sensitivity.
3. DPA has 9 train frames; APA only 3. Frames are not independent captures.
4. Test splits are not globally pristine because historical MP workflows used
   them; new tuning must not use those values.
5. No runnable OpenDPD neural checkpoint/GPU local reproduction.
6. GMP NumPy batch inference is not optimized for target sample rate.
7. Peak RSS was not measured for formal fits.
8. New GMP PSD/AM-AM/AM-PM arrays exist, but canonical rendered plots are not
   generated yet.
9. DPA/APA source and target-calibrated APA-B GMP/sparse payloads and the
   selected DPA/APA spline-memory DPD now have 16/14/12-bit integer reports.
   The DPD result is still a reused-validation replay through a frozen legacy
   surrogate; an independent evaluator/physical-PA integer cascade and target
   timing remain pending. PA cost is not checked against the deployment DPD
   timing gate.
10. No controlled power/temperature captures or physical predistorted output.
11. Gate A→B remains closed; existing DPD is surrogate-only.
12. Checked APA short conjugate supports failed the 0.1 dB OOF threshold;
    `no_correction` remains selected and no physical IQ attribution is made.
13. Checked APA sparse proper-FIR supports also failed 0.1 dB; all folds
    improved, but the best aggregate gain was only 0.018/0.020 dB.
14. APA standalone SPH is implemented and reproducible, but its −30.402 dB
    train-OOF NMSE is 6.652 dB worse than matched MP; it is not an evaluator.
15. The first non-factorized sparse spline-memory PA was rejected:
    `−32.030 dB` OOF and 6.315 dB worse than matched GMP. The separately
    preregistered lag-9 family reaches `−37.792 dB` OOF at 72 MUL and passes
    cheap-Pareto, but remains 0.553 dB behind GMP and is not an evaluator.
16. `APA_200MHz_b` is now an independent capture-transfer check: calibrated
    GMP reaches −37.895 dB and lag-9 sparse −34.801 dB on held-out full
    record, but no controlled operating-point metadata or physical PA cascade
    confirms generalization.

## 9. Extension contract for the next PA model

Do not modify the current evaluator while adding a model. A new causal PA
family must provide:

```text
immutable topology/config dataclass
predict(full vector)
predict_segments(reset per declared frame)
predict_streaming_chunk(state in/out)
save/load with metadata
exact OperationCount
complex deterministic fit + diagnostics
```

Then add, in separate tasks:

1. basis/continuity/fit/serialization/streaming/count tests;
2. preregistered train/validation config with full cost reporting; do not
   apply the deployment DPD latency gate to a PA evaluator;
3. selection artifact without test;
4. coefficient-OOF residual artifact;
5. independent release decision;
6. external/new-capture test, not tuning on already opened DPA/APA test;
7. report update.

The standalone spline/CPWL + short-FIR and first non-factorized sparse family
have been isolated and rejected on APA OOF. The narrow lag-9 dictionary is now
also frozen and completed: it passes cheap-Pareto but not the evaluator gate.
No further local delay expansion is allowed before independent capture
evidence. A state-conditioned model is blocked until independent long-capture
slow-state evidence exists.

## 10. Immediate next implementation order

1. Obtain exact harmonic/spur and 1000-real-MUL timing-reference definitions.
2. Complete full validation-quality training of the preregistered OpenDPD PA
   evaluator; sealed CPU runner/config and bounded runtime preflight are done.
3. Record provenance for `APA_200MHz_b` and obtain controlled physical-PA data.
4. Retain the completed fixed-point spline-memory DPD as a numerical regression
   reference; evaluate `desired x -> DPD -> independent PA evaluator/physical
   PA` only after Gate A→B is reopened.
