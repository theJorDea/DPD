# Implementation notes

Дата среза: 2026-07-29.

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
- EnhancedESN_FAN scalar and I/Q pair.

Fields are not collapsed into FLOPs:

```text
real_multiplications, real_additions, real_divisions
nonlinear_operations, comparisons, lookups
real_memory_reads, real_memory_writes
stored_real_coefficients, stored_real_constants, state_real_values
```

Current acceptance convention is `4M+2A` per complex multiply and strict
`real_multiplications < 1000`.

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

### 3.5 Widely-linear residual audit

`experiments/audit_widely_linear_pa.py` refits the frozen GMP recipe in each
leave-one-frame-out fold, then fits only the conjugate correction to that
fold's fit residual. Candidate selection uses OOF full/common gains; reused
validation is descriptive and test access is absent. The runner checks source
and data hashes, exact cost, full rank, frame reset and arbitrary-chunk
streaming equivalence before publishing an atomic result bundle.

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

## 5. Tests and invariants

Last recorded complete code suite after the widely-linear audit runner:
**153/153 passed in 1.253 s** on the environment in this document.
Documentation-only commits after that do not change code.
Test modules cover:

- gain/delay and frame-safe fractional alignment;
- spline partition of unity/continuity and complex ridge;
- correct DPD desired-input direction;
- MP/GMP basis, fit, rank truncation, save/load and streaming;
- exact complexity/state counts;
- PA evaluator test guards and no post-hoc fitting;
- deterministic selector and strict `<1000` exclusion;
- residual OOF/test isolation/atomic publication;
- causal widely-linear fit/save/load/streaming, cost and deterministic audit
  selection/fallback;
- release predicates and streaming checks;
- existing spline fixed-point arithmetic/saturation.

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
```

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

1. Exact Huawei metric and physical multiplier meaning are unknown.
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
9. No 16/14/12-bit selected GMP or spline-memory DPD simulator.
10. No controlled power/temperature captures or physical predistorted output.
11. Gate A→B remains closed; existing DPD is surrogate-only.
12. Checked APA short conjugate supports failed the 0.1 dB OOF threshold;
    `no_correction` remains selected and no physical IQ attribution is made.

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
2. preregistered train/validation config under `<1000 MUL`;
3. selection artifact without test;
4. coefficient-OOF residual artifact;
5. independent release decision;
6. external/new-capture test, not tuning on already opened DPA/APA test;
7. report update.

The next preregistered family should be spline/CPWL memoryless nonlinearity
followed by short complex FIR; sparse complex spline-memory follows only if
that isolated ablation justifies it. A state-conditioned model is blocked
until independent long-capture slow-state evidence exists.

## 10. Immediate next implementation order

1. Ask/record metadata for `APA_200MHz_b` measurement B.
2. Preregister external-capture transfer and target-train nuisance alignment.
3. Implement one low-complexity PA spline/FIR family with tests and analytical
   counter.
4. Select using primary APA train OOF/validation only.
5. Evaluate zero-shot and limited coefficient recalibration on APA measurement
   B; target test once after freeze.
6. Reassess Gate A→B.
7. Only if it passes, resume DPD comparison; otherwise continue PA/physical
   evidence work.
