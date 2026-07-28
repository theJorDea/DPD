# Experiment plan and execution protocol

Дата среза: 2026-07-29.

Цель второго этапа — не продолжать оптимизировать DPD через evaluator
сопоставимой точности, а последовательно решить два разных контура:

```text
Контур A — PA identification:
measured x -> frozen PA model -> y_hat; compare with measured y

Контур B — DPD:
desired x -> frozen DPD -> independently frozen PA evaluator/physical PA
          -> y; compare with g*x
```

Старые spline/ESN результаты первого этапа сохраняются как evidence, но не
переименовываются в physical-PA results. Выполнение идёт малыми Git-задачами:
model/tests, preregistered config, validation selection, frozen test, residual
analysis и report — отдельные commits с push после каждого завершённого шага.

## 1. Текущий статус

| Step | Status на дату среза | Допустимый вывод |
|---|---|---|
| Requirements audit | выполнен | точное определение Huawei `10^-5` и `<1000 multipliers` всё ещё неизвестно |
| A0 integer-only PA protocol | реализован | primary current protocol; fractional correction не применяется |
| A1 fractional-alignment sensitivity | transform, runner и production configs реализованы/tested и зафиксированы; numerical results и protocol decision ещё отсутствуют | только sensitivity, не measurement-path de-embedding |
| Complex MP PA selection | DPA и APA выполнены | measured forward validation result |
| Frozen MP PA test | DPA и APA выполнены | held-out measured forward test result |
| MP residual analysis | train OOF + validation выполнены | выбор следующего inductive bias, test не использован |
| Causal factorized GMP PA | model, tests, selector, configs и frozen-test loader готовы; formal sweeps **не запускались** | пока нет GMP quality result |
| GMP residual analysis | **не реализован** в generic runner и не запускался | current residual CLI поддерживает только MP |
| Sparse spline-memory / CPWL+FIR PA | **не реализованы и не запускались** | следующий PA family только после GMP residual |
| Existing spline-memory DPD | выполнен через старый MP surrogate | surrogate-only; не новый cross-evaluator result |
| OpenDPD neural PA/DPD | bundled numeric evidence доступен; checkpoint binaries отсутствуют | не локальный rerun |
| Physical PA verification | недоступна | никаких over-the-air/bench claims |

Gate A→B сейчас закрыт. Арифметические margins из уже существующих DPD и новых
MP PA чисел являются projections, а не выполненными cascade experiments.

## 2. Frozen provenance и ресурсы

| Item | Frozen/current value |
|---|---|
| Project remote | `git@github.com:theJorDea/DPD.git` |
| OpenDPD source | `7426bbf8a47624b59bd7f045a86641b403023f3c` |
| Egor source | `8e8127cfbea4b2d67cc3d944514b4835e4c7e947` |
| chaotic_library source | `f4ebc3e7c302e83d2eb1c44244f5ecd6e2d884ce` |
| Host CPU | Intel Core i5-12450H, 8 physical / 12 logical cores |
| Host RAM | 15 GiB |
| Host accelerator | no detected NVIDIA device; `nvidia-smi` unavailable |
| Python | 3.14.6 |
| NumPy / SciPy | 2.5.1 / 1.18.0 |
| pandas / scikit-learn | 3.0.5 / 1.8.0 |

Все команды ниже запускаются из repository root и используют `.venv`. Для
каждого result manifest сохраняются command, config/source/data hashes,
environment, split access и timing. `--overwrite` в canonical commands
намеренно отсутствует: в clean checkout output создаётся один раз, а
случайная перезапись frozen evidence должна завершаться ошибкой.

## 3. Datasets, splits и checkpoints

Primary second-stage datasets используются отдельно:

| Dataset | Train / validation / test complex samples | Fs | `nperseg` | Waveform |
|---|---:|---:|---:|---|
| `DPA_200MHz` | 23,040 / 7,680 / 7,680 | 800 MHz | 2,560 | 10×20 MHz LTE, 64-QAM |
| `APA_200MHz` | 58,980 / 19,662 / 19,662 | 983.04 MHz | 19,662 | 5 carriers, LTE TM3.1a, 256-QAM metadata |

Files are the committed split CSVs under
`vendor/OpenDPD/datasets/<dataset>/`. They are used verbatim as 60/20/20;
there is no random re-split and no row-wise normalization. `DPA_160MHz` and
`APA_200MHz_b` are also present, but are out of scope for initial selection.
They represent other dataset/capture conditions and must not be appended to
the primary train/test rows as if samples came from one PA session.

Availability of evaluators/checkpoints:

- frozen MP NPZ, selection, validation, test and residual artifacts exist in
  `experiments/results/pa_mp_{dpa200,apa200}_selection/` and
  `experiments/results/pa_mp_{dpa200,apa200}_residuals/`;
- no `pa_gmp_*_selection/` result directory exists at this snapshot;
- OpenDPD bundled JSON records neural checkpoint paths and hashes, but no
  `.pt`, `.pth`, `.ckpt` or `.onnx` binary is present in the vendored tree;
- MP/GMP OpenDPD controls can be refit from CSV because they are closed-form;
- first-stage spline DPD and its old MP surrogate NPZ files exist, but are not
  independent PA evaluators;
- no physical-PA capture produced from a predistorted waveform is available.

## 4. Split, alignment, state and seed contract

1. Train is the only calibration source.
2. Architecture/regularization selection uses validation.
3. Test is opened only by the separate frozen-test command after model,
   config and selection hashes have been reviewed and frozen.
4. Integer alignment and gain diagnostics are frozen from train. No
   post-prediction gain, phase or delay fit is allowed. The diagnostics remain
   separately named:
   \(g_{LS}=\sum x^*[n]y[n]/\sum|x[n]|^2\) and
   \(g_{peak}=\max|y|/\max|x|\), where `opendpd_peak` is compatibility
   metadata rather than a replacement for complex-LS gain.
5. A0 uses integer delay zero for both primary datasets and no fractional
   transform.
6. A1 may use only an explicitly supplied train-frozen delay with the
   versioned frame-safe FIR, symmetric guard and no circular wrap. A1 remains
   sensitivity-only unless independent feedback/loopback calibration validates
   that delay.
7. State resets at each dataset `nperseg` frame; partial final frames remain
   explicit for pooled metrics. The OpenDPD-compatible metric right-zero-pads
   the final partial model input and reference, runs inference on that padded
   input, and therefore scores any causal memory tail. It records both real and
   padded sample counts. A common warm-up/cooldown is used across candidates.
8. MP, GMP and spline ridge/SVD fits are deterministic: `seed=null`.
9. Egor audit reproduction retains PA seeds `{42,43}` and DPD seeds
   `{100,101}` because those are the notebook settings.
10. The bundled OpenDPD report is seed 0 only. A future stochastic
    apples-to-apples rerun first reproduces seed 0, then uses seeds `{0,1,2}`
    without changing selection rules between seeds.

## 5. Metric and complexity contract

### 5.1 Forward PA identification

For \(e[n]=\hat y[n]-y[n]\), the primary score is

\[
\mathrm{NMSE}_{pool,dB}
=10\log_{10}
\frac{\sum_n|\hat y[n]-y[n]|^2}
     {\sum_n|y[n]|^2}.
\]

Every frozen PA result must also retain:

- full-record and common-interior pooled complex NMSE;
- OpenDPD mean-per-segment-dB NMSE, including a right-zero-padded final partial
  segment exactly as declared by the runner; pooled metrics continue to use
  only real, non-padding samples;
- ordinary MSE, relative error power and time-domain RMS sample EVM;
- residual/error PSD with exact Welch parameters;
- AM/AM gain and AM/PM residuals with train-frozen bins;
- extrapolation beyond maximum training input amplitude;
- fit time and host batch inference timing;
- real MUL/ADD, sqrt/nonlinear, comparisons, LUTs, reads/writes;
- coefficient/constants/state storage and declared numeric precision;
- later: bit-accurate fixed-point degradation and chunk equivalence.

Current PA error PSD uses a periodic Hann window, `nfft=nperseg`, 50% overlap,
constant detrend and density scaling, normalized by integrated measured-output
power. DPA uses `fs=800e6`, `nperseg=2560`; APA uses `fs=983.04e6`,
`nperseg=19662`. OpenDPD spectral-bin “EVM”, strongest-inband-subchannel ACLR
and conventional total-main-band ACLR remain separately labelled definitions.

`error < 10^-5` remains unresolved. If it means normalized error power, the
gate is pooled NMSE below −50 dB. If it means MSE or SSE, scaling and
aggregation must first be supplied by Huawei.

### 5.2 DPD

The only deployment score path is:

```text
desired x_split -> DPD -> frozen independent PA -> compare with g*x_split
```

Measured `y_test` may enter only a separately labelled ILA/postdistorter
diagnostic, never the deployment DPD input. DPD comparison retains pooled and
OpenDPD NMSE, EVM definitions, ACLR L/R/average, PSD, PAPR, peak drive,
support violations, stability, operations, memory, calibration time and
fixed-point degradation.

### 5.3 Operation convention

```text
1 complex multiply = 4 real MUL + 2 real ADD
FMA = 1 real MUL + 1 real ADD
sqrt/nonlinear, compare, lookup and memory traffic are separate columns
```

The project gate is strictly `<1000`, so a candidate with exactly 1000 real
MUL/complex sample is rejected. Parameter count is never substituted for
operation count.

## 6. Exact commands and execution order

### 6.1 Tests

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest discover -s tests -v
```

Measured after the runner/config changes: **120 tests passed in 0.396 s** on
the host above. The timing is a local unit-suite measurement, not an inference
benchmark.

### 6.2 A0/A1 alignment sensitivity — infrastructure ready, not yet run

The current primary protocol is integer-only. Inside the paired sensitivity
runner, A0 is a zero-fractional-shift control passed through the same FIR
implementation and symmetric crop as A1. Its score is therefore an
equal-support control, not a bit-identical replay of the existing uncropped MP
artifact. The reusable A1 transform is in
`baseline/fractional_alignment.py`; the fixed-recipe train-OOF/validation
runner is `experiments/evaluate_fractional_alignment_sensitivity.py`.
Infrastructure tests confirm frame-safe equal support, fixed MP/GMP recipes,
hash guards, immutable single-writer publication and no test access. Runner
commit `79089f8` and config commit `754a069` are frozen; the commands below are
now runnable, but neither numerical result has been produced yet:

```bash
.venv/bin/python -m experiments.evaluate_fractional_alignment_sensitivity \
  --config experiments/configs/pa_alignment_sensitivity_dpa200.json
.venv/bin/python -m experiments.evaluate_fractional_alignment_sensitivity \
  --config experiments/configs/pa_alignment_sensitivity_apa200.json
```

Required production config contract:

- a portable repository-relative `dataset`, such as
  `vendor/OpenDPD/datasets/DPA_200MHz`; execution must not depend on the
  absolute checkout path recorded by an older selection manifest;
- exact hash-bound MP selection manifest;
- one fixed causal GMP recipe embedded inline, not a mutable path indirection;
- `output_dir` and `alignment_filter={tap_count,kaiser_beta}`;
- `max_real_multiplications_per_sample=1000`, interpreted as an exclusive
  ceiling, so every MP/GMP probe must satisfy strictly fewer than 1000 real
  multiplications per complex sample;
- A0 zero fractional shift and A1 delay read from the train-frozen MP
  selection manifest, never refit on validation/test;
- `tap_count=65`, Kaiser `beta=8.6`, 32-sample symmetric guard per frame for
  both A0 and A1;
- each MP/GMP architecture, solver and ridge identical between A0 and A1;
- common causal warm-up and bit-identical input/scored support across A0/A1;
- exact preregistered `decision_rule`:

  ```json
  {
    "primary_metric": "common_causal_interior",
    "gmp_a1_minus_a0_max_db": -0.25,
    "mp_corroboration_a1_minus_a0_max_db": 0.0,
    "required_splits": ["train_oof", "validation"],
    "require_full_record_same_sign": true,
    "fallback_variant": "a0",
    "accepted_a1_scope": "sensitivity_protocol_not_proven_feedback_deembedding"
  }
  ```

- train/validation only; test access forbidden;
- report both protocols even if A1 is worse;
- immutable output publication: an existing owned artifact is an error, the
  canonical command has no `--overwrite`, and an atomically acquired
  single-writer lock prevents concurrent publication races.

The train OOF score is conditional on the one delay diagnostic frozen from the
complete training split: held-out frames are excluded from coefficient fitting,
but the delay is not re-estimated inside each fold. It must not be described as
nested OOF preprocessing. Validation is not used for fitting inside this
runner; however, the MP recipe was previously selected on this same validation
split, so its validation result is corroborative rather than independent.
The fixed GMP recipe was preregistered and is not tuned here.

Pooled full/interior metrics use the actual partial-frame samples. For the
OpenDPD-compatible mean-per-segment metric, the final partial input and
reference are right-zero-padded to the effective segment length, inference is
run on the padded input, and the resulting delayed output tail is included in
the segment average. The report records padding and tail-error counts.

The result JSON evaluates every decision predicate after all fixed fits and
provides a machine-readable A1/A0 recommendation. That recommendation does not
itself modify configs or freeze the primary protocol. A separate reviewed
decision commit must accept the recommendation or retain A0 before formal GMP
selection.

The correlation diagnostics currently read approximately −0.00719 sample for
DPA and +0.07726 sample for APA. These are hypotheses for sensitivity, not a
calibrated feedback-path delay and not automatic permission to promote A1 to
the primary protocol.

### 6.3 Existing MP PA selection — completed

Canonical clean-checkout selection commands:

```bash
.venv/bin/python -m experiments.select_pa_mp \
  --config experiments/configs/pa_mp_dpa200.json
.venv/bin/python -m experiments.select_pa_mp \
  --config experiments/configs/pa_mp_apa200.json
```

They read only train/validation and produce frozen NPZ, validation ledger and
selection manifest. Existing selected results are:

| Dataset | Validation pooled NMSE | Test pooled NMSE | Real MUL/sample |
|---|---:|---:|---:|
| DPA_200MHz | −34.9617 dB | −35.0990 dB | 792 |
| APA_200MHz | −37.0952 dB | −36.9905 dB | 960 |

These commands must not be rerun merely to change a report. In an intentional
full regeneration, archive/remove only the exact owned output directory,
rerun selection, review the new hashes, and treat all previous test artifacts
as invalid.

### 6.4 Frozen MP test — completed as separate commands

```bash
.venv/bin/python -m experiments.evaluate_frozen_pa \
  --selection-manifest \
  experiments/results/pa_mp_dpa200_selection/selection_manifest.json
.venv/bin/python -m experiments.evaluate_frozen_pa \
  --selection-manifest \
  experiments/results/pa_mp_apa200_selection/selection_manifest.json
```

The runner verifies config/model/source/train/validation hashes before its
first read of `test_input.csv` or `test_output.csv`; it does not refit.

### 6.5 MP train-OOF and validation residuals — completed

```bash
.venv/bin/python -m experiments.analyze_pa_residuals \
  --config experiments/configs/pa_residual_dpa200.json
.venv/bin/python -m experiments.analyze_pa_residuals \
  --config experiments/configs/pa_residual_apa200.json
```

This runner is deliberately MP-specific. It performs leave-one-explicit-frame
out discovery on train and a separate validation diagnostic; it never reads
test. The existing residual supports testing causal cross-memory/GMP before
adding slow state.

### 6.6 Causal factorized GMP selection — preregistered, not yet run

Committed configs:

- `experiments/configs/pa_gmp_dpa200.json`;
- `experiments/configs/pa_gmp_apa200.json`.

Each declares 139 architecture candidates that survive the strict MUL filter,
followed by eight ridge and seven non-duplicate truncated-SVD refinements for
the validation-selected architecture: 154 fits/dataset. All
selection-eligible topologies are causal and have zero lookahead.

Run DPA and APA as separate tasks:

```bash
.venv/bin/python -m experiments.select_pa_gmp \
  --config experiments/configs/pa_gmp_dpa200.json
```

```bash
.venv/bin/python -m experiments.select_pa_gmp \
  --config experiments/configs/pa_gmp_apa200.json
```

Do not run them until A0/A1 sensitivity is resolved or explicitly waived in a
small documented decision commit. Selection produces:

```text
experiments/results/pa_gmp_<dataset>_selection/
  selected_gmp_pa.npz
  selected_validation_evaluation.json
  validation_trials.json
  selection_manifest.json
```

After each selection, inspect rank/condition, input support, boundary
full/interior difference, operation/state count and hashes. Commit and push
the frozen selection before opening test.

### 6.7 Frozen GMP test — separate and not yet run

Only after the corresponding selection manifest is frozen:

```bash
.venv/bin/python -m experiments.evaluate_frozen_pa \
  --selection-manifest \
  experiments/results/pa_gmp_dpa200_selection/selection_manifest.json
```

```bash
.venv/bin/python -m experiments.evaluate_frozen_pa \
  --selection-manifest \
  experiments/results/pa_gmp_apa200_selection/selection_manifest.json
```

Test is a final report, not a reason to change topology, rank cutoff or ridge.
If test disappoints, record it as a negative result; do not reopen selection.

### 6.8 GMP residual analysis, then PA spline — blocked in this snapshot

The current `experiments.analyze_pa_residuals` loader is tied to
`MemoryPolynomialPA`. It must not be pointed at a GMP manifest. The next small
implementation task after frozen GMP is:

1. generalize only the model-loading/prediction part of the residual runner;
2. retain the same train-OOF/validation boundary-safe analysis and test guard;
3. add separate GMP residual configs and unit tests;
4. run and commit GMP residual artifacts;
5. only then choose between sparse complex spline-memory PA and
   memoryless spline/CPWL + short complex FIR.

No exact GMP-residual or PA-spline command is published yet because no callable
API/config exists. This is an implementation blocker, not a completed or
estimated experiment. State-conditioned PA remains prohibited without
independent long captures demonstrating slow thermal/bias state.

### 6.9 Existing first-stage spline DPD — retained, not the next PA step

These are the commands recorded by the old artifacts. They contain
`--overwrite` and therefore must run only in a clean checkout or after the
exact old result directory has been archived intentionally:

```bash
.venv/bin/python -m baseline.train_spline \
  --config experiments/configs/spline_dpa200.json --overwrite
.venv/bin/python -m baseline.train_spline \
  --config experiments/configs/spline_apa200.json --overwrite
```

Their test command is:

```bash
.venv/bin/python -m baseline.evaluate_spline \
  --dataset vendor/OpenDPD/datasets/DPA_200MHz \
  --training-report \
  experiments/results/spline_dpa200_surrogate/training_report.json \
  --pa-surrogate \
  experiments/results/spline_dpa200_surrogate/pa_surrogate.npz \
  --mode both \
  --output-json \
  experiments/results/spline_dpa200_surrogate/test_evaluation.json \
  --output-npz \
  experiments/results/spline_dpa200_surrogate/test_waveforms.npz \
  --overwrite
```

APA uses the analogous `APA_200MHz` and `spline_apa200_surrogate` paths.
These commands are preserved for reproducibility, but outputs remain
`surrogate-only`. They must not be used to bypass the A→B gate. The old memory
ablation `experiments/run_spline_memory_ablation.py` is likewise DPD code, not
a sparse spline-memory PA implementation.

### 6.10 Egor audit reproduction — completed diagnostic

```bash
.venv/bin/python -m experiments.reproduce_egor \
  --data-directory vendor/DPD_for_PA/data1 \
  --output-json experiments/results/egor_reproduction_dpa200.json
```

It reports PA-only, circular inverse→forward reconstruction and correct
desired-input surrogate path separately. It is not a primary PA/DPD baseline
until its split/evaluator contract is made apples-to-apples.

### 6.11 OpenDPD control — bundled evidence only on this host

The intended upstream reproduction is:

```bash
cd vendor/OpenDPD
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -v -m "not extended"
bash benchmark/reproduce_benchmark_report.sh --device 0
```

It is **not runnable as the full neural matrix on the current host**: no NVIDIA
device is detected, and archived neural checkpoint binaries are absent.
Installing upstream dependencies is also a separate environment task, not
part of the NumPy PA sweep.

## 7. Preserved ablation, robustness and hardware matrices

The first-stage DPD search space remains preregistered, but is deferred until
Gate A→B passes:

- memoryless complex spline:
  \(K\in\{8,12,16,24,32,48,64\}\);
- knots: uniform amplitude, uniform power, quantile and compression-aware;
- ridge:
  \(\{0,10^{-10},10^{-9},\ldots,10^{-2}\}\);
- second-difference smoothness starts at zero, then receives a separate
  validation-only local sweep;
- memory branches are added incrementally as signal delays `{0}`, `{0,1}`,
  `{0,1,2}`, then sparse selected delays;
- SPH uses spline followed by a short complex FIR with
  \(L\in\{2,4,8\}\);
- a neural residual branch is allowed only after a simpler branch ablation
  leaves a reproducible residual.

The PA contour uses the same anti-complexity rule: after causal GMP residual
analysis, test sparse complex spline-memory PA first, then spline/CPWL + short
FIR. Do not implement both simultaneously, and do not add slow state without
independent long captures.

Robustness is a separate stage:

- DPA and APA are reported independently, never pooled;
- when multiple operating points become available:
  train at point A, test at B, then calibrate with
  \(N=\{64,128,256,512,1024,2048,\ldots\}\) new samples;
- report quality versus sample count and wall-clock, coefficient drift and
  maximum stable update rate;
- waveform/PA transfer is labelled explicitly and is not an ordinary test
  split.

The existing `experiments/evaluate_fixed_point.py` is only a first-stage
spline-DPD surrogate evaluator. It currently covers FP16-like storage and
signed 16/12-bit paths; it does not cover 14 bit or the new PA families. Its
archival invocation below also overwrites its exact output and is not in the
active execution queue:

```bash
.venv/bin/python -m experiments.evaluate_fixed_point \
  --dataset vendor/OpenDPD/datasets/DPA_200MHz \
  --training-report \
  experiments/results/spline_dpa200_surrogate/training_report.json \
  --pa-surrogate \
  experiments/results/spline_dpa200_surrogate/pa_surrogate.npz \
  --output-json \
  experiments/results/spline_dpa200_surrogate/fixed_point_evaluation.json \
  --overwrite
```

Second-stage hardware acceptance requires a bit-accurate simulator for the
selected PA and DPD models at signed 16/14/12-bit coefficients and
activations, explicit input/output/accumulator/state formats, scale, rounding,
saturation and interpolation addressing. Full-record and arbitrary streaming
chunks must agree. Analytical operations and bytes remain separate from
measured FPGA/DSP latency, throughput, DSP packing and power.

## 8. Acceptance gates

### 8.1 PA model

A new PA model is retained as a Pareto point only if:

1. topology and solver were selected without test;
2. validation pooled NMSE improves, or a declared cost reduction compensates
   for a quality loss;
3. strict `<1000` real MUL/sample is satisfied for a qualifying Huawei
   candidate;
4. full-record and common-interior results show whether gain is a boundary
   artifact;
5. rank/condition and coefficient norm are numerically acceptable;
6. causal full-record and arbitrary chunk predictions agree;
7. no validation/test input exceeds model support without an explicit count;
8. fixed-point degradation is reported before a hardware claim.

If normalized error power is the intended Huawei error, final acceptance is
`<10^-5` or pooled NMSE `<-50 dB`; current MP does not pass it.

### 8.2 Gate A→B

Surrogate-based DPD optimization resumes only when:

1. PA validation error power is at least 10 dB below the DPD residual being
   resolved;
2. DPD ranking agrees on at least two independently fitted frozen evaluators;
3. predistorted drive remains inside verified evaluator support;
4. evaluator state/boundary/streaming semantics match deployment.

The 10 dB margin is a conservative internal criterion, not a recovered Huawei
requirement. Physical-PA remeasurement remains the decisive evidence.

### 8.3 “Better than OpenDPD”

The claim requires the same dataset or physical PA, split, gain/alignment,
framing, spectral definitions and test discipline; at least three seeds for
stochastic models; NMSE/EVM/ACLR/PAPR/peak drive; operation/state/memory
counts; calibration and inference timing; fixed point; and physical-PA
verification or an explicit `surrogate-only` limitation.

## 9. Runtime and capacity estimates

| Task | Current evidence / planning estimate on i5-12450H | Status |
|---|---|---|
| Final unit suite | 120 tests in 0.396 s measured | completed after runner/config validation |
| MP DPA 46-trial selection | 15.39 s sum of fit timers; selected fit 0.918 s; total wall not archived | completed |
| MP APA 46-trial selection | 43.23 s sum of fit timers; selected fit 1.988 s; total wall not archived | completed |
| MP residual OOF fitting | 3.94 s DPA / 2.96 s APA fit-only; analysis wall not archived | completed |
| A0/A1 fixed-model sensitivity | reserve 1–10 min/dataset after production configs exist; replace with measured wall time | not run |
| GMP DPA 154 fits | **estimate:** fixed 384-feature probe ≈1 s; reserve approximately 5–30 min | not run |
| GMP APA 154 fits | **estimate:** fixed 384-feature probes ≈4.4–5.3 s; reserve approximately 15–90 min | not run |
| Frozen PA test | seconds to a few minutes, no fit; record actual wall time | GMP not run |
| GMP residual | reserve 1–10 min/dataset after generic runner exists | not run |
| Old 280-candidate spline DPD fits | 21.23 s DPA / 55.25 s APA sum of stored fit timers; total wall not archived | completed, surrogate-only |
| Egor audit wrapper | 15.87 s total measured | completed diagnostic |
| Bundled full OpenDPD matrix | 16,369 s reported on RTX PRO 6000; not extrapolated to this CPU | not locally run |

The exploratory timing probes above use one fixed 384-feature fit; they are not
formal sweep results and do not imply perfectly linear scaling across feature
counts, ridge/SVD modes or ranks. GMP maximum preregistered dense calibration
matrix has 450 complex columns: approximately 158 MiB raw complex128 design
storage on DPA and 405 MiB on APA, before solver workspaces. Sequential
execution should fit 15 GiB, but this is not a measured peak-memory result.
Record `/usr/bin/time -v` maximum RSS and wall time for formal runs; replace the
planning ranges with measured values after the first completed sweep without
changing the preregistered grid.

Physical PA work has no meaningful local runtime estimate: it requires an RF
session, calibrated feedback path, operating-point metadata and newly captured
outputs for predistorted waveforms.

## 10. Planned order after this document

1. [x] Commit and push the implemented/tested A0/A1 runner as its own small
   task (`79089f8`).
2. [x] Add and review the portable DPA/APA sensitivity configs without running
   either dataset (`754a069`).
3. Run DPA train/validation sensitivity without test access; commit and push
   only its immutable result bundle.
4. Run APA as a separate task; commit and push only its immutable result
   bundle.
5. In a separate decision commit, inspect both machine-readable outcomes and
   freeze A0 or A1 independently for each PA. A runner recommendation is not
   itself the protocol decision.
6. Run one causal GMP selection per dataset and commit each frozen selection.
7. Run each frozen GMP test once in a separate commit.
8. Generalize residual analysis to GMP; run train-OOF/validation diagnostics.
9. Implement only the PA spline/FIR family supported by residual evidence.
10. Re-evaluate the A→B gate.
11. Only after the gate passes, evaluate DPD through frozen independent
   evaluators, then fixed point, robustness/adaptation and finally physical PA.
