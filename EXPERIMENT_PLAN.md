# Experiment plan and execution protocol

Дата среза: 2026-07-30.

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

Уточнение научного руководителя от 2026-07-30 меняет acceptance contract:
1000-real-MUL-equivalent time budget относится только к streaming DPD, не к
PA model; формулы слайда ненормативны; primary quality связана с затуханием
паразитных гармоник. Exact bands/reference/threshold и target timing reference
ещё не получены. Поэтому PA fidelity searches больше не ограничиваются
`<1000 MUL`, а DPD хранит полный operation vector и впоследствии проходит
target timing measurement.

## 1. Текущий статус

| Step | Status на дату среза | Допустимый вывод |
|---|---|---|
| Requirements audit | обновлён после ответа руководителя | DPD-only latency scope известен; exact spur metric и timing reference всё ещё неизвестны |
| A0 integer-only PA protocol | реализован | primary current protocol; fractional correction не применяется |
| A1 fractional-alignment sensitivity | DPA/APA train-OOF+validation выполнены; A1 rejected, A0 frozen | sensitivity не является measurement-path de-embedding |
| Complex MP PA selection | DPA и APA выполнены | measured forward validation result |
| Frozen MP PA test | DPA и APA выполнены | held-out measured forward test result |
| MP residual analysis | train OOF + validation выполнены | выбор следующего inductive bias, test не использован |
| Causal factorized GMP PA | DPA/APA selection, residual release gate и one-shot frozen test выполнены | measured forward PA identification; не DPD и не Gate A→B |
| GMP residual analysis | train coefficient-OOF + validation выполнены для DPA/APA | reproducible OOF gain; test не участвовал в release decision |
| Standalone APA SPH (`spline/CPWL + short FIR`) | выполнен train-only staged OOF; selected `K=32,L=8` rejected | 37 MUL/sample, но −30.4024 dB OOF; не evaluator и не Gate A→B |
| Non-factorized sparse spline-memory PA | выполнен train-only staged OOF; selected `K=12`, 6 branches rejected | 54 MUL/sample, −32.0300 dB OOF; reused validation only, Gate A→B closed |
| Residual-guided lag-9 sparse PA | выполнен preregistered train-only staged OOF; selected 9-branch `K=12` family | 72 MUL/sample, −37.7925 dB OOF, cheap-Pareto only; evaluator gate closed |
| `APA_200MHz -> APA_200MHz_b` capture transfer | pre-test selection + frozen held-out release completed | GMP −37.895 dB, sparse −34.801 dB target-test full NMSE after `N=16384` calibration; capture-transfer only |
| Bit-accurate PA arithmetic | DPA/APA source plus target-calibrated `APA_200MHz_b` train→freeze→validation completed; APA families include GMP and lag-9 sparse | 16/14/12-bit degradation, saturation and streaming evidence; no test/RTL/DPD cascade |
| Existing spline-memory DPD | выполнен через старый MP surrogate | surrogate-only; не новый cross-evaluator result |
| Frozen spectral-region evaluator + input-only validation replay | DPA/APA `signal_delay_012` replay completed after preregistration | conventional baseband diagnostics only; no measured output opened; no model reselection |
| DPD-only timing diagnostic | DPA/APA completed on one pinned CPU core with paired/interleaved 1000-MUL scalar reference | exact streaming equivalence and host trace only; customer/target gate not evaluable |
| OpenDPD neural PA/DPD | bundled numeric evidence доступен; checkpoint binaries отсутствуют | не локальный rerun |
| Physical PA verification | недоступна | никаких over-the-air/bench claims |

Gate A→B сейчас закрыт. Lag-9 result is a within-capture PA-model result;
arithmetic margins из уже существующих DPD и PA чисел являются projections, а
не выполненными independent cascade experiments.

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
`APA_200MHz_b` are also present. `APA_200MHz_b` is a separate
capture-transfer target with its own 60/20/20 framing; it must not be appended
to the primary train/test rows as if samples came from one PA session. Its
held-out target split is released only through the frozen transfer command in
§6.17.

Availability of evaluators/checkpoints:

- frozen MP NPZ, selection, validation, test and residual artifacts exist in
  `experiments/results/pa_mp_{dpa200,apa200}_selection/` and
  `experiments/results/pa_mp_{dpa200,apa200}_residuals/`;
- frozen GMP selection artifacts exist in
  `experiments/results/pa_gmp_{dpa200,apa200}_selection/`; residual/release
  artifacts are in `pa_gmp_{dpa200,apa200}_residuals/`, and one-shot test
  artifacts are in `pa_gmp_{dpa200,apa200}_test/`;
- OpenDPD bundled JSON records neural checkpoint paths and hashes, but no
  `.pt`, `.pth`, `.ckpt` or `.onnx` binary is present in the vendored tree;
- MP/GMP OpenDPD controls can be refit from CSV because they are closed-form;
- first-stage spline DPD and its old MP surrogate NPZ files exist, but are not
  independent PA evaluators;
- frozen transfer pre-test and held-out bundles exist in
  `experiments/results/pa_transfer_apa200_to_b_{pretest,test_release}/`;
- independent release verification is
  `experiments/results/pa_transfer_apa200_to_b_test_release_verification.json`;
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
- bit-accurate fixed-point degradation and chunk equivalence where a frozen
  arithmetic contract exists; current PA reports cover DPA/APA source models
  and target-calibrated APA-B payloads. Only DPD fixed-point remains pending.

Current PA error PSD uses a periodic Hann window, `nfft=nperseg`, 50% overlap,
constant detrend and density scaling, normalized by integrated measured-output
power. DPA uses `fs=800e6`, `nperseg=2560`; APA uses `fs=983.04e6`,
`nperseg=19662`. OpenDPD spectral-bin “EVM”, strongest-inband-subchannel ACLR
and conventional total-main-band ACLR remain separately labelled definitions.

`error < 10^-5` is not an active Huawei gate. If relative error power is
reported as an optional diagnostic, `10^-5` is mathematically equivalent to
pooled NMSE `−50 dB`; it does not control architecture selection or acceptance.

#### 5.1.1 Fixed-point PA execution already completed

The preregistered APA arithmetic matrix is declared in
`experiments/configs/pa_fixed_point_apa200.json` and executed by
`experiments/evaluate_fixed_point_pa.py`.  The runner:

1. verifies train/validation file and frozen-model hashes;
2. loads train only and freezes all scales from train peaks and frozen
   coefficients;
3. loads validation only after the freeze;
4. evaluates signed 16/14/12-bit GMP and lag-9 sparse PA with explicit
   accumulator/state widths and saturation counters;
5. checks reset-per-frame and arbitrary-chunk bit equivalence;
6. writes no test payload and cannot select a format from validation.

The committed APA report shows the following validation degradation relative to
the corresponding FP32 PA prediction:

| Model / bits | PA NMSE | Fixed-vs-float NMSE | Saturation |
|---|---:|---:|---|
| GMP / 16 | −38.6459 dB | −64.20 dB | none |
| GMP / 14 | −38.4320 dB | −51.62 dB | none |
| GMP / 12 | −34.4282 dB | −36.41 dB | none |
| lag-9 sparse / 16 | −37.8604 dB | −77.29 dB | none |
| lag-9 sparse / 14 | −37.8523 dB | −65.29 dB | none |
| lag-9 sparse / 12 | −37.7464 dB | −53.59 dB | none |

These are forward PA identification metrics on APA measured train/validation,
not DPD cascade quality.  The raw sparse schedule counts six integer
divisions; reciprocal-multiply mapping must be charged separately when
comparing hardware cost.

The same contract was run separately on DPA GMP using
`experiments/configs/pa_fixed_point_dpa200.json`; validation fixed NMSE is
`−35.3633/−35.3490/−35.2975 dB` at 16/14/12 bits versus float `−35.3659 dB`,
with zero saturation and exact chunk equivalence.  APA sparse topology is not
silently applied to DPA because it has no DPA-specific fit.

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

Для уже завершённых PA searches strict `<1000 MUL/complex sample` сохраняется
как historical preregistered search bound. Это не Huawei gate для PA model.
Для deployment DPD hard condition имеет вид
`T_DPD/sample <= T_reference(1000 real MUL)` на одном target/word length и
охватывает все operations/memory effects. Parameter count никогда не
подменяет ни operation vector, ни measured latency.

## 6. Exact commands and execution order

### 6.1 Tests

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest discover -s tests -v
```

Последний recorded full-suite result после hardened timing runner:
**291/291 tests passed in 7.064 s**. Более ранние 286/286, 272/272, 257/257,
234/234, 222/222, 201/201, 131/131 и 120-test snapshots являются
историческими и не заменяют текущий count.
Следующий code change обязан сохранить новый wall time/execution record;
unit-suite timing не является inference benchmark.

### 6.2 A0/A1 alignment sensitivity — completed, A0 frozen

The current primary protocol is integer-only. Inside the paired sensitivity
runner, A0 is a zero-fractional-shift control passed through the same FIR
implementation and symmetric crop as A1. Its score is therefore an
equal-support control, not a bit-identical replay of the existing uncropped MP
artifact. The reusable A1 transform is in
`baseline/fractional_alignment.py`; the fixed-recipe train-OOF/validation
runner is `experiments/evaluate_fractional_alignment_sensitivity.py`.
Infrastructure tests confirm frame-safe equal support, fixed MP/GMP recipes,
hash guards, immutable single-writer publication and no test access. Runner
commit `79089f8` and config commit `754a069` are frozen. Выполненные команды:

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
  multiplications per complex sample; это frozen historical PA-search
  contract, не актуальный Huawei DPD acceptance rule;
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

Result JSON оценивает каждый predicate. Отдельный reviewed decision artifact
`experiments/results/pa_alignment_protocol_decision.json` заморозил A0 для
обоих datasets до formal GMP selection; SHA-256:
`c4554c6d62f22bd66420a016743650add4e4379dedb0c84748087e28b54fc2a8`.

| Dataset | GMP OOF common Δ A1−A0 | GMP validation common Δ | Decision |
|---|---:|---:|---|
| DPA_200MHz | −0.0011406 dB | −0.0001744 dB | A0: improvement far below −0.25 dB gate |
| APA_200MHz | −0.0084092 dB | −0.0065438 dB | A0: below gate and OOF full-record sign reversal |

Test split не читался. A0 означает integer delay zero/no fractional
transform, а не calibrated feedback-path correction.

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

Runner теперь dispatches frozen complex MP/GMP from an immutable schema-2
config. Для каждой OOF fold coefficients refit только на remaining train
frames; validation остаётся отдельной diagnostic. Test никогда не читается.
Historical MP configs остаются воспроизводимыми, а GMP использует отдельные
hash-bound configs из раздела 6.8.

### 6.6 Causal factorized GMP selection — completed

Committed configs:

- `experiments/configs/pa_gmp_dpa200.json`;
- `experiments/configs/pa_gmp_apa200.json`.

Each declares 139 architecture candidates that survive the strict MUL filter,
followed by eight ridge and seven non-duplicate truncated-SVD refinements for
the validation-selected architecture: 154 fits/dataset. All
selection-eligible topologies are causal and have zero lookahead.

Выполненные отдельные commands:

```bash
.venv/bin/python -m experiments.select_pa_gmp \
  --config experiments/configs/pa_gmp_dpa200.json
```

```bash
.venv/bin/python -m experiments.select_pa_gmp \
  --config experiments/configs/pa_gmp_apa200.json
```

Selection outputs:

```text
experiments/results/pa_gmp_<dataset>_selection/
  selected_gmp_pa.npz
  selected_validation_evaluation.json
  validation_trials.json
  selection_manifest.json
```

Frozen winners:

| Dataset | Winner | Ridge/solver | Validation full/common | Cost | Final fit / selection wall |
|---|---|---|---:|---:|---:|
| DPA | `both_k4_m1`, `ka7/la24`, `kb4/mb1`, causal `kc4/mc1` | 1e−5 / `ridge_lstsq` | −35.3659/−35.4684 dB | 766 MUL, 759 ADD, 356 complex coeff. | 1.234/70.988 s |
| APA | `both_k2_m2`, `ka7/la30`, `kb2/mb2`, causal `kc2/mc2` | 1e−7 / `ridge_lstsq` | −38.6653/−38.7346 dB | 954 MUL, 947 ADD, 444 complex coeff. | 5.555/212.762 s |

Selection manifests:

- DPA SHA-256 `933ee11379fc5c825ee3bc8aa5f87592963357abd3c868c42d5fc52d1902728e`;
- APA SHA-256 `ef48c9afdfc24ab6066e51f14e3b5abe6b306aa4332313bbd8e5fc83620018dd`.

Оба manifests фиксируют `test_split_accessed=false`. Их нельзя перезапускать
ради улучшения отчёта после просмотра test.

### 6.7 Frozen GMP test — completed once per dataset

Commands были разрешены только после coefficient-OOF residual audit и
machine-readable release-gate PASS:

```bash
.venv/bin/python -m experiments.evaluate_frozen_pa \
  --selection-manifest \
  experiments/results/pa_gmp_dpa200_selection/selection_manifest.json \
  --output-dir experiments/results/pa_gmp_dpa200_test
```

```bash
.venv/bin/python -m experiments.evaluate_frozen_pa \
  --selection-manifest \
  experiments/results/pa_gmp_apa200_selection/selection_manifest.json \
  --output-dir experiments/results/pa_gmp_apa200_test
```

| Dataset | Full pooled | OpenDPD-compatible | Common interior | Refit/gain/delay fit |
|---|---:|---:|---:|---|
| DPA | −35.385021 dB | −35.398306 dB | −35.419159 dB | false/false/false |
| APA | −38.608112 dB | −38.608112 dB | −38.707462 dB | false/false/false |

Test был открыт ровно один раз per dataset. Разрешение израсходовано; команды
нельзя повторять с `--overwrite`. Это workflow-specific seal, поскольку
исторический MP workflow ранее использовал тот же dataset test split. Test
остался final report и не стал причиной менять topology/ridge.

### 6.8 GMP residual analysis — completed before test release

Generic runner, schema-2 configs и tests завершены. Выполненные commands:

```bash
.venv/bin/python -m experiments.analyze_pa_residuals \
  --config experiments/configs/pa_gmp_residual_dpa200.json
.venv/bin/python -m experiments.analyze_pa_residuals \
  --config experiments/configs/pa_gmp_residual_apa200.json

.venv/bin/python -m experiments.decide_gmp_test_release \
  --config experiments/configs/pa_gmp_residual_dpa200.json
.venv/bin/python -m experiments.decide_gmp_test_release \
  --config experiments/configs/pa_gmp_residual_apa200.json
```

| Dataset | Train OOF full/common | Validation full/common | OOF gain над matched MP full/common | Residual wall |
|---|---:|---:|---:|---:|
| DPA | −35.3157/−35.4224 dB | −35.3659/−35.4684 dB | 0.2952/0.3009 dB | 10.259 s |
| APA | −38.3454/−38.7505 dB | −38.6653/−38.7346 dB | 1.2911/1.6506 dB | 24.872 s |

Все folds full rank; support, OOF→validation, boundary, operation и streaming
predicates прошли. Release reports разрешили только one-shot test и явно не
установили Gate A→B, fixed-point readiness или physical-PA validity.

Следующая ограниченная гипотеза выбрана из residual evidence:
APA residual содержит устойчивую pseudo-correlation с каузальным
`conj(x[n-d])` на лагах 0…3. Поэтому перед nonlinear spline/CPWL
family выполняется дешёвый widely-linear/IQ audit. Это diagnostic
measurement/PA asymmetry, а не заявление о физическом PA mechanism.
State-conditioned PA остаётся запрещён без independent long captures.

### 6.9 APA widely-linear residual audit — completed negative result

Preregistration:
`experiments/configs/pa_widely_linear_residual_apa200.json`.
Он создан до любого candidate fit и фиксирует

\[
\hat y_{WL}[n]=\hat y_{GMP}[n]+\sum_{d\in D}b_d x^*[n-d].
\]

Correction обучается two-stage: в каждом leave-one-frame-out fold
GMP с уже frozen topology/solver переобучается только на fit frames,
затем coefficients `b_d` fit только на residual этих же fit frames. Joint GMP/WL
refit в этом audit запрещён: нужно изолировать цену и эффект
самой conjugate branch. Complex ridge фиксирован как `1e-8`,
intercept запрещён.

| Support `D` | Real MUL/sample | Real ADD/sample | Stored real coefficients |
|---|---:|---:|---:|
| no correction | 954 | 947 | 888 |
| `{0}` | 958 | 951 | 890 |
| `{0,1}` | 962 | 955 | 892 |
| `{0,1,2}` | 966 | 959 | 894 |
| `{0,1,2,3,4}` | 974 | 967 | 898 |

Конвенция: complex multiply = 4 real MUL + 2 real ADD;
добавление complex tap к base output = ещё 2 real ADD. Conjugation —
sign/wiring, existing GMP delay state переиспользуется.

Selection score — minimum из full-record и common-interior OOF gains над
`no_correction`. Оба gain должны быть не менее 0.1 dB; из eligible
вариантов выбирается минимальный support в 0.02 dB от лучшего score.
Отрицательные lags означают future samples и недопустимы в inference.

И train OOF residual, и validation residual уже просмотрены при выборе
семейства/support ceiling. Поэтому result маркируется
`post_discovery_internal_resampling_only`; validation можно показать только
как descriptive reused split, а test читать нельзя. Independent acceptance
требует нового capture/operating point и measurement-path IQ audit.
DPA-specific delays по уже просмотренному DPA residual не tuning.

Exact completed command:

```bash
.venv/bin/python -m experiments.audit_widely_linear_pa \
  --config experiments/configs/pa_widely_linear_residual_apa200.json
```

| Candidate | OOF full gain | OOF common gain | Minimum fold full/common | Decision |
|---|---:|---:|---:|---|
| `conj_d0` | 0.02677 dB | 0.02978 dB | 0.02408/0.02673 dB | ineligible |
| `conj_d0_d1` | **0.02735 dB** | **0.03055 dB** | 0.02391/0.02690 dB | ineligible |
| `conj_d0_d2` | 0.02684 dB | 0.02990 dB | 0.02450/0.02702 dB | ineligible |
| `conj_d0_d4` | 0.02479 dB | 0.02956 dB | 0.01840/0.02688 dB | ineligible |

Все fits были full rank; frame reset и arbitrary-chunk streaming дали exact
equivalence. Ни один support не прошёл 0.1 dB full/common threshold, поэтому
frozen selection — `no_correction`: 954 MUL, 947 ADD, 888 real coefficients
и 236 real state values. Reused-validation metrics остались baseline
−38.66526/−38.73463 dB full/common и не участвовали в selection. Total wall
time — 14.805 s, OOF fit time — 13.224 s. `test_split_accessed=false`;
test hashes и test-named result files отсутствуют.

Evidence bundle:
`experiments/results/pa_widely_linear_residual_apa200/`. Результат отвергает
практическую полезность проверенных short conjugate supports при текущем
threshold, но не идентифицирует физический источник residual и не является
independent validation.

### 6.10 APA proper-complex long-FIR residual audit — completed negative result

После conjugate fallback train-OOF и already-viewed validation residual
показали согласованный proper-complex correlation peak на causal lags 43…48.
До fit был отправлен config
`experiments/configs/pa_long_fir_residual_apa200.json`, фиксирующий

\[
\hat y_{FIR}[n]=\hat y_{GMP}[n]+\sum_{d\in D}b_d x[n-d].
\]

Exact completed command:

```bash
.venv/bin/python -m experiments.audit_complex_fir_pa \
  --config experiments/configs/pa_long_fir_residual_apa200.json
```

| Candidate | MUL/ADD/state | OOF full/common gain | Minimum fold full/common | Decision |
|---|---:|---:|---:|---|
| `proper_d45` | 958/951/268 | 0.01332/0.01469 dB | 0.00834/0.00935 dB | ineligible |
| `proper_d44_d46` | 966/959/270 | **0.01818/0.02007 dB** | 0.01142/0.01265 dB | ineligible |
| `proper_d43_d48` | 978/971/274 | 0.01775/0.01995 dB | 0.01155/0.01441 dB | ineligible |
| `proper_d42_d49` | 986/979/276 | 0.01769/0.02013 dB | 0.01143/0.01435 dB | ineligible |

Все folds дали положительный gain, были full rank и прошли exact reset/
arbitrary-chunk streaming checks. Но ни один support не достиг frozen 0.1 dB
full/common threshold, поэтому selected candidate — `no_correction` и
evaluator остался на 954 MUL / 947 ADD / 236 state reals. Baseline OOF
воспроизведён как −38.34541/−38.75053 dB full/common. Reused validation
остался −38.66526/−38.73463 dB и не участвовал в selection.

Total wall time — 25.473 s, OOF fit time — 23.395 s. Bundle содержит только
train/validation, `test_split_accessed=false`, test hashes и test-named files
отсутствуют. При fallback selected/base predictions побитно одинаковы.
Evidence:
`experiments/results/pa_long_fir_residual_apa200/`.

Config был committed до реализации counter, поэтому manifest честно отмечает
ожидаемый mismatch preregistered/current hash только для
`baseline/complexity.py`; hashes numerical discovery artifacts, GMP и
residual analyzer совпали. Result class остаётся
`post_discovery_internal_resampling_only`, не independent confirmation.

Frozen follow-up: не расширять long linear delay grid и не превращать те же
lags в nonlinear spline branches без отдельного evidence. Standalone
phase-equivariant spline/CPWL + short complex FIR PA теперь проверен как SPH;
его отрицательный result описан ниже. Non-factorized sparse follow-up теперь
также опубликован как отдельный negative evaluator result.

### 6.11 APA standalone SPH selection — completed negative result

The preregistered standalone candidate was executed only on `APA_200MHz`:

```bash
.venv/bin/python -m experiments.run_pa_sph \
  --config experiments/configs/pa_sph_apa200.json
```

The runner verified config/source/evidence/dataset hashes before loading
waveforms, searched train leave-one-frame-out OOF in four frozen stages, froze
the recipe and full-train parameters, and loaded validation only afterwards.
It never opened or hashed APA test. Publication was atomic and immutable.

| Item | Recorded result |
|---|---|
| Selected recipe | `amplitude_uniform_K32_L8_cr1e-08_sm1e-08_fr0e+00` |
| Train OOF full/common NMSE | −30.402374 / −30.437014 dB |
| Matched MP OOF full/common | −37.054329 / −37.099951 dB |
| Matched GMP OOF full/common | −38.345410 / −38.750526 dB |
| Loss versus MP full/common | +6.651955 / +6.662937 dB |
| Loss versus GMP full/common | +7.943037 / +8.313512 dB |
| Exact cost | 37 MUL, 36 ADD, 1 sqrt, 5 comparisons, 4 LUT |
| Storage/state | 78 real coefficients, 63 constants, 14 state reals |
| Search workload | 60 unique recipes, 180 completed OOF fit calls, 4 cache hits |
| Runtime | 620.531 s before publication on recorded CPU |
| Decision | `neither_evaluator_nor_cheap_pareto`; Gate A→B closed |

Hard-validity details are part of
`experiments/results/pa_sph_apa200_selection/staged_trials.json`. Raw K48/K64
variants were not promoted because at least one fold was rank-deficient
(`47/48` and `62–63/64` control ranks). Train OOF residual analysis found a
stable proper causal correlation peak around lags 22–24, while the slow-state
branch remains ineligible (`independent_capture_count=0`). This closes the
factorized SPH branch; it does not justify DPD tuning through SPH.

### 6.12 APA non-factorized sparse spline-memory selection — completed negative result

The preregistered config was
`experiments/configs/pa_sparse_spline_memory_apa200.json`; the command was:

```bash
.venv/bin/python -m experiments.run_pa_sparse_spline_memory \
  --config experiments/configs/pa_sparse_spline_memory_apa200.json
```

The runner verified all frozen input/evidence hashes before waveform access,
loaded only train, selected by explicit-frame leave-one-out OOF, refit and
hashed the final model, then loaded validation descriptively. It never opened
or hashed the APA test split. S0 had seven topology trials, S1 evaluated four
knot counts for the retained topology, and S2 evaluated five ridge values.
Because only one topology survived S0, the run made 16 stage associations,
14 unique recipe fits and 42 OOF fold fits; two associations were cache hits.

| Item | Recorded result |
|---|---|
| Selected recipe | `mixed_diagonal_long_K12_r0e+00_b0:0,1:1,2:2,22:22,23:23,24:24` |
| Train OOF full/common NMSE | −32.030011 / −32.088250 dB |
| Full-train refit full/common NMSE | −32.049190 / −32.107450 dB |
| Reused validation full/common NMSE | −32.048219 / −32.071529 dB |
| Matched MP OOF full/common | −37.054329 / −37.099951 dB |
| Matched GMP OOF full/common | −38.345410 / −38.750526 dB |
| Loss versus MP full/common | +5.024318 / +5.011702 dB |
| Loss versus GMP full/common | +6.315399 / +6.662276 dB |
| Exact cost | 54 MUL, 58 ADD, 6 sqrt, 24 comparisons, 12 LUT |
| Storage/state | 144 real coefficients, 23 constants, 48 state reals |
| Identifiability | rank 72/72, condition 78.57, minimum support 8 |
| Runtime | 33.5888 s before atomic publication |
| Decision | `neither_evaluator_nor_cheap_pareto`; Gate A→B closed |

Residual analysis found the strongest causal proper-correlation at lag 9
(`0.69064` train OOF and `0.69131` reused validation). Radial-envelope
correlation was secondary and the slow-state branch was ineligible because
`independent_capture_count=0`. This evidence was used only to preregister the
bounded lag-9 run below; it did not authorize an unbounded delay or knot sweep.

Evidence bundle:
`experiments/results/pa_sparse_spline_memory_apa200_selection/`.

### 6.13 APA residual-guided lag-9 sparse PA — completed cheap-Pareto result

The config was committed before fitting:
`experiments/configs/pa_sparse_spline_memory_lag9_apa200.json`.

Exact command:

```bash
.venv/bin/python -m experiments.run_pa_sparse_spline_memory \
  --config experiments/configs/pa_sparse_spline_memory_lag9_apa200.json
```

The runner verified 11 evidence records, 9 source hashes and the complete
dataset contract before waveform load. It evaluated nine S0 families, retained
one topology, tested `K={8,12,16}` and four ridge values. The actual run made
16 stage associations, 14 unique recipe evaluations, 42 OOF fold fits and
two cache hits; the preregistered worst-case budget was 22 associations/66
fold fits. Validation loaded only after recipe and full-train coefficient
freeze; test was never opened or hashed.

| Item | Recorded result |
|---|---|
| Selected recipe | `parent_plus_signal_lag8_10_current_envelope_K12_r1e-08_b0:0,1:1,2:2,22:22,23:23,24:24,8:0,9:0,10:0` |
| Train OOF full/common NMSE | −37.792478 / −37.852832 dB |
| Full-train refit full/common NMSE | −37.866643 / −37.927296 dB |
| Reused validation full/common NMSE | −37.860728 / −37.898605 dB |
| Gain versus frozen parent full/common | +5.762467 / +5.764583 dB |
| Minimum fold gain full/common | +5.717845 / +5.731338 dB |
| Gain versus matched MP full/common | +0.738150 / +0.752881 dB |
| Loss versus matched GMP full/common | +0.552932 / +0.897694 dB |
| Exact cost | 72 MUL, 82 ADD, 6 sqrt, 24 comparisons, 18 LUT |
| Storage/state | 216 real coefficients, 23 constants, 48 state reals |
| Identifiability | rank 108/108, augmented condition 2427.39 |
| Runtime | 62.5693 s before atomic publication |
| Decision | `cheap_pareto_only`; evaluator gate and Gate A→B closed |

Envelope-only `(0,d)` variants were hard-invalid because their designs were
rank-deficient; `K=16` was also rejected in OOF folds by the rank gate. Bundle:
`experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/`.

### 6.14 Existing first-stage spline DPD — retained, not the next PA step

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

### 6.15 Egor audit reproduction — completed diagnostic

```bash
.venv/bin/python -m experiments.reproduce_egor \
  --data-directory vendor/DPD_for_PA/data1 \
  --output-json experiments/results/egor_reproduction_dpa200.json
```

It reports PA-only, circular inverse→forward reconstruction and correct
desired-input surrogate path separately. It is not a primary PA/DPD baseline
until its split/evaluator contract is made apples-to-apples.

### 6.16 OpenDPD control — sealed CPU preflight completed

The intended upstream reproduction is:

```bash
cd vendor/OpenDPD
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -v -m "not extended"
bash benchmark/reproduce_benchmark_report.sh --device 0
```

The archived full CUDA matrix is not directly runnable on this host because no
NVIDIA device/checkpoint binaries are available. A separate CPU environment is
locked in `experiments/requirements/opendpd_cpu_py312.lock`; the local runner
`experiments/train_opendpd_pa.py` preserves upstream model/optimizer/framing
semantics while physically prohibiting test access.

Preregistered bounded command:

```bash
/tmp/dpd_opendpd_venv/bin/python experiments/train_opendpd_pa.py \
  --config experiments/configs/opendpd_pa_cpu_preflight_apa200.json \
  --max-epochs 1 --max-train-batches 10 --max-val-batches 1 \
  --output-root experiments/results/opendpd_pa_cpu_preflight_apa200
```

Fit timers were `0.594 s` GRU-H28, `0.649 s` TRes-GRU-H27 and `3.370 s`
TRes-DeltaGRU-H27. These are runtime-smoke values, not quality results. All
reports are train/validation-only and explicitly record 398 upstream-flat
windows crossing declared APA segment boundaries.

### 6.17 APA capture transfer and frozen held-out release — completed

Этот protocol отделяет выбор `N` от target test. Сначала source models,
topology и coefficient-only learning curves фиксируются на source/target
train+validation. Только затем release command получает explicit
`--release-test` acknowledgement и открывает target test.

Pre-test command (15.036 s recorded; target test forbidden):

```bash
.venv/bin/python -m experiments.transfer_pa_apa200_to_b \
  --config experiments/configs/pa_transfer_apa200_to_b.json
.venv/bin/python -m experiments.verify_pa_transfer_bundle \
  --bundle experiments/results/pa_transfer_apa200_to_b_pretest
```

Pre-test config SHA-256:
`4ca1c44bb66aa8b06eb00f164f1651d0622cb208d2d902991fd0b5bebec7ca2f`.
Published manifest SHA-256:
`570c3f98af77961f23d30eaa71f38f35c80745a523656042a2dfee1d7e8ddd00`.
Verifier reproduced 20 validation metric records and asserted
`test_never_opened_or_hashed=true`.

After validation selected `N=16384` for both families, the release config was
frozen (SHA-256
`e1808673eae86fff88c8693f680109d91f8c38ada6659d9794374741317a058f`) and the
following exact command was run:

```bash
.venv/bin/python -m experiments.release_pa_transfer_apa200_to_b \
  --config experiments/configs/pa_transfer_apa200_to_b_release.json \
  --release-test
.venv/bin/python -m experiments.verify_pa_transfer_release \
  --bundle experiments/results/pa_transfer_apa200_to_b_test_release \
  --output experiments/results/pa_transfer_apa200_to_b_test_release_verification.json
```

The immutable release manifest SHA-256 is
`067a00e66032ae3b0dfde35437a3116ea45931f65ef5bf833aca3ebafe635d07`;
verification artifact SHA-256 is
`399d378c1b6cbe33bde95a308aa4f7268201cc415b1089b61d7241fd1b2bb963`.
The target test hashes are input
`5027d3d69391ed22ad79c410831bdfed47b25045088dda0756801cf591c947bf` and
output
`9cdc65d4785ed0ef8abf33c5fab26fe1d18712f8b0e5216acaa1526f34f2f477`.
The final release process took `0.8733 s` before publication.

The access audit must be read together with
`experiments/results/pa_transfer_apa200_to_b_release_incident_001.json`
(SHA-256
`d03217f7ec74f49fbcd3f8619c528d7737b907e281d609b90363339ecacb2a34`).
The first access failed after loading the pair but before inference or metric
because a guard assumed three full `19662` train frames; the frozen framing is
`19662,19662,19656`. Only that guard was fixed. The retry used unchanged
models, coefficients, `N`, selection and metric protocol. Therefore the
published audit is `access_count=2`,
`strict_single_open_execution=false`, `test_used_for_selection=false`, and
`test_used_for_coefficient_fit=false`. The incident is deliberately retained.

### 6.18 DPD-only paired timing diagnostic — completed, not a hardware gate

The selected `signal_delay_012` DPD was timed without PA inference. Both runs
used the first 512 validation desired-input samples, CPU affinity `[0]`,
NumPy/BLAS/OpenMP thread-control environment variables set to 1, two warm-up
pairs and nine paired/interleaved repeats for chunks `1,8,64,512`. The actual
outer wrapper was:

```bash
taskset -c 0 env \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/bin/python experiments/benchmark_dpd_timing.py \
  --model experiments/results/spline_memory_dpa200/signal_delay_012.npz \
  --input experiments/results/dpd_spectral_replay_dpa200_validation/waveforms.npz \
  --output experiments/results/dpd_timing_dpa200_validation.json \
  --chunk-sizes 1,8,64,512 --max-samples 512 \
  --warmup 2 --repeats 9 \
  --git-dir /tmp/dpd_remote_audit/.git --require-clean-git \
  --workload-note \
  'no known concurrent project workload; OS background activity not sealed'
```

APA used the identical wrapper with `dpa200` replaced by `apa200` in model,
input and output paths. The JSON command records the inner Python invocation;
host metadata separately proves affinity `[0]` and records the three
thread-control environment variables.

| Dataset | chunk 1 / 8 / 64 / 512 median µs/sample | Artifact SHA-256 | Timed wall |
|---|---|---|---:|
| DPA | 177.496 / 21.700 / 3.076 / 0.625 | `bdd2a9d30ef903fbfa70fbfa910e287ca51058eb4842bd00c17f5ad32d11bb19` | 7.185 s |
| APA | 186.089 / 22.354 / 2.949 / 0.534 | `550f27cd592c143b932d740f6aa90ebc592d0ed4f85a4c6cdaf633ca92283553` | 7.448 s |

Runner SHA-256:
`354d9c17f4b6734ea0413af987e517cf6b5662fd954d4545af99ad5294aebc61`.
All chunk outputs equal the independent full-stream output exactly.
`customer_gate_evaluable=false`: the analytical 21-MUL schedule is not the
timed NumPy trace, and the scalar Python kernel is not a customer target
reference.

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
- SPH used spline followed by a causal complex FIR with
  \(L\in\{1,2,4,8\}\) in the completed APA search; the selected `L=8` result
  is retained as a negative control, not as a DPD evaluator;
- a neural residual branch is allowed only after a simpler branch ablation
  leaves a reproducible residual.

The APA conjugate and proper long-FIR residual audits are complete; both
failed their 0.1 dB internal thresholds, so `no_correction` remains frozen.
The standalone factorized SPH PA audit is also complete and failed the
three-dB cheap-Pareto gate, despite its 37-MUL cost. The first non-factorized
sparse family failed the quality gate; the separately completed lag-9 family
passes the cheap-Pareto gate but not the GMP/evaluator gate. Local delay
expansion is now frozen; do not add slow state without independent long
captures.

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
3. full operation/state/memory cost is reported, but the DPD latency budget is
   not applied to this offline PA evaluator;
4. full-record and common-interior results show whether gain is a boundary
   artifact;
5. rank/condition and coefficient norm are numerically acceptable;
6. causal full-record and arbitrary chunk predictions agree;
7. no validation/test input exceeds model support without an explicit count;
8. fixed-point degradation is reported before a hardware claim.

If normalized error power is retained as an additional diagnostic, `10^-5`
would equal pooled NMSE `<-50 dB`; current MP/GMP points remain above this
optional reference, without a Huawei pass/fail conclusion. The
clarified primary acceptance direction is harmonic/spur attenuation, whose
exact RF bands, reference and threshold remain unresolved.

### 8.2 Gate A→B

Surrogate-based DPD optimization resumes only when:

1. PA validation error power is at least 10 dB below the DPD residual being
   resolved;
2. DPD ranking agrees on at least two independently fitted frozen evaluators;
3. predistorted drive remains inside verified evaluator support;
4. evaluator state/boundary/streaming semantics match deployment.

The 10 dB margin is a conservative internal criterion, not a recovered Huawei
requirement. Physical-PA remeasurement remains the decisive evidence.

Current arithmetic projection (not a cascade measurement):

| Dataset | GMP validation / test fidelity | Old spline-DPD validation / test residual | Projected margin validation / test |
|---|---:|---:|---:|
| DPA | −35.366 / −35.385 dB | −30.532 / −29.864 dB | 4.834 / 5.521 dB |
| APA | −38.665 / −38.608 dB | −32.380 / −32.741 dB | 6.285 / 5.867 dB |

Обе точки ниже provisional 10 dB; второй independently fitted evaluator и
physical predistorted capture отсутствуют. Decision: **Gate A→B closed**.

### 8.3 DPD spectral and latency acceptance

DPD quality is evaluated only on the correct path
`desired x -> DPD -> frozen evaluator/physical PA`. Once the customer freezes
the definition, harmonic/spur attenuation is the primary acceptance metric.
Until then ACLR, baseband PSD, NMSE and EVM remain separately labelled
diagnostics; they are not silently renamed “harmonic attenuation”.

Complexity passes only after the streaming DPD implementation satisfies

```text
T_DPD/sample <= T_reference(1000 real multiplications)
```

on the same target, numeric format, resource allocation and timing protocol.
The analytical operation vector remains mandatory for explanation and
portability, but real-MUL count alone is not a pass.

The completed paired host-Python diagnostic validates instrumentation and
streaming semantics only. It is not this acceptance protocol and cannot be
converted into target throughput, FPGA resources or a Huawei pass.

### 8.4 “Better than OpenDPD”

The claim requires the same dataset or physical PA, split, gain/alignment,
framing, spectral definitions and test discipline; at least three seeds for
stochastic models; NMSE/EVM/ACLR/PAPR/peak drive; operation/state/memory
counts; calibration and inference timing; fixed point; and physical-PA
verification or an explicit `surrogate-only` limitation.

## 9. Runtime and capacity estimates

| Task | Current evidence / planning estimate on i5-12450H | Status |
|---|---|---|
| Final unit suite | 291/291 tests passed; 7.064 s observed | completed; not an inference benchmark |
| MP DPA 46-trial selection | 15.39 s sum of fit timers; selected fit 0.918 s; total wall not archived | completed |
| MP APA 46-trial selection | 43.23 s sum of fit timers; selected fit 1.988 s; total wall not archived | completed |
| MP residual OOF fitting | 3.94 s DPA / 2.96 s APA fit-only; analysis wall not archived | completed |
| A0/A1 fixed-model sensitivity | 26.191 s DPA / 30.268 s APA wall | completed, A0 frozen |
| GMP DPA 154 fits | 70.988 s wall; selected final fit 1.234 s | completed |
| GMP APA 154 fits | 212.762 s wall; selected final fit 5.555 s | completed |
| Frozen GMP test | 0.066 s DPA / 0.31 s APA process wall, no fit | completed once/dataset |
| GMP residual | 10.259 s DPA / 24.872 s APA wall | completed pre-test |
| APA transfer pre-test | 15.036 s wall | completed, test unopened |
| APA held-out release + verification | 0.873 s producer; verifier deterministic | completed, access count 2 with first metric-free failure |
| APA SPH four-stage search | 620.531 s before atomic publication | completed; train/validation only, Gate A→B closed |
| APA sparse staged search | 33.589 s before atomic publication | completed; 14 unique recipes/42 OOF fits, train/validation only |
| APA lag-9 sparse staged search | 62.569 s before atomic publication | completed; 14 unique recipes/42 OOF fits, train/validation only, cheap-Pareto only |
| DPA/APA source fixed-point PA matrix | ~6.2 s combined runner wall | completed; DPA GMP + APA GMP/sparse × 3 bit widths, no test access |
| Target-calibrated APA-B fixed-point PA matrix | 3.703 s runner wall | completed; GMP/sparse × 3 bit widths, train/validation only |
| Frozen DPA/APA spectral validation replay | 0.6 s combined replay + spectral evaluation wall | completed; input-only, no measured output, surrogate-only |
| Frozen DPA/APA legacy-test spectral replay | 7.3 s combined replay + spectral evaluation wall | completed descriptive re-evaluation; historical test access, no tuning |
| Pinned-core DPA/APA DPD timing diagnostics | 7.185 / 7.448 s for 512-sample, 2-warm-up, 9-pair protocols | completed host-Python diagnostic; no PA and no hardware pass |
| Old 280-candidate spline DPD fits | 21.23 s DPA / 55.25 s APA sum of stored fit timers; total wall not archived | completed, surrogate-only |
| Egor audit wrapper | 15.87 s total measured | completed diagnostic |
| APA OpenDPD CPU bounded preflight | 0.594/0.649/3.370 s candidate fit timers | 10 train batches + 1 validation batch; runtime only, test sealed |
| Bundled full OpenDPD matrix | 16,369 s reported on RTX PRO 6000; not extrapolated to this CPU | not locally run |

GMP wall times выше измерены для полного formal workflow, но peak RSS не был
измерен (`/usr/bin/time -v` недоступен в recorded environment). Maximum
preregistered dense calibration matrix имеет 450 complex columns:
приблизительно 158 MiB raw complex128 design storage на DPA и 405 MiB на APA,
до solver workspaces. Это analytical storage estimate, не measured peak.
Frozen-test host batch wall не является real-time latency/throughput.

Physical PA work has no meaningful local runtime estimate: it requires an RF
session, calibrated feedback path, operating-point metadata and newly captured
outputs for predistorted waveforms.

## 10. Planned order after this document

1. [x] Commit and push the implemented/tested A0/A1 runner as its own small
   task (`79089f8`).
2. [x] Add and review the portable DPA/APA sensitivity configs without running
   either dataset (`754a069`).
3. [x] Run DPA and APA train/validation sensitivity as separate immutable
   result commits, without test access.
4. [x] Freeze A0 independently for both PA in a reviewed decision artifact.
5. [x] Run one causal GMP selection per dataset and commit each frozen
   selection.
6. [x] Generalize residual analysis to GMP; run coefficient-OOF/validation
   diagnostics and release gates before test.
7. [x] Open each frozen DPA/APA GMP test exactly once in separate commits.
8. [x] Re-evaluate Gate A→B: closed; projected margin remains below 10 dB and
   no second independent evaluator/physical PA exists.
9. [x] Synchronize mandatory living docs and deprecate the stale secondary
   `experiments/experiment_plan.md` status ledger.
10. [x] Preregister the bounded APA widely-linear/IQ residual audit before fit,
    including exact operation counts, reused-validation status and no test access.
11. [x] Implement/test the conjugate correction and run APA post-discovery
    leave-one-frame-out audit under its frozen historical strict `<1000 MUL`
    bound without test access.
12. [x] Apply the negative-result branch: keep `no_correction`, make no
    physical IQ attribution and do not tune DPA conjugate delays.
13. [x] Preregister, implement and run the nested proper-complex long-FIR
    audit at causal lags 42…49 without test access.
14. [x] Apply its negative-result branch: keep `no_correction`; do not add a
    larger linear delay grid or spline branches at those lags without evidence.
15. [x] Preregister, implement and execute standalone spline/CPWL + short-FIR
    SPH PA; record its negative OOF/complexity gate and immutable bundle.
16. [x] Preregister, implement and execute the bounded non-factorized sparse
    spline-memory PA family; selected APA result fails the quality/evaluator
    gates and is preserved as a negative bundle.
17. [x] Preregister and run a narrow residual-guided lag-9 branch family with
    the same train-only OOF/rank/support/operation contract; cheap-Pareto and
    incremental gates passed, evaluator gate failed.
18. [x] Validate the surviving lag-9 sparse family and GMP on independent
    `APA_200MHz_b` capture with limited-calibration curves; target held-out
    release and access incident are disclosed in §6.17.
19. [ ] Obtain operating-point metadata for measurement B; until then label
    results `capture transfer`, not power/thermal adaptation.
20. [x] Run bit-accurate 16/14/12-bit PA-model evaluation for DPA/APA source
    models and hash-bound target-calibrated `APA_200MHz_b` GMP/sparse payloads;
    no target test access.
20a. [x] Add and execute the sealed spectral-region evaluator plus
    input-only frozen `signal_delay_012` validation replays for DPA/APA;
    preserve absolute/relative leakage definitions and no measured-output
    access.
20b. [x] Re-run the same frozen candidate on legacy test inputs only as a
    descriptive reproducibility check; mark `historical_test_access=true` and
    forbid any post-test selection.
20c. [x] Harden and run a DPD-only paired timing diagnostic on one pinned
    host CPU core; preserve raw repeats, dependency hashes and exact streaming
    equivalence, while declaring the customer gate not evaluable.
21. [ ] Reproduce a high-fidelity OpenDPD PA evaluator on the same frozen
    splits without applying the deployment-DPD latency cap.
22. [ ] Obtain controlled physical-PA data and only after Gate A→B passes
    evaluate DPD through frozen independent evaluators, then hardware claims.
