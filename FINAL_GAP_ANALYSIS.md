# Final gap analysis: current evidence versus project goal

Дата среза: 2026-07-29.

## 1. Executive decision

Проект успешно исправил методологию и улучшил forward PA evaluator, но пока
**не имеет оснований** заявлять:

- `error < 10^-5` по неизвестной Huawei metric;
- DPD лучше OpenDPD apples-to-apples;
- fixed-point/real-time implementation готова;
- linearization физического PA или базовой станции Huawei доказана.

Лучший локальный forward PA result:

| Dataset | Frozen model | Test pooled NMSE | Relative error power | MUL/sample |
|---|---|---:|---:|---:|
| DPA_200MHz | causal GMP | −35.385 dB | 2.8940e−4 | 766 |
| APA_200MHz | causal GMP | −38.608 dB | 1.3778e−4 | 954 |

Обе точки удовлетворяют analytical `<1000 real MUL/sample`, но если
`10^-5` — normalized error power, они выше порога в 28.94× и 13.78×.

Gate A→B остаётся **closed**: PA evaluator недостаточно отделён по error power
от DPD residual, нет второго independent evaluator и нет physical-PA cascade.

## 2. Что следует из предоставленных Huawei slides

Slides явно показывают:

- RF nonlinear system identification с complex time sequences (X,Y,D);
- objective вида \(\mathcal{E}(f)=\lVert f(X)-Y\rVert_2^2\);
- PA nonlinearities with bandwidth, thermal and trapping-memory effects;
- classic Volterra/memory-polynomial, neural and CPWL model families;
- desired properties: strong expression, low complexity и coefficients,
  пригодные для real-time calculation;
- stated goals `E(f) < 10^-5` on verification sets и fewer than 1000 real
  multipliers;
- staged verification на public data, затем real-world service data.

Один slide **не задаёт полный acceptance contract**. Он не определяет
normalization, aggregation, exact model direction, RF setup, spectral masks,
fixed-point format или физический смысл “multiplier”. Поэтому рабочее
разделение на PA identification и independently evaluated DPD является
инженерным contract проекта, согласующимся со слайдами и OpenDPD, но не
выдаётся за дословную закрытую спецификацию Huawei.

Полный requirements audit: `REQUIREMENTS.md`.

## 3. Что уже доказано

### 3.1 Evaluation methodology

- Forward PA path унифицирован:
  `x_split -> frozen PA model -> y_hat -> measured y`.
- Alignment/gain diagnostics frozen по train; после prediction нет gain/delay
  fit.
- DPA/APA splits не смешиваются; framing/state reset explicit.
- Full-record pooled NMSE — primary, OpenDPD-compatible and common-interior
  metrics сохраняются отдельно.
- Config/model/source/data hashes проверяются до test access.
- GMP architecture/ridge выбраны на validation, затем coefficient-OOF audit и
  release gate выполнены до one-shot test.
- Streaming/reset equivalence GMP доказана в floating-point software.

Evidence: `PA_MODEL_BENCHMARK.md`, `EXPERIMENT_PLAN.md` и machine-readable
artifacts under `experiments/results/pa_gmp_*`.

### 3.2 Forward PA identification on measured captures

MP и causal factorized GMP fitted на measured input/output CSV и проверены на
held-out measured output:

| Dataset | MP test | GMP test | GMP gain | GMP cost |
|---|---:|---:|---:|---:|
| DPA | −35.099 dB | −35.385 dB | 0.286 dB | 766 MUL, 759 ADD |
| APA | −36.990 dB | −38.608 dB | 1.618 dB | 954 MUL, 947 ADD |

OOF improvement над matched MP воспроизводится: 0.295/0.301 dB full/common
DPA и 1.291/1.651 dB APA. Все folds full rank; frozen test не участвовал в
selection или release decision.

Это measured-data forward identification, но не новая measurement session:
CSV были собраны upstream, а physical PA не был повторно запущен нами.

### 3.3 Отрицательный APA widely-linear ablation

Preregistered causal residual corrections
\(\sum_{d\in D}b_d x^*[n-d]\) с supports `{0}`, `{0,1}`, `{0,1,2}` и
`{0,1,2,3,4}` были проверены поверх каждого coefficient-OOF GMP fold. Все
fits full rank и streaming/reset exact, но максимальный gain составил только
0.02735 dB full и 0.03055 dB common при 962 MUL/sample. Это ниже frozen
0.1 dB threshold, поэтому выбран `no_correction` и стоимость evaluator
осталась 954 MUL / 947 ADD.

Это доказывает только отсутствие практически значимого улучшения у
проверенной short-conjugate family на post-discovery internal resampling.
Validation уже была просмотрена, test не читался, physical IQ/PA attribution
не выполнялась. Evidence:
`experiments/results/pa_widely_linear_residual_apa200/`.

### 3.4 Code/repository audit conclusions

- OpenDPD neural DLA использует правильное deployment direction:
  desired `x -> DPD -> frozen PA`, target `g*x`; circular `y_test` input там не
  является основным DLA test.
- В Egor notebook ILA mapping `y/g -> x` допустим как training postdistorter,
  но cells 10/14 выполняют circular inverse→forward reconstruction известного
  `y_test`; cell 11 — отдельный correct-direction surrogate path.
- Egor cached (R^2) не заменяет complex NMSE/ACLR; cached values около
  −31.67 dB PA NMSE и −32.09 dB circular cascade не достигают −50 dB.
- `EnhancedESN_FAN` matrices исполняются dense: DPD pair (R=600) оценивается
  примерно в 728,622 real MUL/sample, а full DPD→PA surrogate cascade — около
  2,020,044; coefficient sparsity без sparse kernel не сокращает runtime.
- Separate random I/Q reservoirs не гарантируют phase equivariance; notebook
  PSD uses incorrect `fs=200`, `nperseg=256` versus DPA spec 800 MHz/2560.

Evidence: `research/opendpd_audit.md` и
`research/egor_pipeline_audit.md`.

## 4. Что доказано только на surrogate

Legacy complex spline-memory DPD `signal_delay_012` дал:

| Dataset | No DPD | With spline DPD | Legacy PA surrogate fidelity |
|---|---:|---:|---:|
| DPA | −20.189 dB | −29.864 dB | −30.130 dB |
| APA | −19.948 dB | −32.741 dB | −31.091 dB |

Путь использует desired (x), но output вычислен старым MP surrogate. На APA
DPD residual даже ниже error самого evaluator. ACLR/EVM/PAPR/peak artifacts
полезны для software regression, но не являются physical-PA evidence.

Также surrogate-only:

- first-stage 16/12-bit spline numerical emulation;
- arithmetic projection существующего DPD относительно нового GMP;
- Egor correct-direction and circular cascade через learned PA model;
- любые AM/AM/PSD plots без повторного physical measurement.

Bundled OpenDPD neural numbers — upstream numeric evidence, не локальный rerun:
checkpoint binaries отсутствуют, current host не имеет NVIDIA GPU. Их нельзя
объединять с локальными rows как единый execution environment.

## 5. Насколько PA evaluator ограничивает DPD conclusions

Internal 10 dB rule требует evaluator error power не более 10% от DPD residual
power. По test arithmetic projection:

| Dataset | DPD residual | GMP PA error | Margin | PA error / DPD residual |
|---|---:|---:|---:|---:|
| DPA | −29.864 dB | −35.385 dB | 5.521 dB | 28.0% |
| APA | −32.741 dB | −38.608 dB | 5.867 dB | 25.9% |

Evaluator error fraction всё ещё примерно в 2.8×/2.6× выше internal ceiling
10%. Кроме того, один differentiable/fitted evaluator может иметь structured
errors, которые DPD optimization эксплуатирует, даже если scalar NMSE кажется
достаточным. Поэтому истинный риск определяется не только margin:

- PA and DPD fitted на related captures;
- same model family bias может сохранять ranking artifact;
- predistorted drive может выходить за measured support;
- reset/frame boundary не обязательно соответствует continuous hardware;
- no feedback-path IQ/frequency-response de-embedding;
- no physical output confirms spectral regrowth after predistortion.

Release-gate PASS означал только, что frozen GMP можно открыть на test один
раз. Он не означает Gate A→B PASS.

Short widely-linear correction также не открыла Gate A→B: её лучший OOF
gain примерно в 3.7 раза меньше даже минимального 0.1 dB ablation threshold
и пренебрежимо мал по сравнению с недостающими 4.1 dB evaluator margin на
APA. Поэтому pseudo-correlation residual нельзя интерпретировать как готовый
путь к требуемой PA fidelity.

## 6. Чего не хватает для Huawei/base-station claim

### 6.1 Requirements

- exact definition of `E(f) < 10^-5`: SSE, MSE, normalized/relative error,
  per-frame or pooled, steady-state or full record;
- относится ли error к PA forward model, inverse/postdistorter, DPD cascade
  или всем компонентам;
- `<1000 multipliers`: operations/sample, physical DSP blocks, clock-amortized
  units или coefficient-update cost;
- maximum calibration/update latency and allowed compute platform;
- accepted coefficient/state/accumulator formats and saturation rules.

### 6.2 RF experiment

- DUT topology, carrier frequency, output power, backoff, bias, temperature;
- waveform/bandwidth/modulation and spectral masks;
- calibrated feedback transfer function, DC, IQ imbalance and noise floor;
- chronological independent captures and operating-point labels;
- physical PA output for each predistorted waveform;
- repeatability across sessions/devices.

### 6.3 Apples-to-apples OpenDPD

- runnable checkpoint or reproducible retraining;
- identical split/alignment/gain/framing/PSD/ACLR definitions;
- same frozen physical PA/evaluators;
- three stochastic seeds;
- operation schedule, calibration/inference wall time, memory and fixed point;
- peak drive/support/stability reporting.

### 6.4 Hardware

- bit-accurate selected PA and DPD at 16/14/12 bit;
- exact accumulator/intermediate widths and nonlinear primitive;
- synthesis/place-and-route target;
- measured throughput/latency/resources/power;
- streaming equivalence with deployment state lifetime.

## 7. Claims matrix

| Claim | Current status | Allowed wording |
|---|---|---|
| Low-complexity forward PA model under 1000 counted MUL | supported | causal GMP reaches −35.385/−38.608 dB on held-out measured captures at 766/954 MUL/sample |
| Huawei `10^-5` met | unsupported | metric unresolved; if normalized power, current GMP fails |
| DPD linearizes physical PA | unsupported | legacy DPD is surrogate-only |
| Better than OpenDPD | unsupported | no complete apples-to-apples run |
| Real-time FPGA-ready | unsupported | analytical counts only; no synthesis/fixed-point GMP |
| Online adaptation demonstrated | unsupported | no controlled operating-point captures/curves |
| Egor reservoir meets cost gate | refuted for dense code | dense (W@state) exceeds gate by orders of magnitude |
| Short APA conjugate residual branch improves GMP materially | refuted for checked supports | best internal-resampling gain is 0.027/0.031 dB, so `no_correction` remains selected |

## 8. Следующий эксперимент максимальной информационной ценности

Лучший independent-data experiment — **external-capture PA validation and
limited recalibration on `APA_200MHz_b`**, после уточнения metadata “measurement
B”. Причины:

- waveform/spec nominally matches `APA_200MHz`;
- это отдельный capture, поэтому он измеряет generalization, а не ещё один
  random split той же записи;
- он даёт новое test evidence после того, как primary APA test уже открыт;
- coefficient-only GMP/spline fits позволяют построить calibration quality
  versus samples/time без GPU.

Протокол:

1. Зафиксировать, какие axes изменились в measurement B; если ответа нет,
   label строго `capture transfer`.
2. До target access preregister source-frozen GMP и один residual-motivated
   low-complexity candidate: sparse complex spline-memory PA или spline/CPWL +
   short complex FIR.
3. Fit/select architecture только на `APA_200MHz` train OOF/validation.
4. Zero-shot coefficient-model score на `APA_200MHz_b`; measurement nuisance
   alignment/gain diagnostics разрешены только по target train с заранее
   frozen algorithm, без coefficient adaptation и без target validation/test.
5. Recalibrate coefficients на target train prefixes
   (N=64,128,256,\ldots\), choose update rule/N only on target validation.
6. Open target test once after freeze; report NMSE/error PSD/AM-AM/AM-PM,
   support, cost and wall time.
7. Не переходить к DPD, если evaluator margin/independent-ranking gate всё ещё
   не выполнен.

Пока provenance/operating-point metadata для measurement B не подтверждены,
следующий полностью локальный model experiment — заранее ограниченный
phase-equivariant spline/CPWL PA с короткой causal linear memory. Его topology,
identifiability constraint, operation count и OOF threshold должны быть
зафиксированы до fit; already-viewed validation остаётся descriptive, а ранее
открытый APA test запрещён для selection. Этот experiment может проверить
новый nonlinear inductive bias, но не заменяет independent capture.

Самый ценный **decisive** experiment остаётся physical PA remeasurement:
одинаковый desired waveform подать no-DPD/OpenDPD/new-DPD на один calibrated
DUT и измерить NMSE/EVM/ACLR/peak/stability. Без него claim для базовой станции
Huawei остаётся недоказанным независимо от surrogate score.
