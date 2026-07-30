# Final gap analysis: current evidence versus project goal

Дата среза: 2026-07-30.

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
| APA_200MHz | lag-9 sparse, reused validation | −37.861 dB | 1.637e−4 | 72 |

Обе точки удовлетворяют analytical `<1000 real MUL/sample`, но если
`10^-5` — normalized error power, они выше порога в 28.94× и 13.78×.

Gate A→B остаётся **closed**: PA evaluator недостаточно отделён по error power
от DPD residual, нет второго independent evaluator и нет physical-PA cascade.

Последний train-only lag-9 candidate улучшил parent на `5.762/5.765 dB` и
превзошёл matched MP на `0.738/0.753 dB` при `72 MUL/sample`. Это первый
локальный sparse PA point, проходящий internal cheap-Pareto gate. Однако он
всё ещё на `0.553/0.898 dB` хуже matched GMP (full/common), проверен только
внутри APA capture и поэтому не является независимым evaluator или основанием
для продолжения DPD tuning.

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

### 3.4 Отрицательный APA proper long-FIR ablation

Следующий preregistered nested experiment проверил обычные causal features
\(x[n-d]\) около устойчивого proper-correlation peak на lag 45. Лучший
support `{44,45,46}` дал только 0.01818/0.02007 dB full/common OOF gain при
966 MUL / 959 ADD / 270 state reals. Все три folds улучшились, fits были full
rank и streaming/reset exact, но ни один support не достиг 0.1 dB threshold.
Поэтому снова выбран `no_correction` и evaluator не изменён.

Это показывает, что correlation peak воспроизводим, но объясняет слишком
малую долю residual error power. Validation была уже просмотрена, test не
читался; результат не является independent confirmation. Evidence:
`experiments/results/pa_long_fir_residual_apa200/`.

### 3.5 Отрицательный APA standalone SPH benchmark

После linear residual checks был выполнен отдельный preregistered search
phase-equivariant spline-Hammerstein PA. Это самостоятельный forward model,
а не additive correction к GMP:

```text
v[n] = x[n] C(|x[n]|)
y_hat[n] = v[n] + sum(l=1..7) h[l] v[n-l],  h[0] = 1
```

Search использовал только train leave-one-frame-out OOF для выбора recipe;
validation была загружена после freeze исключительно как reused descriptive
evidence, test не открывался. Выбран `K=32`, `L=8`, amplitude-uniform knots,
`control_ridge=1e-8`, `smoothness=1e-8`, `fir_ridge=0`.

| Model | Full OOF NMSE | Common OOF NMSE | MUL / ADD | Stored coeff. / state |
|---|---:|---:|---:|---:|
| Matched MP reference | −37.054329 dB | −37.099951 dB | 960 / 628 | 300 / 58 |
| Matched GMP reference | −38.345410 dB | −38.750526 dB | 954 / 947 | 888 / 236 |
| Selected SPH | −30.402374 dB | −30.437014 dB | **37 / 36** | **78 / 14** |

SPH therefore loses to matched MP by `6.651955/6.662937 dB` (full/common)
and to GMP by `7.943037/8.313512 dB`. Its 37-MUL cost is real and exactly
recomputed from the serialized model, but the preregistered cheap-Pareto gate
allowed at most 3 dB loss versus MP. Classification is consequently
`neither_evaluator_nor_cheap_pareto`; Gate A→B remains closed.

The failure is informative rather than inconclusive. The SPH OOF residual has
a stable causal proper-correlation peak at lags 22–24 (`0.684–0.723`), while
instantaneous radial-envelope correlation is only `0.024`. Increasing knot
count is not the missing degree of freedom: raw K48/K64 scores were rejected
because the control design was rank-deficient in one or more folds (`47/48`
and `62–63/64`). The factorized spline/FIR form cannot represent sufficiently
delay-dependent nonlinear coefficients. The next candidate should therefore be
a bounded **non-factorized sparse spline-memory** dictionary, with explicit
branch selection and the same train-only OOF/identifiability gates.

Evidence: `experiments/results/pa_sph_apa200_selection/selection_manifest.json`,
`staged_trials.json`, `train_oof_residual_analysis.json` and
`validation_reused_residual_analysis.json`. The immutable execution record
certifies 60 unique recipes, 180 completed OOF fits, exact streaming/reset
checks and `test_split_accessed=false`.

### 3.6 Отрицательный APA non-factorized sparse spline-memory benchmark

Preregistered candidate:

```text
y_hat[n] = sum_b x[n-m_b] C_b(|x[n-d_b]|)
```

Он был реализован как joint complex, phase-equivariant, causal forward PA
model с двумя локальными spline basis functions на branch. Selection прошла
только на трёх explicit train frames: S0 topology screen (7 families), S1
`K={8,12,16,24}` и S2 ridge sweep. Внутри preregistered retention window
осталась одна family; фактически выполнены 16 stage associations, 14 unique
recipes и 42 OOF fits. Validation загружена после recipe/parameter freeze;
test не открывался и не хешировался.

| Model | Full OOF NMSE | Common OOF NMSE | Full loss vs GMP | MUL / ADD | Stored coeff. / state |
|---|---:|---:|---:|---:|---:|
| Matched MP | −37.054329 | −37.099951 | +1.291081 dB | 960 / 628 | 300 / 58 |
| Matched GMP | −38.345410 | −38.750526 | 0 dB | 954 / 947 | 888 / 236 |
| Selected sparse | −32.030011 | −32.088250 | **+6.315399 dB** | **54 / 58** | **144 / 48** |

The selected recipe was
`mixed_diagonal_long_K12_r0e+00_b0:0,1:1,2:2,22:22,23:23,24:24`.
It passed hard identifiability gates (design rank `72/72`, augmented condition
`78.57`, minimum feature support `8`, bounded coefficients) and exact
streaming/reset checks. The cheap-Pareto gate allowed no more than 3 dB loss
versus MP; observed losses were `5.024318 dB` full and `5.011702 dB` common.
The final classification is therefore
`neither_evaluator_nor_cheap_pareto`; Gate A→B stays closed.

The run is still diagnostically valuable. Its residual has a strong causal
proper correlation at lag 9 (`0.69064` train OOF, `0.69131` reused validation),
while the earlier lag-22–24-only topology was not sufficient. Envelope
correlation is secondary (largest radial value about `0.140` at lag 2), and a
slow-state branch is not eligible without independent captures. This evidence
motivated the separately preregistered lag-9 neighborhood reported in
Section 3.7; it was not a claim that lag correlation alone would close the
6.3 dB evaluator gap.

Evidence: `experiments/results/pa_sparse_spline_memory_apa200_selection/` and
commit `5b804f3`. The immutable manifest records `test_split_accessed=false`,
runtime `33.5888 s`, and all artifact/input hashes.

### 3.7 Residual-guided lag-9 sparse PA: positive local result, closed evaluator gate

The follow-up config
`experiments/configs/pa_sparse_spline_memory_lag9_apa200.json` was committed
before fitting and froze nine topology families, `K={8,12,16}`, four ridge
values and 66 maximum OOF fit calls. It used the previously selected sparse
model as an immutable incremental control:

```text
(m,d) = (0,0),(1,1),(2,2),(22,22),(23,23),(24,24),(8,0),(9,0),(10,0)
K = 12, ridge = 1e-8
```

| Model / split | Full NMSE | Common NMSE | Gain over parent | Cost |
|---|---:|---:|---:|---:|
| Parent sparse train OOF | −32.030011 dB | −32.088250 dB | — | 54 MUL / 58 ADD |
| Lag-9 sparse train OOF | **−37.792478 dB** | **−37.852832 dB** | **+5.762467 / +5.764583 dB** | **72 MUL / 82 ADD** |
| Lag-9 sparse full-train refit | −37.866643 dB | −37.927296 dB | — | 72 MUL / 82 ADD |
| Lag-9 sparse reused validation | −37.860728 dB | −37.898605 dB | — | 72 MUL / 82 ADD |

The minimum fold gain was `+5.717845/+5.731338 dB`, so the preregistered
incremental gate passed. The candidate also passes the internal cheap-Pareto
threshold against MP (`0.738150/0.752881 dB` improvement), but remains
`0.552932/0.897694 dB` worse than GMP. Decision:
`cheap_pareto_only`; `evaluator_candidate_gate_passed=false`;
`gate_a_to_b_opened=false`.

The selected design is full rank (`108/108`), has augmented condition
`2427.39`, exact streaming/reset equivalence, 216 stored real coefficients and
48 state reals. Two repeated-signal-delay envelope-only families were
hard-invalid due partition-of-unity rank deficiency; the `K=16` selected
variant also failed rank in OOF folds. The residual lag-9 peak collapsed to a
maximum causal proper correlation of `0.07106` at lag 32 on train OOF. These
are strong within-capture diagnostics, not independent-capture evidence.

Immutable evidence:
`experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/`,
config SHA `93807ab6…`, recipe SHA
`5a2cb735d6637c5a9bd1f449268a2b9c98eeba3e6d9e65245c52fa273cc8a6c6`, runtime
`62.5693 s`, test access false.

### 3.8 Code/repository audit conclusions

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

### 3.9 Independent APA capture transfer: completed pre-test layer

The preregistered source-frozen transfer runner compared the two already
selected PA families on `APA_200MHz_b` without opening its held-out split.
Train/validation inputs are byte-identical between captures, while measured
outputs differ. The result is therefore evidence about capture transfer, not
known power, bias or thermal drift.

| Model / mode | Target validation full NMSE | Common NMSE | Calibration samples/frame | Fit time | MUL / ADD |
|---|---:|---:|---:|---:|---:|
| causal GMP zero-shot | −23.794841 dB | −23.793859 dB | 0 | 0 s | 954 / 947 |
| lag-9 sparse zero-shot | −23.701383 dB | −23.703027 dB | 0 | 0 s | 72 / 82 |
| causal GMP coefficient-only | **−37.890764 dB** | **−37.961563 dB** | 16384 | 6.860 s | 954 / 947 |
| lag-9 sparse coefficient-only | −35.358475 dB | −35.446027 dB | 16384 | 1.513 s | 72 / 82 |

The source models fit their original APA validation at approximately
−38.67/−37.86 dB, so the zero-shot B-capture drop is about 15 dB. Fixed
topology coefficient-only calibration recovers most of the GMP fidelity, while
the 72-MUL sparse model is about 2.53 dB worse after the same long prefix.
Short sparse prefixes can be worse than zero-shot; those points remain in the
published curve. GMP `N=64/128` are explicitly infeasible rank cases.

The target-train-only nuisance diagnostic found integer delay 0 and
`|complex-LS gain|=1.152146`; strict metrics intentionally did not refit gain.
The immutable bundle is
`experiments/results/pa_transfer_apa200_to_b_pretest/`, and
`experiments/verify_pa_transfer_bundle.py` reproduces 20 metric records and
all sealed hashes. This closes neither Gate A→B nor the physical-PA claim.

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

Short widely-linear и proper long-FIR corrections также не открыли Gate A→B:
их лучшие OOF gains примерно в 3.7× и 5.5× меньше даже минимального 0.1 dB
ablation threshold и пренебрежимо малы по сравнению с недостающими 4.1 dB
evaluator margin на APA. Поэтому residual correlation нельзя интерпретировать
как готовый путь к требуемой PA fidelity.

Standalone SPH даёт противоположный урок: малый operation count сам по себе не
делает evaluator пригодным. Его error power примерно `10^(6.65/10)≈4.6×`
выше matched MP и `10^(7.94/10)≈6.2×` выше GMP на OOF. Поэтому дальнейшая
DPD-оптимизация через SPH была остановлена до попытки использовать его как
frozen evaluator.

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
| Sparse APA long-FIR branch improves GMP materially | refuted for checked supports | best internal-resampling gain is 0.018/0.020 dB despite positive gains in every fold |
| Standalone APA SPH replaces GMP under the cheap-quality gate | refuted on train OOF | 37 MUL/sample, but 6.652 dB worse than matched MP and 7.943 dB worse than GMP |
| Non-factorized sparse spline-memory replaces GMP under the cheap-quality gate | refuted on train OOF | 54 MUL/sample, but 5.024 dB worse than MP and 6.315 dB worse than GMP |
| Lag-9 sparse PA is a cheap low-complexity Pareto point | supported within APA capture | 72 MUL/sample; 0.738/0.753 dB better than MP, but 0.553/0.898 dB worse than GMP |
| Lag-9 sparse PA is an independent DPD evaluator | unsupported | same-capture reused validation only; no second evaluator or physical cascade |
| Source PA coefficients transfer zero-shot to `APA_200MHz_b` | refuted on target validation | GMP −23.795 dB and lag-9 sparse −23.701 dB; target held-out sealed |
| Fixed-topology coefficient-only calibration recovers B-capture fidelity | supported on target validation only | GMP −37.891 dB and sparse −35.358 dB at N=16384/frame; capture-transfer scope |
| `APA_200MHz_b` proves power/thermal adaptation | unsupported | “measurement B” axes are unknown; no controlled labels |

## 8. Следующий эксперимент максимальной информационной ценности

Первый external-capture pre-test уже выполнен; следующий independent-data
step — **metadata-gated held-out release and, при наличии сопоставимых
метаданных, обратный transfer `APA_200MHz_b -> APA_200MHz`**. Причины:

- waveform/spec nominally matches `APA_200MHz`;
- это отдельный capture, поэтому он измеряет generalization, а не ещё один
  random split той же записи;
- source coefficients уже проверены zero-shot и после limited recalibration;
- target held-out evidence ещё не открыта и может быть опубликована только
  после freeze metadata/config/N;
- coefficient-only GMP/spline fits позволяют построить calibration quality
  versus samples/time без GPU.

Протокол:

1. Зафиксировать, какие axes изменились в measurement B; если ответа нет,
   label строго `capture transfer`.
2. Сохранить уже frozen source models, transfer config и selected-N rule;
   target test не использовать для изменения этих решений.
3. Выполнить target held-out release once после metadata/config freeze;
   report NMSE/error PSD/AM-AM/AM-PM, support, cost and wall time.
4. Если metadata подтверждает same DUT, выполнить обратный transfer
   `APA_200MHz_b -> APA_200MHz` на ещё не использованном split.
5. Только после известного operating-point experiment строить adaptation
   curves с physical power/temperature labels.
6. Не переходить к DPD, если evaluator margin/independent-ranking gate всё ещё
   не выполнен. Current lag-9 result already fails this gate; no further local
   delay dictionary expansion is authorized before transfer evidence.

Пока provenance/operating-point metadata для measurement B не подтверждены,
результат следует маркировать `capture transfer`, а не power/thermal
adaptation. Independent capture validation имеет большую информационную
ценность, чем повторное OOF tuning на `APA_200MHz`.

Самый ценный **decisive** experiment остаётся physical PA remeasurement:
одинаковый desired waveform подать no-DPD/OpenDPD/new-DPD на один calibrated
DUT и измерить NMSE/EVM/ACLR/peak/stability. Без него claim для базовой станции
Huawei остаётся недоказанным независимо от surrogate score.
