# DPD benchmark: evidence, protocol and release gate

Дата среза: 2026-07-30.

Уточнение научного руководителя от 2026-07-30: complexity limit относится
только к deployment DPD и задаётся как время всех операций, эквивалентное
времени 1000 real multiplications. PA evaluator этим limit не ограничен.
Primary quality должна отражать затухание паразитных гармоник; exact RF
regions, reference и threshold пока не определены. Поэтому ACLR/baseband PSD
сохраняются как diagnostics и не выдаются за окончательную customer metric.

## 1. Текущий статус

Контур B пока **не открыт для новой optimization**. Формальный PA-model этап
завершил causal GMP forward evaluators, но Gate A→B остаётся закрытым:

- DPA frozen GMP: −35.385 dB pooled test NMSE, 766 real MUL/sample;
- APA frozen GMP: −38.608 dB pooled test NMSE, 954 real MUL/sample;
- projected evaluator margin относительно существующего spline-DPD residual:
  5.521 dB DPA и 5.867 dB APA, ниже internal 10 dB criterion;
- APA standalone SPH PA search is complete but rejected: 37 MUL/sample at
  −30.402 dB train-OOF NMSE, 6.652 dB worse than matched MP;
- APA non-factorized sparse spline-memory PA search is complete but the first
  topology is rejected:
  54 MUL/sample at −32.030 dB train-OOF NMSE, 5.024 dB worse than matched MP
  and 6.315 dB worse than GMP;
- residual-guided lag-9 sparse PA is complete: 72 MUL/sample,
  −37.792 dB train-OOF NMSE and −37.861 dB reused validation; it passes the
  incremental/cheap-Pareto gates but remains 0.553 dB worse than GMP;
- independent `APA_200MHz_b` capture-transfer release is complete: after frozen
  `N=16384` coefficient-only calibration, GMP reaches −37.895 dB held-out
  full NMSE and lag-9 sparse −34.801 dB;
- DPA/APA frozen spline-memory DPD has a pinned-core, paired DPD-only timing
  diagnostic with exact chunk equivalence. It is host-Python evidence only,
  not the unknown Huawei target timing gate;
- sealed DPA/APA 16/14/12-bit DPD validation is complete with train-frozen
  formats, desired-input direction, zero saturation/collision and exact
  streaming. It remains a floating-surrogate result and did not select a
  precision;
- всё ещё нет второго evaluator с достаточным error margin, controlled
  operating-point labels или physical-PA output для predistorted waveform.

Поэтому в этом документе есть три строго разделённых evidence слоя:

1. уже выполненный **legacy surrogate-only** DPD benchmark;
2. preregistered future benchmark через independently frozen evaluator;
3. PA-model results (включая lag-9 cheap-Pareto result), которые не являются
   DPD results.

Ни один результат ниже не является доказательством линеаризации физического PA
Huawei или превосходства над OpenDPD.

## 2. Правильное направление DPD

Пусть desired complex baseband signal

\[
x[n]=I[n]+jQ[n],
\]

а frozen reference gain (g) оценивается только по train согласно выбранному
protocol. Deployment-like path:

```text
desired x_split -> frozen DPD -> z_split
                -> independently frozen PA / physical PA -> y_split
                -> compare y_split with g*x_split
```

На validation/test вход DPD всегда равен **desired (x)**. Measured
(y_\text{test}) не может формировать deployment input.

ILA/postdistorter diagnostic разрешён отдельно:

\[
u[n]=y[n]/g,\qquad \hat x[n]=P_\theta(u[n]).
\]

Путь

```text
known y_test/g -> inverse model -> x_hat -> forward model -> y_reconstructed
```

проверяет согласованность inverse и forward models на уже известном output.
Математически он минимизирует величину вида

\[
\lVert F(P(y/g))-y\rVert^2,
\]

а требуемая predistortion задача имеет вид

\[
\lVert F(P(x_\text{desired}))-g x_\text{desired}\rVert^2.
\]

Аргументы (y/g) и (x_\text{desired}) различны; хорошая reconstruction
первого выражения не доказывает generalization или линеаризацию второго.

## 3. Legacy surrogate-only evidence

Первый этап использовал короткий MP PA surrogate с test forward fidelity
−30.130 dB DPA и −31.091 dB APA. Лучшей validation-selected spline branch в
обоих случаях была `signal_delay_012`:

\[
z[n]=\sum_{m\in\{0,1,2\}}x[n-m]C_m(|x[n]|),
\]

где (C_m) — complex local linear-spline correction. Это корректный
desired-input cascade direction, но PA output был вычислен тем же surrogate,
а не повторно измерен.

### 3.1 Test metrics

| Metric | DPA_200MHz | APA_200MHz |
|---|---:|---:|
| No DPD vs ideal, pooled NMSE | −20.1886 dB | −19.9477 dB |
| Spline DPD vs ideal, pooled NMSE | −29.8645 dB | −32.7408 dB |
| PA surrogate forward fidelity | −30.1305 dB | −31.0908 dB |
| OpenDPD-compatible cascade NMSE | −30.0254 dB | −32.6154 dB |
| OpenDPD spectral EVM | −32.8672 dB | −33.8163 dB |
| ACLR left / right / average | −37.101 / −38.864 / −37.983 dB | −44.375 / −41.663 / −43.019 dB |
| Predistorted peak amplitude | 1.1865 | 0.9927 |
| Predistorted PAPR | 10.008 dB | 9.993 dB |

Primary warning: на APA reported DPD residual (−32.741 dB) формально ниже
forward error самого evaluator (−31.091 dB). Это признак evaluator-limited
inference, а не доказательство extra physical linearization.

Evidence:

- DPA:
  `experiments/results/spline_memory_dpa200/memory_ablation_report.json`;
- APA:
  `experiments/results/spline_memory_apa200/memory_ablation_report.json`.

### Frozen validation replay (new, no refit)

После preregistration был выполнен отдельный input-only replay уже выбранного
`signal_delay_012`. Он читает только `val_input.csv`; measured
`val_output.csv` не открывается. PA output в этой процедуре остаётся
суррогатным, поэтому это не physical-PA result и не claim относительно
OpenDPD.

The spectral bands below are explicitly conventional complex-baseband bands:
main `[-bw_main/2, bw_main/2)` and one adjacent band of width `bw_sub_ch` on
each side. They are not a customer-defined RF harmonic metric. `relative
leakage improvement` is computed after normalizing each signal by its own main
band power; absolute region-power suppression is retained separately.

| Dataset | Main power change | Left dBc no→DPD | Right dBc no→DPD | Relative leakage improvement L/R | Pooled NMSE no→DPD |
|---|---:|---:|---:|---:|---:|
| DPA_200MHz validation | −0.052 dB | −42.206 → −46.956 | −41.149 → −48.886 | +4.749 / +7.737 dB | −20.338 → −30.532 dB |
| APA_200MHz validation | −0.041 dB | −33.945 → −50.425 | −34.016 → −47.880 | +16.480 / +13.864 dB | −19.969 → −32.380 dB |

Predistorted peak/PAPR were `1.1926/10.469 dB` for DPA and
`1.0615/10.584 dB` for APA. These values are descriptive validation evidence;
no spectral winner was selected after looking at them. Immutable replay and
spectral bundles are stored under:

```text
experiments/results/dpd_spectral_replay_dpa200_validation/
experiments/results/dpd_spectral_replay_dpa200_validation_spectral/
experiments/results/dpd_spectral_replay_apa200_validation/
experiments/results/dpd_spectral_replay_apa200_validation_spectral/
```

### Fixed-point preservation of the frozen validation result

После отдельной preregistration выбранный three-branch DPD был выполнен в
signed 16/14/12-bit integer arithmetic. Runner открыл только desired
`train_input.csv` и `val_input.csv`: train зафиксировал input/output/coefficient
formats, validation не меняла scale, topology или precision. Frozen PA
surrogate оставался floating, чтобы измерять только деградацию DPD.

| Dataset / format | Fixed-vs-float drive NMSE | Cascade NMSE vs ideal | Configured absolute adjacent suppression L/R | Peak / PAPR |
|---|---:|---:|---:|---:|
| DPA float | reference | −30.5332 dB | 4.801 / 7.789 dB | 1.1926 / 10.469 dB |
| DPA 16 bit | −78.9244 dB | −30.5322 dB | 4.798 / 7.788 dB | 1.1926 / 10.469 dB |
| DPA 14 bit | −67.0128 dB | −30.5336 dB | 4.793 / 7.776 dB | 1.1927 / 10.469 dB |
| DPA 12 bit | −54.8714 dB | −30.5148 dB | 4.782 / 7.692 dB | 1.1922 / 10.466 dB |
| APA float | reference | −32.3840 dB | 16.521 / 13.905 dB | 1.0615 / 10.584 dB |
| APA 16 bit | −77.8237 dB | −32.3851 dB | 16.511 / 13.906 dB | 1.0615 / 10.583 dB |
| APA 14 bit | −65.7806 dB | −32.3703 dB | 16.535 / 13.871 dB | 1.0617 / 10.586 dB |
| APA 12 bit | −53.5984 dB | −32.3790 dB | 16.369 / 13.982 dB | 1.0608 / 10.578 dB |

Worst cascade-NMSE degradation was `0.0185 dB` DPA and `0.0137 dB` APA.
Worst loss in configured absolute adjacent-region suppression was `0.0966 dB`
DPA and `0.1520 dB` APA. Peak-amplitude change stayed within `0.00514 dB`.
All six fixed rows reported zero saturation and knot collisions, exact
arbitrary-chunk streaming and bit-exact 90-degree rotation for the evaluated
signals.

The integer reference schedule is `20 MUL, 25 ADD, 1 DIV, 1 integer sqrt,
8 LUT, 28 reads, 2 writes` per complex sample with four persistent state
reals; DPA/APA interval search costs five/three comparisons. This is not a
customer-equivalent timing result. Validation had already selected the
floating model historically, so no word length is declared a winner and this
is not untouched final evidence. Bundles:

```text
experiments/results/dpd_fixed_point_dpa200_validation/
experiments/results/dpd_fixed_point_apa200_validation/
experiments/results/dpd_fixed_point_{dpa200,apa200}_spectrum_{float,16bit,14bit,12bit}_validation/
```

They prove preservation of one legacy surrogate result under the declared
integer arithmetic. They do not prove physical-PA or independent-evaluator
fixed-point behavior, RF harmonic attenuation, RTL resources or target
latency.

### Descriptive legacy test replay (historically opened split)

The same frozen coefficients and surrogate were replayed on the already-opened
legacy test inputs. This is a reproducibility check, not a sealed independent
release: the original ablation had already read and hashed the test split, so
these bundles carry `historical_test_access=true`.

| Dataset | Main power change | Left dBc no→DPD | Right dBc no→DPD | Relative leakage improvement L/R | Pooled NMSE no→DPD |
|---|---:|---:|---:|---:|---:|
| DPA_200MHz legacy test | −0.047 dB | −40.526 → −46.485 | −39.149 → −48.248 | +5.959 / +9.098 dB | −20.189 → −29.864 dB |
| APA_200MHz legacy test | −0.060 dB | −34.219 → −51.262 | −34.221 → −48.547 | +17.043 / +14.326 dB | −19.948 → −32.741 dB |

No candidate, knot count, gain, delay or spectral band was changed after
seeing these values. The result remains `surrogate_only` and cannot establish
physical-PA harmonic attenuation.

### 3.2 Analytical inference cost of the selected legacy spline

| Dataset | Real MUL | Real ADD | Nonlinear | Compare | LUT | Real coeff. | State reals |
|---|---:|---:|---:|---:|---:|---:|---:|
| DPA | 21 | 24 | 1 | 5 | 6 | 144 | 4 |
| APA | 21 | 24 | 1 | 3 | 6 | 48 | 4 |

Counts assume binary interval selection, shared envelope lookup and
`1 complex multiply = 4 real MUL + 2 real ADD`. Они являются analytical
software schedule; hardware latency, memory bandwidth и DSP packing не
измерены. Различие stored coefficients связано с выбранным (K=24) DPA и
(K=8) APA.

### 3.3 DPD-only streaming timing diagnostic

The frozen three-branch model was timed on the first 512 validation desired
samples with CPU affinity `[0]`, NumPy/BLAS/OpenMP thread-control environment
variables set to 1, two warm-up pairs and nine paired/interleaved repeats per
chunk. PA inference was excluded.

| Dataset | chunk 1 | chunk 8 | chunk 64 | chunk 512 |
|---|---:|---:|---:|---:|
| DPA | 177.496 µs/sample | 21.700 µs | 3.076 µs | 0.625 µs |
| APA | 186.089 µs/sample | 22.354 µs | 2.949 µs | 0.534 µs |

Every chunk path is bit-for-bit equal to one continuous floating-point
record. The scalar Python reference median was about 254–283 µs/sample across
the paired rows. These ratios are diagnostics of two different Python
implementations, not a multiplication-equivalent hardware result: the NumPy
path does not implement the analytical 21-MUL schedule, and the scalar
reference is not a customer-supplied target kernel.

Artifacts:
`experiments/results/dpd_timing_{dpa200,apa200}_validation.json`.
Both force `customer_gate_evaluable=false` and `hardware_pass_claim=false`.

### 3.4 Что legacy evidence доказывает

- complex phase-equivariant spline branches реализуют очень дешёвый DPD;
- signal delays 0,1,2 существенно улучшают тот legacy surrogate;
- desired \(x\) действительно использован как test input в этом evaluator;
- имеются numerical NMSE/EVM/ACLR/PAPR/peak-drive artifacts;
- continuous streaming state transfer exact for chunks 1, 8, 64 and 512 in
  the host reference;
- declared 16/14/12-bit integer arithmetic preserves the frozen
  validation-surrogate result with zero saturation/collision.

Он не доказывает:

- качество через новый GMP evaluator;
- ranking относительно OpenDPD на едином evaluator;
- устойчивость к evaluator mismatch;
- fixed-point behavior on an independent evaluator or physical PA;
- RTL/HLS resources, timing closure or a chosen reciprocal/sqrt datapath;
- target sample-rate throughput/latency или прохождение 1000-MUL-equivalent
  customer timing gate;
- physical-PA spectral mask или Huawei acceptance.

### 3.5 Why SPH is not a DPD result

The completed APA SPH run belongs to contour A:

```text
measured PA input x -> SPH forward model -> measured PA output y_hat
```

It was selected with train OOF and descriptive reused validation only. Its
`K=32,L=8` model has exact 37 MUL/36 ADD cost, but OOF NMSE is −30.402374 dB;
it loses to matched MP by 6.651955 dB and to GMP by 7.943037 dB. The result is
therefore a cheap negative control and residual-analysis evidence, not a frozen
PA evaluator for contour B. No DPD coefficients were tuned or tested through
SPH, and no new DPD claim is made.

### 3.6 Why the sparse PA result still does not open contour B

The completed non-factorized sparse spline-memory run also belongs to contour
A:

```text
measured PA input x -> sparse forward PA model -> measured PA output y_hat
```

Its selected topology has six causal branches
`(m,d)={(0,0),(1,1),(2,2),(22,22),(23,23),(24,24)}` with `K=12` knots per
branch. It reaches −32.030011 dB full train-OOF NMSE at an analytical cost of
54 real MUL and 58 real ADD per sample. This improves SPH by about 1.628 dB,
but loses to matched MP by 5.024 dB and GMP by 6.315 dB on the same OOF
protocol. The model is therefore classified
`neither_evaluator_nor_cheap_pareto`.

Validation was descriptive reused-validation only, test was not opened, and
no DPD was fitted through this model. Its strongest remaining causal proper
residual correlation occurs at lag 9 (`|rho|=0.690641` on train OOF), which
justifies one bounded preregistered lag-9 ablation but not DPD optimization.

### 3.7 Why the lag-9 sparse PA result is still not a DPD result

The separately preregistered lag-9 run selected
`(m,d)={(0,0),(1,1),(2,2),(22,22),(23,23),(24,24),(8,0),(9,0),(10,0)}`,
`K=12`, ridge `1e-8`. It reaches `−37.792478 dB` full OOF NMSE and
`−37.860728 dB` reused validation at 72 real MUL/82 real ADD. The incremental
parent gate passed by `+5.762467/+5.764583 dB` full/common, and the cheap-Pareto
gate beats matched MP by `0.738150/0.752881 dB`.

It remains `0.552932/0.897694 dB` behind matched GMP, has only same-capture
reused validation on source, and its independent B-capture held-out score after
calibration is `−34.801474 dB` full / `−35.437986 dB` common. It was never
evaluated on a predistorted waveform. Therefore `cheap_pareto_only` is a
contour-A label, not a frozen DPD evaluator; Gate A→B remains closed. The
immutable source bundle is
`experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/`.

The B-capture release is useful robustness evidence, not a DPD evaluator:
zero-shot GMP/sparse are `−23.795441/−23.695838 dB`, while calibrated GMP is
`−37.895152 dB`. The sparse model is about `13.25x` cheaper in real MUL and
`4.53x` faster to calibrate, but about `3.094 dB` worse than GMP on the
primary held-out score. The release audit records two accesses because the
first failed before inference/metric; no topology, `N` or coefficient choice
changed after that access.

## 4. Gate A→B и запрет silent reuse

Новая DPD optimization разрешается только после одновременного выполнения:

1. PA architecture/regularization frozen без DPD test tuning;
2. validation evaluator error power минимум на 10 dB ниже различаемого DPD
   residual — internal criterion, не извлечённое из Huawei slide требование;
3. ranking DPD подтверждён минимум двумя independently fitted evaluators либо
   physical PA;
4. predistorted amplitude лежит внутри verified evaluator support;
5. state reset, warm-up, causal latency и chunk semantics совпадают с
   deployment protocol.

Текущая arithmetic projection не является cascade measurement:

| Dataset | Existing DPD validation / test | GMP fidelity validation / test | Margin validation / test |
|---|---:|---:|---:|
| DPA | −30.532 / −29.864 dB | −35.366 / −35.385 dB | 4.834 / 5.521 dB |
| APA | −32.380 / −32.741 dB | −38.665 / −38.608 dB | 6.285 / 5.867 dB |

Decision: **closed**. GMP one-shot test release PASS и B-capture transfer
release не являются этим gate. Уже открытые DPA/APA/B test values нельзя
использовать для выбора нового DPD.

## 5. Frozen future benchmark matrix

После Gate A→B PASS один evaluator/config используется без изменений для:

1. no DPD;
2. MP DPD;
3. GMP DPD;
4. OpenDPD reference, если checkpoint воспроизводим;
5. memoryless complex linear spline;
6. spline signal delays `{0,1}`;
7. spline signal delays `{0,1,2}`;
8. SPH: complex spline followed by short complex FIR (already evaluated as a
   PA candidate; include in DPD matrix only after Gate A→B and independent
   evaluator criteria, not as the current evaluator);
9. one validation-frozen non-factorized sparse branch model, only if a future
   candidate passes contour-A gates; the completed first sparse model did not.

Architecture progression нельзя менять по test. Для deterministic closed-form
models stochastic seed не применяется; stochastic models используют минимум
`{0,1,2}`.

## 6. Required result schema

Для каждого dataset/evaluator/model сохраняются:

- customer-defined harmonic/spur attenuation после фиксации exact RF
  bands/reference/threshold;
- full-record и common-interior pooled complex NMSE;
- OpenDPD-compatible NMSE;
- sample-domain RMS EVM и, если waveform metadata достаточно, demodulated EVM;
- ACLR/ACPR left, right и average с frozen band definitions;
- output/error PSD с exact sample rate, window, overlap, FFT и normalization;
- AM/AM и AM/PM residuals;
- predistorted PAPR, average/maximum amplitude и support-violation fraction;
- stability/non-finite/saturation counts;
- real MUL/ADD, nonlinear, comparison, lookup, reads/writes;
- coefficients/constants/state bytes и numeric format;
- calibration wall time, inference batch time, streaming latency/throughput;
- fixed-point degradation at signed 16/14/12-bit formats;
- seed/capture/operating-point provenance.

Winner не выбирается одной метрикой. Публикуется Pareto frontier:

```text
quality vs operations/sample
quality vs measured 1000-real-MUL-equivalent DPD latency
quality vs calibration time
quality vs coefficient+state memory
quality vs peak predistorted drive
```

## 7. Следующий разрешённый шаг

Source DPA/APA и target-calibrated APA-B bit-accurate PA arithmetic уже
выполнена; target test не открывался. PA quantization expansion прекращается,
а следующий локальный step контура A — high-fidelity OpenDPD PA retraining
без DPD cost cap. Параллельные внешние prerequisites — exact harmonic/spur and
timing-reference definitions, metadata measurement B и controlled physical-PA
capture с известными power/backoff, bias и temperature axes.

DPD integer reference, timing instrumentation и frozen-host streaming
semantics теперь готовы, но hardware gate остаётся закрытым до target HLS/RTL,
target-specific reference-kernel measurement и physical-PA replay. Это не
открывает Gate A→B и не разрешает новое DPD tuning через старый surrogate.

Уже открытый B test нельзя использовать для нового выбора topology,
regularization или calibration N. Если controlled evidence не даёт второго
evaluator с достаточным margin, честный результат — остановить local PA
dictionary expansion и получить physical-PA measurement. Legacy DPD остаётся
surrogate-only; DPD optimization возобновляется только после Gate A→B.
