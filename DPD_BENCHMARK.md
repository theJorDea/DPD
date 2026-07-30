# DPD benchmark: evidence, protocol and release gate

Дата среза: 2026-07-30.

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
- нет второго independently fitted evaluator, нового operating-point capture
  или physical-PA output для predistorted waveform.

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

### 3.3 Что legacy evidence доказывает

- complex phase-equivariant spline branches реализуют очень дешёвый DPD;
- signal delays 0,1,2 существенно улучшают тот legacy surrogate;
- desired (x\) действительно использован как test input в этом evaluator;
- имеются numerical NMSE/EVM/ACLR/PAPR/peak-drive artifacts.

Он не доказывает:

- качество через новый GMP evaluator;
- ranking относительно OpenDPD на едином evaluator;
- устойчивость к evaluator mismatch;
- fixed-point degradation нового two-loop pipeline;
- physical-PA spectral mask или Huawei acceptance.

### 3.4 Why SPH is not a DPD result

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

### 3.5 Why the sparse PA result still does not open contour B

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

### 3.6 Why the lag-9 sparse PA result is still not a DPD result

The separately preregistered lag-9 run selected
`(m,d)={(0,0),(1,1),(2,2),(22,22),(23,23),(24,24),(8,0),(9,0),(10,0)}`,
`K=12`, ridge `1e-8`. It reaches `−37.792478 dB` full OOF NMSE and
`−37.860728 dB` reused validation at 72 real MUL/82 real ADD. The incremental
parent gate passed by `+5.762467/+5.764583 dB` full/common, and the cheap-Pareto
gate beats matched MP by `0.738150/0.752881 dB`.

It remains `0.552932/0.897694 dB` behind matched GMP, has only same-capture
reused validation, and was never evaluated on a predistorted waveform.
Therefore `cheap_pareto_only` is a contour-A label, not a frozen DPD evaluator;
Gate A→B remains closed. The immutable bundle is
`experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/`.

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

Decision: **closed**. GMP one-shot test release PASS не является этим gate.
Уже открытые DPA/APA test values нельзя использовать для выбора нового DPD.

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
quality vs calibration time
quality vs coefficient+state memory
quality vs peak predistorted drive
```

## 7. Следующий разрешённый шаг

Следующий step находится в контуре A, не B: проверить GMP и surviving lag-9
PA model на independent `APA_200MHz_b` capture (строго как `capture transfer`,
пока measurement-B metadata не расшифрованы) и построить limited-calibration
curves. Нужны source freeze, target-train-only nuisance alignment, zero-shot
режим и preregistered calibration sample counts; старый APA test не открывать
повторно для selection.

Если independent transfer не подтверждает ranking/fidelity, честный результат —
остановить local PA dictionary expansion и получить physical-PA measurement.
Legacy DPD остаётся surrogate-only; DPD optimization возобновляется только
после Gate A→B.
