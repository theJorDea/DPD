# Benchmark low-complexity PA models

Дата среза: 2026-07-29.

## Статус и границы доказательства

Этот документ относится к контуру A:

```text
measured PA input x
    -> frozen forward PA model
    -> predicted measured PA output y_hat
    -> compare y_hat with measured PA output y
```

В текущем срезе полностью выполнены два validation-selected baseline:
complex Memory Polynomial (MP) и causal factorized Generalized Memory
Polynomial (GMP) отдельно на `DPA_200MHz` и `APA_200MHz`.
Данные в CSV являются измеренными входом и выходом PA, поэтому это
**forward identification on held-out measured data**. Это не означает, что
физический PA был повторно измерен после построения модели.

Ни один результат из старого DPD-контура не является результатом forward PA
identification. Ни одна арифметическая оценка margin относительно нового
evaluator не является выполненным DPD experiment. Bundled OpenDPD benchmark
также приводится отдельным evidence layer и не выдаётся за локально
воспроизведённый результат.

Краткий вывод:

- budget-constrained GMP достиг на frozen test −35.3850 dB на DPA и
  −38.6081 dB на APA; это лучше локального MP на 0.2860 и 1.6176 dB;
- все четыре selected MP/GMP точки укладываются в строгое ограничение `<1000` real
  multiplications/complex sample по принятой software convention;
- если `10^-5` означает normalized error power, ни одна selected точка не достигает цели
  −50 dB;
- provisional 10 dB evaluator gate для продолжения DPD не выполнен;
- release-gates GMP прошли, но они разрешали только по одному frozen test и
  не являются Gate A→B;
- residual-selected APA conjugate-FIR audit не прошёл 0.1 dB OOF gate:
  лучший support дал лишь 0.0273/0.0305 dB full/common gain,
  поэтому frozen decision — `no_correction`;
- следующий proper long-FIR audit также не прошёл gate: лучший support
  `{44,45,46}` дал 0.0182/0.0201 dB full/common, и GMP снова не изменён;
- DPD optimization остаётся остановленной до следующего PA-model experiment,
  выбранного по residual evidence, либо до независимого physical-PA capture.

## 1. Определения метрик

Пусть

\[
e[n]=\hat y[n]-y[n],
\]

где \(y\) — measured PA output, а \(\hat y\) — предсказание frozen forward
model. После inference не разрешены дополнительный gain fit, delay fit или
phase alignment.

### 1.1 Primary pooled complex NMSE

\[
\operatorname{NMSE}_{pool}
=10\log_{10}
\frac{\sum_n|\hat y[n]-y[n]|^2}
     {\sum_n|y[n]|^2}.
\]

Это primary ranking metric. Реализация:
`baseline/metrics.py:92-103`; включение в PA evaluator:
`baseline/pa_benchmark.py:280-298`.

Линейный relative error power равен отношению под логарифмом. Если неизвестное
требование `error < 10^-5` означает именно это отношение, то acceptance
эквивалентен `pooled complex NMSE < -50 dB`. Слайды не устанавливают такую
интерпретацию однозначно; варианты SSE/MSE/normalized MSE разделены в
`REQUIREMENTS.md:72-103`.

### 1.2 OpenDPD-compatible NMSE

OpenDPD сначала считает NMSE каждого complete `nperseg` segment в dB, затем
усредняет dB:

\[
\operatorname{NMSE}_{OpenDPD}
=\frac1S\sum_s 10\log_{10}
\frac{\operatorname{mean}_{n\in s}|e[n]|^2}
     {\operatorname{mean}_{n\in s}|y[n]|^2}.
\]

Реализация совместимого варианта:
`baseline/metrics.py:106-131`,
`baseline/pa_benchmark.py:323-347`. Он в общем случае не равен pooled NMSE.
Для APA validation/test имеется один complete segment, поэтому значения
совпадают.

### 1.3 Steady-state score

Для честного сравнения моделей с разной causal memory дополнительно
публикуется pooled score после одинакового для всех candidates warm-up в
начале каждого evaluator frame. В исходном MP sweep использовано 29
samples/frame. В formal matched-support GMP decision до открытия test была
заморожена более строгая общая policy: warm-up 49, cooldown 0 samples/frame.
Она применяется одинаково к matched MP и GMP и не выбирается по test.
Реализация маски:
`baseline/pa_benchmark.py:301-320`.

Full-record NMSE остаётся primary: reset transient не скрывается. Steady-state
колонка показывает, насколько boundary policy влияет на результат.

### 1.4 MSE и sample-domain RMS EVM

\[
\operatorname{MSE}=\operatorname{mean}|e|^2,\qquad
\operatorname{EVM}_{RMS}
=\sqrt{\frac{\sum|e|^2}{\sum|y|^2}}.
\]

`20 log10(EVM_RMS)` численно совпадает с pooled NMSE в dB. Это
**sample-domain error**, а не demodulated constellation EVM
(`baseline/metrics.py:134-153`).

### 1.5 Error PSD и AM/AM–AM/PM residuals

Error PSD строится для \(e[n]\) periodic-Hann Welch с:

- `nfft = nperseg`;
- 50% overlap;
- constant detrend;
- density scaling;
- normalization на integrated measured-output power.

Точная реализация: `baseline/pa_benchmark.py:350-405`,
`baseline/metrics.py:178-259`.

AM/AM и AM/PM считаются в uniform-amplitude bins, границы которых заморожены
по maximum training amplitude. Публикуется разность predicted и measured
характеристик, а не две независимо нормированные картинки:
`baseline/pa_benchmark.py:408-448`.

## 2. Frozen evaluation protocol

Evaluator имеет единственное score-направление:

```text
x_split -> frozen PA model -> y_hat_split; compare with measured y_split
```

Это зафиксировано в `baseline/pa_benchmark.py:1-15,579-612`.

Общие правила:

1. Integer delay, fractional-delay diagnostic, complex-LS gain diagnostic,
   OpenDPD peak-gain diagnostic, maximum training amplitude и AM/AM–AM/PM bin
   edges вычисляются только на train
   (`baseline/pa_benchmark.py:130-234`).
2. Fractional delay записывается, но не применяется скрыто
   (`baseline/pa_benchmark.py:143-147`).
3. Gain является диагностикой. Prediction не подгоняется по gain после
   inference.
4. Состояние модели сбрасывается на каждой границе dataset `nperseg`;
   partial final frame оценивается отдельно.
5. Test нельзя маркировать как model-selection purpose
   (`baseline/pa_benchmark.py:498-527`).
6. Frozen-test runner сначала проверяет hashes модели, selection manifest,
   config, source и train/validation files; test CSV открываются только после
   этих проверок (`experiments/evaluate_frozen_pa.py:59-172`).
7. DPA и APA не смешиваются: это разные physical PA/captures.

| Protocol field | DPA_200MHz | APA_200MHz |
|---|---:|---:|
| Train / validation / test samples | 23,040 / 7,680 / 7,680 | 58,980 / 19,662 / 19,662 |
| Sample rate | 800 MHz | 983.04 MHz |
| `nperseg` | 2,560 | 19,662 |
| Frozen integer delay | 0 | 0 |
| Fractional-delay diagnostic | −0.00719 sample | +0.07726 sample |
| Fractional correction applied | no | no |
| Training complex-LS gain | 3.165638 − j3.5e−11 | 1.162566 − j0.003532 |
| Training OpenDPD peak gain | 2.520809 | 1.0 |
| Prediction gain refit | no | no |
| Frame state | reset each `nperseg` | reset each `nperseg` |

Frozen protocol evidence:

- окончательное решение A0/A1 для обоих datasets:
  `experiments/results/pa_alignment_protocol_decision.json`;
- DPA GMP selection:
  `experiments/results/pa_gmp_dpa200_selection/selection_manifest.json`;
- APA GMP selection:
  `experiments/results/pa_gmp_apa200_selection/selection_manifest.json`;
- split sizes/spec: раздел «Что установлено из OpenDPD» в
  `REQUIREMENTS.md`.

Fractional-delay A1 был проверен только как sensitivity transform. Улучшение
оказалось пренебрежимо малым и нестабильным, поэтому frozen protocol остался
A0: integer delay 0, fractional correction off. Это не является
feedback-path de-embedding.

## 3. Выполненный measured-data forward benchmark: MP

Модель:

\[
\hat y[n]=\sum_{d\in D}\sum_{p\in P}
a_{d,p}x[n-d]|x[n-d]|^{p-1},
\]

с одним complex coefficient на term. Fit выполняется complex ridge least
squares в `complex128`, causal history заполняется нулями отдельно в каждом
`nperseg` frame. Architecture и ridge выбраны только по validation pooled
NMSE. Test runner не выполняет refit и не меняет alignment/gain.

### 3.1 Результаты

| Dataset | Selected MP | Validation pooled / OpenDPD / steady (dB) | Test pooled / OpenDPD / steady (dB) | Selected fit |
|---|---|---:|---:|---:|
| DPA_200MHz | orders {1,3,5,7,9}, delays 0…23, ridge 1e−8, 120 complex coefficients | −34.9617 / −34.9286 / −35.0484 | −35.0990 / −35.1018 / −35.1607 | 0.918 s |
| APA_200MHz | orders {1,2,3,4,5}, delays 0…29, ridge 1e−9, 150 complex coefficients | −37.0952 / −37.0952 / −37.1272 | −36.9905 / −36.9905 / −37.0745 | 1.988 s |

Evidence:

- DPA selected architecture, fit and validation:
  `experiments/results/pa_mp_dpa200_selection/selection_manifest.json:230-320`;
- DPA frozen test and explicit `refit=false`, `post_prediction_gain_fit=false`,
  `post_prediction_delay_fit=false`:
  `experiments/results/pa_mp_dpa200_selection/test_manifest.json:4-22`;
- APA selected architecture, fit and validation:
  `experiments/results/pa_mp_apa200_selection/selection_manifest.json:245-338`;
- APA frozen test:
  `experiments/results/pa_mp_apa200_selection/test_manifest.json:4-22`.

`fit_seconds` — fit только финальной selected model. Это не wall-clock полного
sweep. Сумма отдельных fit timers для 46 trials равна 15.39 s на DPA и
43.23 s на APA; в неё не входят evaluation, serialization и process startup.
Host batch inference timing в JSON — диагностический throughput NumPy, не
hardware latency (`baseline/pa_benchmark.py:622-630`).

### 3.2 Соответствие возможному порогу `10^-5`

| Dataset | Test relative error power | Во сколько раз выше `10^-5` |
|---|---:|---:|
| DPA_200MHz | 3.0910e−4 | 30.91× |
| APA_200MHz | 1.9996e−4 | 20.00× |

Следовательно, если Huawei имеет в виду normalized error power, текущий MP
baseline цель не выполняет. Если имеется в виду MSE или SSE, вывод без
официального scaling/aggregation rule сделать нельзя.

### 3.3 Выполненный causal GMP benchmark

Formal GMP search был ограничен `<1000` counted real MUL/sample. Architecture
и ridge выбраны по validation full-record pooled NMSE; до test выполнен
coefficient-OOF residual audit и отдельный machine-readable release gate.
После PASS каждый frozen model был вызван на test ровно один раз. Никакого
refit, post-prediction gain/delay fit или retry не было.

| Dataset | Frozen topology | Complex coeff. | Validation full / common (dB) | Test full / OpenDPD / common (dB) | Final fit |
|---|---|---:|---:|---:|---:|
| DPA_200MHz | `both_k4_m1`: `ka7/la24`, `kb4/lb24/mb1`, causal `kc4/lc24/mc1`, ridge 1e−5 | 356 | −35.3659 / −35.4684 | −35.3850 / −35.3983 / −35.4192 | 1.234 s |
| APA_200MHz | `both_k2_m2`: `ka7/la30`, `kb2/lb30/mb2`, causal `kc2/lc30/mc2`, ridge 1e−7 | 444 | −38.6653 / −38.7346 | −38.6081 / −38.6081 / −38.7075 | 5.555 s |

Full-record — primary metric; common score использует frozen warm-up 49 и
cooldown 0. На test linear relative error power равен `2.893996e-4` для DPA
и `1.377808e-4` для APA, то есть соответственно 28.94× и 13.78× выше
возможного порога `10^-5`.

| Dataset | GMP minus MP improvement on validation | GMP minus MP improvement on test |
|---|---:|---:|
| DPA_200MHz | 0.4042 dB | 0.2860 dB |
| APA_200MHz | 1.5700 dB | 1.6176 dB |

Отрицательнее означает лучше; в таблице показана положительная величина
улучшения. DPA gain мал, APA gain материален, но это сравнение двух forward
PA identifiers, а не DPD result.

Evidence chain:

- frozen A0 decision:
  `experiments/results/pa_alignment_protocol_decision.json`;
- selection manifests:
  `experiments/results/pa_gmp_dpa200_selection/selection_manifest.json` и
  `experiments/results/pa_gmp_apa200_selection/selection_manifest.json`;
- OOF/validation residual manifests:
  `experiments/results/pa_gmp_dpa200_residuals/residual_manifest.json` и
  `experiments/results/pa_gmp_apa200_residuals/residual_manifest.json`;
- release decisions:
  `experiments/results/pa_gmp_dpa200_residuals/test_release_gate.json` и
  `experiments/results/pa_gmp_apa200_residuals/test_release_gate.json`;
- one-shot frozen-test manifests:
  `experiments/results/pa_gmp_dpa200_test/test_manifest.json` и
  `experiments/results/pa_gmp_apa200_test/test_manifest.json`.

Здесь `sealed` означает только workflow-specific seal: исторический MP
workflow уже обращался к тому же dataset test split. GMP selection, residual
audit и release decision его не открывали; затем test был использован один
раз для frozen GMP. Это не globally pristine benchmark.

## 4. Complexity и исправленная память MP

Принята convention:

```text
1 complex multiply = 4 real MUL + 2 real ADD
FMA = 1 MUL + 1 ADD
sqrt/nonlinear, comparison, lookup и memory traffic считаются отдельно
```

| Dataset | Real MUL | Real ADD | sqrt | Compare | Real reads / writes | Real coefficients | Real constants | State reals | Corrected FP32 total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DPA_200MHz | 792 | 502 | 0 | 24 | 288 / 2 | 240 | 29 | 46 | 1,260 B |
| APA_200MHz | 960 | 628 | 30 | 30 | 360 / 2 | 300 | 35 | 58 | 1,572 B |

Counts относятся к одному complex sample. DPA:

```text
33 MUL per delay * 24 delays = 792 MUL
24 * 11 local ADD + 2 * (120 - 1) accumulation ADD = 502 ADD
state = 23 previous complex inputs = 46 real values
```

APA:

```text
32 MUL per delay * 30 delays = 960 MUL
30 * 11 local ADD + 2 * (150 - 1) accumulation ADD = 628 ADD
30 shared sqrt, one per delayed amplitude
state = 29 previous complex inputs = 58 real values
```

Counter и storage formula:

- `baseline/complexity.py:201-285`;
- `baseline/pa_benchmark.py:451-495`;
- state regression: `tests/test_complexity_count.py:77-89`.

Corrected bytes:

```text
DPA: 240*4 coefficient + 29*4 constant + 46*4 state = 1260 B
APA: 300*4 coefficient + 35*4 constant + 58*4 state = 1572 B
```

### Stale artifact warning

Selection/test JSON были созданы до исправления delay-line state bookkeeping в
commit `c5e0e90`. Поэтому внутри них всё ещё записаны
`state_real_values=0` и totals 1,076/1,340 B. Примеры:

- DPA selection manifest:
  `experiments/results/pa_mp_dpa200_selection/selection_manifest.json:273-280`;
- APA selection manifest:
  `experiments/results/pa_mp_apa200_selection/selection_manifest.json:291-298`.

NMSE, coefficients и arithmetic counts от этой bookkeeping correction не
изменились. Для отчёта выше память пересчитана текущим counter. Перед
машинным объединением Pareto tables artifacts следует регенерировать, чтобы
их embedded source hashes и memory fields снова соответствовали коду.

Counts являются analytical schedule, не FPGA measurement. Dynamic delay-buffer
traffic, address generation, bandwidth, pipeline latency, DSP packing и power
не измерены. Текущий MP predictor строит term dictionary; более глубокая
factorization может изменить MUL/ADD trade-off и должна считаться отдельным
kernel variant, а не бесплатным исправлением существующего результата.

### 4.1 Causal factorized GMP complexity

GMP inference использует schedule

\[
\hat y[n]=\sum_q x[n-q]h_q(\text{delayed envelope streams}),
\]

а не runtime multiplication плотной design matrix на coefficient vector.
Analytical counts в frozen manifests:

| Dataset | Real MUL | Real ADD | Nonlinear | Reads / writes | Real coeff. | Constants | State reals | FP32 coeff+const+state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DPA_200MHz | 766 | 759 | 1 | 1,092 / 8 | 712 | 9 | 188 | 3,636 B |
| APA_200MHz | 954 | 947 | 1 | 1,362 / 8 | 888 | 9 | 236 | 4,532 B |

`Nonlinear=1` — вычисление общей amplitude/envelope primitive в принятой
абстракции counter; это не означает один DSP cycle. Оба kernel укладываются в
строгое `<1000` по real multiplications/sample, но APA оставляет запас только
46 MUL. Coefficient/state memory существенно выше локального MP, поэтому GMP
не доминирует MP по всем Pareto axes.

Operation records:

- DPA: `experiments/results/pa_gmp_dpa200_test/test_manifest.json`;
- APA: `experiments/results/pa_gmp_apa200_test/test_manifest.json`;
- formula и schedule: `baseline/complexity.py`, `baseline/gmp_pa.py`.

Full inference, reset-per-frame и arbitrarily chunked causal streaming дали
максимальную абсолютную ошибку 0 при tolerance `1e-12` в обоих release gates.
Это доказывает software streaming equivalence, но не measured hardware
latency и не bit-accurate fixed-point equivalence.

## 5. Residual evidence после MP и GMP

Residual discovery использует leave-one-explicit-frame-out coefficient fits
на train; validation служит confirmation/diagnostic. Сегменты — `nperseg`
model-reset frames, а не доказанные независимые physical captures. В
artifacts знак задан как \(e_{res}[n]=y[n]-\hat y[n]\), поэтому знак signed
correlations следует читать по этой convention.

Formal GMP residual audit выполнен до frozen test и на том же common support,
что matched MP reference:

| Dataset | GMP train OOF full / common | GMP validation full / common | GMP OOF gain над matched MP full / common |
|---|---:|---:|---:|
| DPA_200MHz | −35.3157 / −35.4224 | −35.3659 / −35.4684 | 0.2952 / 0.3009 dB |
| APA_200MHz | −38.3454 / −38.7505 | −38.6653 / −38.7346 | 1.2911 / 1.6506 dB |

Release predicates прошли для всех folds:

- DPA: 9/9 full rank, maximum held/fit amplitude ratio ≈1.000000001,
  full-vs-common gap 0.1068 dB OOF;
- APA: 3/3 full rank, maximum held/fit amplitude ratio ≈1.0003,
  full-vs-common gap 0.4051 dB OOF;
- maximum fold condition/norm ratios остались внутри preregistered limits;
- никакой test metric не участвовал в этих решениях.

Интерпретация residual evidence:

- GMP cross-envelope branches дали reproducible OOF gain, особенно на APA;
- DPA gain около 0.3 dB недостаточен, чтобы оправдать крупное дальнейшее
  расширение того же словаря без нового residual-selected hypothesis;
- APA common OOF лучше full OOF на 0.405 dB, поэтому reset/boundary semantics
  остаются существенной частью ошибки;
- high-amplitude и lag/slow-envelope diagnostics сохранены в artifacts, но
  они не являются независимым validation после использования тех же records;
- slow-state branch остаётся заблокирован: `independent_capture_count=0`.

Полные GMP результаты:

- `experiments/results/pa_gmp_dpa200_residuals/`;
- `experiments/results/pa_gmp_apa200_residuals/`.

Исторические MP residual artifacts сохранены в
`experiments/results/pa_mp_dpa200_residuals/` и
`experiments/results/pa_mp_apa200_residuals/`; они служат matched baseline,
но уже не определяют следующий model family в одиночку.

Validation residual нельзя использовать для выбора новой branch и затем
повторно назвать тот же validation независимым подтверждением. Следующий
architecture hypothesis должен быть preregistered по train OOF либо проверен
на новом capture/operating point.

## 6. Старый surrogate-only DPD evidence

Старый DPD evaluator — MP с orders `{1,3,5,7,9}`, delays `0…4`, 25 complex
coefficients и ridge `1e-8`. Он обучался на measured train pair, но DPD
оценивался только через этот surrogate; predistorted waveform не подавался
повторно на physical PA.

| Dataset | Old surrogate test fidelity | Spline delays 0,1,2 DPD test NMSE | Paired fidelity margin |
|---|---:|---:|---:|
| DPA_200MHz | −30.1305 dB | −29.8645 dB | +0.2660 dB |
| APA_200MHz | −31.0908 dB | −32.7408 dB | −1.6500 dB |

Evidence:

- DPA surrogate configuration:
  `experiments/results/spline_memory_dpa200/memory_ablation_report.json:72-118`;
- DPA paired test values and `scope=surrogate_only`:
  `experiments/results/spline_memory_dpa200/memory_ablation_report.json:884-915`;
- APA configuration:
  `experiments/results/spline_memory_apa200/memory_ablation_report.json:78-124`;
- APA paired test values:
  `experiments/results/spline_memory_apa200/memory_ablation_report.json:890-921`.

Margin определён как

\[
M = NMSE_{DPD\ residual}-NMSE_{PA\ model}.
\]

Положительные 10 dB означают, что evaluator error power в десять раз меньше
DPD residual. Старый evaluator не проходит gate; на APA DPD residual даже
формально ниже собственного forward error evaluator. Это не доказывает
физическую линеаризацию и является причиной остановки дальнейшей DPD
оптимизации.

## 7. Projection для нового evaluator — не experiment

Existing spline DPD ещё не был прогнан через новый frozen GMP evaluator. Поэтому
следующая таблица — только subtraction уже существующих чисел из разных runs.
Она не является cascade measurement и не должна попадать в итоговую Pareto
frontier как результат.

| Dataset / split | Existing surrogate-only DPD NMSE | Frozen GMP PA NMSE | Projected margin | Shortfall до 10 dB |
|---|---:|---:|---:|---:|
| DPA validation | −30.5324 | −35.3659 | 4.8336 | 5.1664 |
| DPA test | −29.8645 | −35.3850 | 5.5206 | 4.4794 |
| APA validation | −32.3800 | −38.6653 | 6.2852 | 3.7148 |
| APA test | −32.7408 | −38.6081 | 5.8673 | 4.1327 |

При неизменном DPD validation residual потребовалась бы PA fidelity не хуже:

- DPA: −40.5324 dB;
- APA: −42.3800 dB.

Gate A→B, определённый в соответствующих разделах `ROADMAP.md` и
`REQUIREMENTS.md`, остаётся закрытым. GMP release-gate PASS означал только
разрешение one-shot test и не меняет этого решения. Даже после будущего true
cross-evaluator run одно совпадение ranking
недостаточно: нужны два independently fitted evaluators или physical-PA
remeasurement.

## 8. Bundled OpenDPD APA reference — не локальный rerun

Ниже приведён bundled repository report на тех же APA CSV hashes. Эти модели
не были прогнаны через наш evaluator в текущем проекте.

| PA model | Configuration / method | Reported parameters | NMSE validation / test (dB) | OpenDPD spectral EVM validation / test (dB) |
|---|---|---:|---:|---:|
| MP | K=9, Q=150; direct LS | 2,700 real | −37.0405 / −36.9635 | −40.2302 / −39.9571 |
| GMP | 5/30 aligned; 4/30/5 lagging and leading; truncated SVD rank 650/1,350, rcond 1e−4 | 2,700 real | −38.7020 / −38.6606 | −43.0540 / −42.7971 |
| GRU-H28 | AdamW, selected epoch 292 | 2,746 | −38.8504 / −38.9367 | −43.6317 / −43.5732 |
| TRes-GRU-H27 | AdamW, selected epoch 288 | 2,751 | −39.0447 / −39.1293 | −43.9118 / −43.9299 |
| TRes-DeltaGRU-H27 | THX=THH=0, epoch 294 | 2,751 | −39.0929 / −39.1777 | −44.0254 / −43.9683 |

Sources:

- report table: `vendor/OpenDPD/benchmark/benchmark_report.md:21-31`;
- metric definitions:
  `vendor/OpenDPD/benchmark/benchmark_report.md:81-87`;
- configurations and context:
  `vendor/OpenDPD/benchmark/benchmark_report.md:89-100`;
- methodology:
  `vendor/OpenDPD/benchmark/benchmark_report.md:102-112`;
- limitations:
  `vendor/OpenDPD/benchmark/benchmark_report.md:114-123`;
- machine evidence:
  - GMP `vendor/OpenDPD/benchmark/results/benchmark_report_results.json:439-505`;
  - GRU
    `vendor/OpenDPD/benchmark/results/benchmark_report_results.json:507-558`;
  - MP
    `vendor/OpenDPD/benchmark/results/benchmark_report_results.json:560-620`;
  - TRes-DeltaGRU
    `vendor/OpenDPD/benchmark/results/benchmark_report_results.json:622-681`;
  - TRes-GRU
    `vendor/OpenDPD/benchmark/results/benchmark_report_results.json:683-734`;
  - exact APA CSV hashes
    `vendor/OpenDPD/benchmark/results/benchmark_report_results.json:743-750`.

Наш causal APA GMP имеет validation/test −38.6653/−38.6081 dB при 444 complex
coefficients, 954 real MUL/sample и zero lookahead. Bundled GMP сообщает
−38.7020/−38.6606 dB при 1,350 complex coefficients: разница всего
0.0367/0.0525 dB в пользу bundled result. Это не доказательство equivalence
или superiority, потому что boundary semantics, solver и arithmetic schedule
различаются. Наш локальный результат доказывает компактную causal точку, а не
apples-to-apples победу.

Наш APA MP остаётся на 0.055 dB лучше bundled MP на validation и на 0.027 dB
лучше на test при 150 вместо 1,350 complex coefficients; столь малая разница
также не является статистически значимым quality claim.

Bundled GMP и neural rows не удовлетворяют текущему apples-to-apples gate:

- OpenDPD сравнивает stored parameter count, не operations/sample;
- GMP имеет future envelope dependencies и offline boundary semantics
  (`vendor/OpenDPD/benchmark/benchmark_report.md:96-100`);
- один seed, нет fixed-point PA degradation и hardware latency
  (`vendor/OpenDPD/benchmark/benchmark_report.md:114-123`);
- report создан на dirty `benchmark-fix` commit `3df35e…`, а vendored main
  имеет commit `7426bbf…`; exact hashes уменьшают, но не устраняют provenance
  caveat (`vendor/OpenDPD/benchmark/benchmark_report.md:125-138`).

### Checkpoint availability

MP/GMP являются closed-form fits и могут быть переобучены. Neural JSON хранит
expected archived paths, sizes и SHA-256:

- GRU:
  `vendor/OpenDPD/benchmark/results/benchmark_report_results.json:515-520`;
- TRes-DeltaGRU:
  `vendor/OpenDPD/benchmark/results/benchmark_report_results.json:630-635`;
- TRes-GRU:
  `vendor/OpenDPD/benchmark/results/benchmark_report_results.json:691-695`.

Но binaries не входят в current Git tree. Проверка:

```bash
git -C vendor/OpenDPD ls-tree -r --name-only \
  7426bbf8a47624b59bd7f045a86641b403023f3c |
rg '\.(pt|pth|ckpt|onnx)$|(^|/)(save|log)/'
```

не возвращает файлов. Поэтому neural rows — bundled numeric references, а не
доступный frozen evaluator. Для локального comparison нужен исходный evidence
archive либо воспроизводимое retraining с новым checkpoint hash.

## 9. Provisional evaluator gate

DPD stage возобновляется только если:

1. PA architecture/regularization выбраны на train/validation и frozen до
   test;
2. validation PA error power минимум на 10 dB ниже DPD residual, который
   требуется различать;
3. ranking DPD сохраняется минимум на двух independently fitted evaluators;
4. predistorted drive находится внутри verified PA-model support;
5. causality, state и chunk-boundary semantics соответствуют deployment.

10 dB — внутренний conservative research criterion, не извлечённое из слайдов
требование Huawei. Он защищает от optimizer/evaluator coupling, но не заменяет
physical PA test. Текущий status: **gate closed**.

Для APA даже bundled TRes-DeltaGRU PA fidelity даёт только арифметический
projected margin 6.71 dB на validation и 6.44 dB на test относительно
existing spline result. Это также projection: соответствующего checkpoint и
cross-evaluator run нет.

## 10. Formal GMP experiment: завершённый protocol audit

### 10.1 Выполненная последовательность

1. Train/validation-only A0/A1 sensitivity завершилась выбором A0 для обоих
   datasets; решение frozen до formal selection.
2. Formal bounded search выполнил по 154 fits на DPA и APA. Candidates с
   `real_multiplications >= 1000` исключались до fit.
3. Architecture выбиралась по primary validation full-record NMSE; ridge
   refinement выполнялся только после architecture freeze.
4. Выбранные models были refitted только на train, сериализованы и проверены
   по SHA-256.
5. Coefficient-OOF residual audit сравнил GMP с matched-support MP без test.
6. Separate release gate проверил OOF gain, validation gap, rank/conditioning,
   support, diagnostic completeness, operation budget и streaming/reset
   equivalence.
7. После PASS каждый test был открыт ровно один раз; test runner не выполнял
   refit или post-hoc alignment.

Exact commands и configs:

```bash
.venv/bin/python -m experiments.select_pa_gmp \
  --config experiments/configs/pa_gmp_dpa200.json
.venv/bin/python -m experiments.select_pa_gmp \
  --config experiments/configs/pa_gmp_apa200.json

.venv/bin/python -m experiments.analyze_pa_residuals \
  --config experiments/configs/pa_gmp_residual_dpa200.json
.venv/bin/python -m experiments.analyze_pa_residuals \
  --config experiments/configs/pa_gmp_residual_apa200.json

.venv/bin/python -m experiments.decide_gmp_test_release \
  --config experiments/configs/pa_gmp_residual_dpa200.json
.venv/bin/python -m experiments.decide_gmp_test_release \
  --config experiments/configs/pa_gmp_residual_apa200.json

.venv/bin/python -m experiments.evaluate_frozen_pa \
  --selection-manifest \
    experiments/results/pa_gmp_dpa200_selection/selection_manifest.json \
  --output-dir experiments/results/pa_gmp_dpa200_test
.venv/bin/python -m experiments.evaluate_frozen_pa \
  --selection-manifest \
    experiments/results/pa_gmp_apa200_selection/selection_manifest.json \
  --output-dir experiments/results/pa_gmp_apa200_test
```

Formal selection wall time: 70.99 s DPA и 212.76 s APA. Final selected fit
занял 1.234 s и 5.555 s; residual audit — 10.26 s и 24.87 s. Peak RSS не
измерялся. Frozen-test process wall measurements были 0.066 s DPA и 0.31 s
APA, но это host batch runtime, не deployment latency.

### 10.2 Что release gate доказал и чего не доказал

Оба `test_release_gate.json` дали PASS всех hard predicates и
`may_open_gmp_test_once=true`. Это доказало корректность **процесса открытия
test** для уже frozen GMP. Оно не доказало:

- Gate A→B;
- DPD quality или DPD ranking;
- fixed-point readiness;
- generalization между PA/power levels/waveforms;
- physical-PA linearization;
- соответствие неизвестному Huawei acceptance protocol.

После one-shot test разрешение израсходовано. Повторная настройка GMP по этим
test values запрещена; дальнейшие architecture decisions должны использовать
только train OOF/validation либо новые captures.

### 10.3 Следующий PA-model decision

GMP становится текущим наиболее точным локальным frozen evaluator для каждого
dataset, но evaluator margin остаётся лишь 4.8–6.3 dB на validation projection
и 5.5–5.9 dB на test projection. Поэтому Gate A→B остаётся закрытым.

Следующий model family нельзя выбирать простым расширением GMP grid после
просмотра test. Максимально информативная последовательность:

1. [x] preregister и выполнить bounded APA widely-linear/IQ audit;
2. [x] проверить nested proper long-FIR ablation около residual lags 42…49;
3. [ ] после двух negative linear corrections preregister standalone
   `spline/CPWL memoryless nonlinearity + short complex FIR` PA;
4. [ ] только затем проверять sparse complex spline-memory PA,
   если первая nonlinear family не даст достаточного gain;
5. сравнивать по train resampling/reused validation при `<1000` MUL, не
   выдавая уже просмотренные splits за independent confirmation;
6. state-conditioned branch не запускать без independent long captures;
7. новый test claim делать на новом capture/operating point, а не повторно
   использовать уже открытый DPA/APA test как selection feedback.

### 10.4 APA widely-linear residual audit

По GMP residual reports была preregistered модель

\[
\hat y_{WL}[n]=\hat y_{GMP}[n]+\sum_{d\in D}b_d x^*[n-d],
\]

где supports ограничены `{0}`, `{0,1}`, `{0,1,2}` и `{0,1,2,3,4}`.
Каждый fold переобучал только coefficients frozen GMP recipe на
fit frames, затем conjugate coefficients обучались на residual тех же
fit frames. Test не читался и не хэшировался.

| APA candidate | MUL/sample | OOF gain full, dB | OOF gain common, dB | Minimum fold gain full/common, dB | Eligible |
|---|---:|---:|---:|---:|---:|
| `conj_d0` | 958 | 0.0268 | 0.0298 | 0.0241 / 0.0267 | no |
| `conj_d0_d1` | 962 | **0.0273** | **0.0305** | 0.0239 / 0.0269 | no |
| `conj_d0_d2` | 966 | 0.0268 | 0.0299 | 0.0245 / 0.0270 | no |
| `conj_d0_d4` | 974 | 0.0248 | 0.0296 | 0.0184 / 0.0269 | no |

Все correction fits full rank, а streaming/reset checks exact. Но every support
остался ниже preregistered 0.1 dB full/common threshold. Поэтому selected
candidate — `no_correction`, operation point остался 954 MUL / 947 ADD,
а reused validation не участвовал в selection. Wall time — 14.80 s;
OOF fit time — 13.22 s.

Это опровергает сильную версию гипотезы «заметная
pseudo-correlation должна дать практически значимый cheap linear
conjugate correction». Сам diagnostic корректен, но normalized
correlation не равна explained error power и не заменяет OOF ablation.
Физическая PA/IQ attribution по этому result не заявляется.

Evidence:

- `experiments/configs/pa_widely_linear_residual_apa200.json`;
- `experiments/results/pa_widely_linear_residual_apa200/audit_manifest.json`;
- `experiments/results/pa_widely_linear_residual_apa200/`.

### 10.5 APA proper long-FIR residual audit

Согласованный train-OOF/reused-validation proper-correlation peak около
causal lag 45 был проверен моделью

\[
\hat y_{FIR}[n]=\hat y_{GMP}[n]+\sum_{d\in D}b_d x[n-d].
\]

| APA candidate | MUL/ADD/state | OOF gain full/common, dB | Minimum fold full/common, dB | Eligible |
|---|---:|---:|---:|---:|
| `proper_d45` | 958/951/268 | 0.0133/0.0147 | 0.0083/0.0093 | no |
| `proper_d44_d46` | 966/959/270 | **0.0182/0.0201** | 0.0114/0.0127 | no |
| `proper_d43_d48` | 978/971/274 | 0.0177/0.0199 | 0.0115/0.0144 | no |
| `proper_d42_d49` | 986/979/276 | 0.0177/0.0201 | 0.0114/0.0144 | no |

Все folds улучшились, fits были full rank, а reset/streaming checks exact.
Однако лучший aggregate gain в 5.5 раза меньше 0.1 dB threshold, поэтому
selected candidate — `no_correction`; APA Pareto point остался 954 MUL / 947
ADD / 888 real coefficients / 236 state reals. Validation не участвовал в
selection, test не читался и не хэшировался. Wall time — 25.47 s; OOF fit
time — 23.40 s.

Этот результат отделяет наличие correlation от её практической ценности:
long-memory linear component воспроизводим по folds, но слишком мал для
решения evaluator bottleneck. Поэтому следующий candidate должен быть
standalone nonlinear alternative, а не ещё одна correction поверх GMP.

Evidence:

- `experiments/configs/pa_long_fir_residual_apa200.json`;
- `experiments/results/pa_long_fir_residual_apa200/audit_manifest.json`;
- `experiments/results/pa_long_fir_residual_apa200/`.

## 11. Что пока неизвестно или не выполнено

- Не установлено официальное определение Huawei `error < 10^-5`.
- Не установлено, означает ли `<1000 multipliers` operations/sample,
  физических DSP blocks или amortized update cost.
- Нет официальных DUT, carrier/power/backoff, waveform masks, feedback-path
  calibration и verification splits Huawei.
- A0/A1 fractional-delay sensitivity выполнена, но independent feedback-path
  calibration/de-embedding всё ещё отсутствует; A0 frozen без correction.
- Нет independent long captures для thermal/trapping state.
- Нет captures разных power levels/operating points для adaptation curves.
- Нет runnable bundled OpenDPD neural checkpoint.
- Нет locally rerun OpenDPD PA backbone в нашем frozen evaluator.
- Widely-linear и proper long-FIR APA residual branches проверены и отклонены
  по preregistered threshold; всё ещё нет standalone spline/CPWL + FIR или
  sparse spline-memory PA result.
- Нет bit-accurate 16/14/12-bit PA-model evaluation.
- Нет measured latency/throughput на FPGA/ASIC/DSP target.
- Нет physical-PA remeasurement с predistorted waveform.

Поэтому текущая корректная формулировка результата:

> Validation-selected causal factorized GMP воспроизводимо моделирует
> held-out measured DPA/APA captures на −35.385/−38.608 dB full-record pooled
> NMSE при 766/954 counted real multiplications/sample. Он улучшает локальный
> MP, особенно на APA, но не достигает возможной −50 dB цели и не обеспечивает
> 10 dB evaluator margin для текущего surrogate-only DPD residual. Release
> gates и one-shot tests завершены; Gate A→B остаётся закрытым, поэтому DPD
> optimization всё ещё приостановлена до более независимого и/или точного PA
> evidence.
