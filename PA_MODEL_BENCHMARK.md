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

В текущем срезе полностью выполнен один validation-selected baseline:
complex Memory Polynomial (MP) отдельно на `DPA_200MHz` и `APA_200MHz`.
Данные в CSV являются измеренными входом и выходом PA, поэтому это
**forward identification on held-out measured data**. Это не означает, что
физический PA был повторно измерен после построения модели.

Ни один результат из старого DPD-контура не является результатом forward PA
identification. Ни одна арифметическая оценка margin относительно нового
evaluator не является выполненным DPD experiment. Bundled OpenDPD benchmark
также приводится отдельным evidence layer и не выдаётся за локально
воспроизведённый результат.

Краткий вывод:

- новый budget-constrained MP улучшил fidelity старого surrogate примерно с
  −30…−31 dB до −35.10 dB на DPA и −36.99 dB на APA;
- обе модели укладываются в строгое ограничение `<1000` real
  multiplications/complex sample по принятой software convention;
- если `10^-5` означает normalized error power, обе модели не достигают цели
  −50 dB;
- provisional 10 dB evaluator gate для продолжения DPD не выполнен;
- следующий model experiment — causal factorized GMP, выбранный только по
  train/validation; test должен оставаться закрытым до freeze.

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
начале каждого evaluator frame. В MP sweep использовано 29 samples/frame:
это maximum causal warm-up preregistered search space, а не индивидуально
выбранная выгодная обрезка. Реализация маски:
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

- DPA: `experiments/results/pa_mp_dpa200_selection/selection_manifest.json:43-97`;
- APA: `experiments/results/pa_mp_apa200_selection/selection_manifest.json:49-103`;
- split sizes/spec: `REQUIREMENTS.md:130-156`.

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

## 5. Residual evidence после MP

Residual discovery использует leave-one-explicit-frame-out predictions на
train; validation служит confirmation/diagnostic. Test не открывался.
Сегменты — `nperseg` model-reset frames, а не доказанные независимые physical
captures.

В residual-analysis artifacts знак задан как
\(e_{res}[n]=y[n]-\hat y[n]\), то есть противоположно обозначению в разделе
метрик. NMSE/PSD от этого не меняются, но знак signed correlations и bias
следует читать именно по artifact definition.

Основные наблюдения:

- DPA train-OOF и validation steady NMSE: −35.1141/−35.0484 dB.
  Train-OOF proper residual/input correlations малы и максимальны около
  lags 36–42 (около 0.03), тогда как validation имеет более сильные
  correlations на lags 0,1,2 (0.164, 0.149, 0.106). Pattern не стабилен между
  discovery и confirmation, поэтому он не оправдывает крупную
  state-conditioned модель.
- APA train-OOF и validation steady NMSE: −37.0991/−37.1272 dB.
  Proper correlations около lags 44–47 воспроизводятся на обоих splits
  (примерно 0.045–0.054), хотя они остаются умеренными. Это даёт evidence
  сначала проверить longer causal memory и только затем nonlinear
  cross-envelope terms.
- High-amplitude q90/q95 regions не хуже complement; DPA q99 лишь примерно
  на 0.55 dB хуже на validation. Compression-only branch не является первым
  выбором по текущему residual.
- Reset region имеет заметно худший NMSE, но содержит только boundary samples.
  Он маркирован как model reset transient, а не автоматически как physical
  memory:
  `experiments/results/pa_mp_dpa200_residuals/residual_manifest.json:261-305`,
  `experiments/results/pa_mp_apa200_residuals/residual_manifest.json:99-143`.
- Slow-state branch заблокирован: `independent_capture_count=0`; текущие
  короткие records не могут доказать thermal/bias state
  (`experiments/results/pa_mp_dpa200_residuals/residual_manifest.json:354-356`,
  `experiments/results/pa_mp_apa200_residuals/residual_manifest.json:198-200`).

Полные machine-readable результаты:

- `experiments/results/pa_mp_dpa200_residuals/residual_manifest.json`;
- `experiments/results/pa_mp_dpa200_residuals/train_oof_residual_analysis.json`;
- `experiments/results/pa_mp_dpa200_residuals/validation_residual_analysis.json`;
- соответствующие APA artifacts в
  `experiments/results/pa_mp_apa200_residuals/`.

Validation residual нельзя использовать для выбора новой branch и затем
повторно назвать тот же validation независимым подтверждением. Этот guard
записан в residual manifests как `future_branch_selection_rule`.

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

Existing spline DPD ещё не был прогнан через новый frozen MP evaluator. Поэтому
следующая таблица — только subtraction уже существующих чисел из разных runs.
Она не является cascade measurement и не должна попадать в итоговую Pareto
frontier как результат.

| Dataset / split | Existing DPD NMSE | New MP PA NMSE | Projected margin | Shortfall до 10 dB |
|---|---:|---:|---:|---:|
| DPA validation | −30.5324 | −34.9617 | 4.4294 | 5.5706 |
| DPA test | −29.8645 | −35.0990 | 5.2345 | 4.7655 |
| APA validation | −32.3800 | −37.0952 | 4.7152 | 5.2848 |
| APA test | −32.7408 | −36.9905 | 4.2497 | 5.7503 |

При неизменном DPD validation residual потребовалась бы PA fidelity не хуже:

- DPA: −40.5324 dB;
- APA: −42.3800 dB.

Gate из `ROADMAP.md:92-107` и `REQUIREMENTS.md:269-278` остаётся закрытым.
Даже после будущего true cross-evaluator run одно совпадение ranking
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

Наш APA MP на 0.055 dB лучше bundled MP на validation и на 0.027 dB лучше на
test при 150 вместо 1,350 complex coefficients. Разница слишком мала, чтобы
заявлять устойчивое quality superiority без единого evaluator rerun. Уже
доказан более компактный coefficient set; не доказано статистически значимое
превосходство NMSE.

Bundled GMP и neural rows точнее нашего MP на APA, но не удовлетворяют
текущему apples-to-apples gate:

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

## 10. Следующий эксперимент: causal factorized GMP PA

### 10.1 Что уже реализовано

Готов, но ещё не benchmarked:

- exact aligned/lagging/leading complex GMP dictionary и явная causality
  policy: `baseline/gmp_pa.py:1-20,45-156`;
- dense calibration matrix, reset по frames:
  `baseline/gmp_pa.py:173-211`;
- factorized inference:
  `baseline/gmp_pa.py:285-348`;
- causal streaming state/chunk path:
  `baseline/gmp_pa.py:352-406`;
- column-scaled complex ridge и rank-controlled truncated-SVD fit без normal
  equations: `baseline/gmp_pa.py:477-585`;
- factorized operation/state counter:
  `baseline/complexity.py:290-437`;
- tests, включая exact basis/inference/fit/streaming/counts:
  `tests/test_gmp_pa.py`.

Результатов GMP пока нет. Отсутствуют ожидаемый
`experiments/select_pa_gmp.py`, frozen configs, selection manifests и test
artifacts. Следовательно, ни одно GMP число не должно появляться в разделе
completed results.

### 10.2 Pre-registered порядок

1. До formal sweep провести отдельную train-frozen fractional-delay
   sensitivity A0/A1: current integer-only protocol против versioned
   frame-safe band-limited transform. Это не считается de-embedding без
   independent feedback/loopback calibration; test остаётся закрытым.
2. Добавить selector/result schema, не меняя evaluator внутри GMP sweep.
3. До sweep зафиксировать solver axis, уже поддерживаемый fit API:
   `ridge_lstsq` либо `truncated_svd` с явным `svd_rcond`; не выбирать solver
   по test. Unit tests rank truncation:
   `tests/test_gmp_pa.py:215-260`.
4. Проверить aligned-only sanity candidate:
   - APA `ka=5, la=30` должен представлять ту же consecutive-power family, что
     selected APA MP, с учётом другого coefficient order/kernel;
   - prediction equivalence проверяется unit test до quality sweep.
5. Candidate structure определить по train OOF, не по test и не повторно по
   уже просмотренному validation residual.
6. Architecture stage: один fixed regularizer; затем regularization refinement
   только для selected architecture.
7. Любой candidate с `real_multiplications >= 1000` исключать до fit.
8. Все candidates сравнивать при общем maximum warm-up; full-record score
   остаётся primary.
9. После freeze выполнить единственный integrity-gated test run.

Предлагаемый bounded grid:

```text
ka in {3, 5, 7, 9}
la in {8, 16, 24, 30, 48}

topology ablation at each surviving aligned configuration:
  disabled
  lagging kb2/mb1
  causal-leading kc2/mc1
  both kb2/mb1 + kc2/mc1
  lagging kb2/mb2
  both kb2/mb2 + kc2/mc2
  both kb4/mb1 + kc4/mc1
  lagging kb4/mb2
  every active lb/lc equals la

regularization refinement:
  ridge in {0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4}
  truncated-SVD rcond in
    {1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
```

Каждый product candidate сначала фильтруется строгим `<1000 MUL`, поэтому
широкие cross-memory combinations не обязательно будут fitted.
`opendpd_exact` leading mode является отдельным latency-bearing lookahead
diagnostic. Его можно реализовать с задержкой вывода, но нельзя смешивать с
zero-lookahead causal frontier.

Cost anchors текущего factorized counter:

| GMP candidate | MUL | ADD | Real coeffs | State reals | Lookahead |
|---|---:|---:|---:|---:|---:|
| `ka5/la30`, aligned only | 364 | 359 | 300 | 174 | 0 |
| `ka5/la30 + kb2/lb30/mb2 + causal kc2/lc30/mc2` | 832 | 827 | 768 | 178 | 0 |
| `ka5/la48`, aligned only | 580 | 575 | 480 | 282 | 0 |
| `ka5/la48 + kb2/lb48/mb1`, leading off | 772 | 767 | 672 | 284 | 0 |
| `ka9/la24 + kb2/lb24/mb2 + causal kc2/lc24/mc2` | 860 | 851 | 804 | 234 | 0 |
| OpenDPD APA full GMP reference dimensions | 2,764 | 2,759 | 2,700 | 224 | 5 |

Это factorized schedule
\(y[n]=\sum_q x[n-q]h_q(\text{envelope streams})\), не dense
`Phi @ coefficients`; ограничение явно записано в
`baseline/complexity.py:332-342`.

### 10.3 Acceptance для GMP

GMP становится новым frozen PA evaluator только если одновременно:

- validation pooled NMSE лучше selected MP;
- improvement воспроизводится на train inner blocked/OOF analysis и не
  является только reset-boundary эффектом;
- strict `<1000` MUL соблюдён;
- condition/rank не указывает на неуправляемую ill-conditioning;
- causal streaming chunks эквивалентны full causal inference;
- test не участвовал в выборе;
- evaluator margin заметно приближается к 10 dB gate.

Если GMP даёт малый gain при большом coefficient/state-memory росте, остаётся
MP Pareto point, а следующий шаг выбирается по новому residual analysis.
State-conditioned spline запрещён без independent long captures.

## 11. Что пока неизвестно или не выполнено

- Не установлено официальное определение Huawei `error < 10^-5`.
- Не установлено, означает ли `<1000 multipliers` operations/sample,
  физических DSP blocks или amortized update cost.
- Нет официальных DUT, carrier/power/backoff, waveform masks, feedback-path
  calibration и verification splits Huawei.
- Fractional-delay diagnostic пока не превращён в versioned band-limited
  correction protocol.
- Нет independent long captures для thermal/trapping state.
- Нет captures разных power levels/operating points для adaptation curves.
- Нет runnable bundled OpenDPD neural checkpoint.
- Нет locally rerun OpenDPD PA backbone в нашем frozen evaluator.
- Нет GMP selection/test result.
- Нет sparse spline-memory PA или spline/CPWL + FIR PA result.
- Нет bit-accurate 16/14/12-bit PA-model evaluation.
- Нет measured latency/throughput на FPGA/ASIC/DSP target.
- Нет physical-PA remeasurement с predistorted waveform.

Поэтому текущая корректная формулировка результата:

> Budget-constrained complex MP воспроизводимо моделирует held-out measured
> DPA/APA captures примерно на −35/−37 dB при менее 1000 counted real
> multiplications/sample. Он существенно точнее старого surrogate, но не
> достигает возможной −50 dB цели и не обеспечивает 10 dB evaluator margin для
> текущего DPD residual. GMP — следующий ограниченный experiment; DPD
> optimization остаётся приостановленной.
