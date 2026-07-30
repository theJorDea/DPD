# Benchmark low-complexity PA models

Дата среза: 2026-07-30.

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
Дополнительно завершён отдельный train-OOF search standalone
phase-equivariant spline-Hammerstein (SPH) PA на `APA_200MHz`. SPH был
запущен как следующий изолированный low-complexity candidate; он не изменял
GMP evaluator и не открывал контур B.
После SPH выполнен preregistered non-factorized sparse spline-memory PA search
на том же `APA_200MHz`: он также является самостоятельной forward-моделью,
а не residual correction к GMP. Его OOF selection и full-train refit
опубликованы отдельно; validation загружена только после freeze.
Residual-guided lag-9 neighborhood был затем проверен отдельным
preregistered run с тем же frozen implementation; он улучшил parent до
cheap-Pareto уровня, но не заменил GMP evaluator.
После freeze source family выполнен отдельный preregistered capture transfer
`APA_200MHz -> APA_200MHz_b`, включая frozen one-shot held-out release.
Zero-shot fidelity резко падает примерно до −23.8 dB, тогда как
coefficient-only recalibration на длинном target-train prefix возвращает GMP
к −37.895 dB на held-out test; это evidence capture drift, а не доказательство
power или thermal adaptation.
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
- APA SPH search дал −30.4024 dB train-OOF full-record NMSE при 37 real
  MUL/sample, то есть на 6.652 dB хуже matched MP и на 7.943 dB хуже GMP;
  поэтому он отклонён как evaluator replacement и как cheap Pareto point;
- bounded non-factorized sparse spline-memory search улучшил SPH до
  −32.0300 dB train-OOF full (−32.0882 dB common) при 54 MUL/58 ADD, но
  всё ещё на 5.024 dB хуже matched MP и на 6.315 dB хуже GMP; evaluator gate
  не пройден;
- sparse run выявил новый воспроизводимый causal proper-correlation peak около
  lag 9 (`|proper|=0.691` train OOF, `0.691` reused validation), тогда как
  ранее проверенные lag 22–24 branches были недостаточны;
- bounded lag-9 sparse PA достиг `−37.7925/−37.8528 dB` full/common train OOF
  и `−37.8607 dB` reused validation при `72 MUL/82 ADD`, превзойдя matched MP
  на `0.738/0.753 dB`, но уступив GMP на `0.553/0.898 dB`;
- DPD optimization остаётся остановленной: следующий источник информации —
  capture metadata и controlled physical-PA experiment, не новый local delay
  sweep.
- На independent declared capture `APA_200MHz_b` zero-shot source transfer
  дал `−23.7948 dB` GMP и `−23.7014 dB` lag-9 sparse на validation. После
  coefficient-only target-train calibration (`N=16384` samples/frame) GMP
  достиг `−37.8908 dB`, sparse `−35.3585 dB`.
- Frozen held-out release подтвердил GMP: `−37.8952 dB` full-record и
  `−38.0038 dB` common-support test NMSE. Sparse дал `−34.8015 dB`
  full-record и `−35.4380 dB` common-support; его full-record degradation
  относительно validation равна `0.5570 dB`, но common-support degradation
  только `0.0080 dB`, поэтому основная нестабильность локализована на
  24-sample frame boundary.
- Ни GMP (`1.6236e-4`), ни sparse (`3.3102e-4`) не достигли `1e-5`
  normalized error power на target test. Gate A→B остаётся закрытым.

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

### 10.6 APA standalone spline-Hammerstein search — завершён, отрицательный результат

После двух малых linear residual ablations был выполнен отдельный
train-only staged search для модели

\[
\hat y[n] = v[n] + \sum_{m=1}^{L-1}h_m v[n-m],\qquad
v[n]=x[n]C(|x[n]|),
\]

где `C` — complex piecewise-linear spline с локальной поддержкой, а `h[0] = 1`
зафиксирован. Такая форма сохраняет phase equivariance и causal streaming,
но является factorized: одна и та же spline correction используется для всех
FIR delays. Search был заранее ограничен вариантами coordinate/knot placement,
`K`, `L` и ridge/smoothness; fit выполнялся deterministic complex ALS на
leave-one-explicit-frame-out train folds.

Команда:

```bash
.venv/bin/python -m experiments.run_pa_sph \
  --config experiments/configs/pa_sph_apa200.json
```

| Candidate / split | Full pooled NMSE, dB | Common-interior NMSE, dB | OpenDPD-compatible, dB | MUL / ADD | Stored real coeff. | State reals |
|---|---:|---:|---:|---:|---:|---:|
| Matched MP reference, train OOF | −37.054329 | −37.099951 | — | 960* / 628* | 300* | 58* |
| Matched GMP reference, train OOF | −38.345410 | −38.750526 | −38.478780 | 954* / 947* | 888* | 236* |
| Selected SPH (`K=32`, `L=8`), train OOF | −30.402374 | −30.437014 | −30.378779 | **37 / 36** | **78** | **14** |
| Selected SPH, full-train refit | −30.413203 | −30.444686 | −30.391324 | 37 / 36 | 78 | 14 |
| Selected SPH, reused validation (descriptive) | −30.473868 | −30.480138 | −30.473868 | 37 / 36 | 78 | 14 |

`*` Reference rows reproduce the frozen GMP/MP ledger convention; the exact
matched GMP numbers are retained in the machine-readable bundle. The SPH
result is the only new fitted candidate in this section. No APA test file was
opened or hashed.

The final hard-valid recipe was
`amplitude_uniform_K32_L8_cr1e-08_sm1e-08_fr0e+00`. It was selected after:

- `K=1…4` FIR memory ablation showed the dominant improvement came from
  increasing causal memory (`−22.2645`, `−25.5506`, `−29.1590`, and
  `−30.3936 dB` full-record for `L=1,2,4,8` in the representative S0 path);
- knot placement changed the score only by hundredths of a dB;
- `K=48` and `K=64` had slightly better raw scores but were rejected by the
  frozen identifiability gate: one K48 fold had data-design rank `47/48`
  (minimum nonzero-feature count `0`), while K64 folds had rank `62–63/64`;
- the selected regularization was `control_ridge=1e-8`,
  `smoothness=1e-8`, `fir_ridge=0` under the preregistered tie rule.

The decision is quantitative, not a visual judgment:

- SPH loses to matched MP by `6.651955 dB` full and `6.662937 dB` common;
- SPH loses to frozen GMP by `7.943037 dB` full and `8.313512 dB` common;
- worst train-fold loss versus GMP is `8.344366 dB` full and `8.380414 dB`
  common;
- the internal cheap-Pareto gate allowed at most 3 dB loss versus MP, so it
  fails even though its arithmetic cost is far below 1000 MUL/sample.

Residual evidence explains the failure mode. On train OOF, the SPH residual
has radial/tangential RMS `0.007716/0.008034`; its largest causal proper
correlation is at lags 22–24 (`0.684–0.723`) and the same peak appears on
reused validation (`0.678–0.718`). Correlation with the instantaneous
envelope is small (`0.024` radial at lag zero), and the slow-state gate remains
ineligible because the capture count is zero. Thus increasing `K` or adding a
state is not justified by this run; the next bounded family should allow
delay-dependent nonlinear coefficients, e.g. a sparse non-factorized
spline-memory dictionary, while retaining `<1000` MUL and train-only OOF
selection.

Evidence and hashes are immutable in
`experiments/results/pa_sph_apa200_selection/`; the publication manifest
records source/config/data hashes, all 60 unique recipes, 180 completed OOF
fits, validation-after-freeze ordering, exact streaming/reset checks and the
fact that `test_split_accessed=false`.

## 10.7 Non-factorized sparse spline-memory PA — completed negative result

После factorized SPH был выполнен отдельный preregistered search:

```text
y_hat[n] = sum_b x[n-m_b] C_b(|x[n-d_b]|)
```

`C_b` — complex local linear spline на общей amplitude-knot сетке; каждый
sample активирует ровно две соседние control points на branch. Signal и
envelope delays causal, I/Q fit joint complex, frame boundaries reset. Search
не был additive correction к GMP и не использовал measured output как
кандидатный input для DPD.

Команда:

```bash
.venv/bin/python -m experiments.run_pa_sparse_spline_memory \
  --config experiments/configs/pa_sparse_spline_memory_apa200.json
```

S0 проверил 7 topology families при `K=12`; внутри preregistered окна был
оставлен один topology. S1 проверил `K={8,12,16,24}`, S2 — пять ridge values.
Фактически выполнено 16 stage associations, 14 уникальных recipes и 42
OOF-fold fits (повторные recipes использовали hash cache). Selection samples
были только тремя явными train frames; validation загружена после freeze,
а test split не открывался и не хешировался.

| Candidate / split | Full pooled NMSE, dB | Common-interior NMSE, dB | OpenDPD-compatible, dB | MUL / ADD | Stored real coeff. / state |
|---|---:|---:|---:|---:|---:|
| Matched MP reference, train OOF | −37.054329 | −37.099951 | — | 960 / 628 | 300 / 58 |
| Matched GMP reference, train OOF | −38.345410 | −38.750526 | −38.478780 | 954 / 947 | 888 / 236 |
| Selected sparse, train OOF | −32.030011 | −32.088250 | — | **54 / 58** | **144 / 48** |
| Selected sparse, full-train refit | −32.049190 | −32.107450 | −32.045143 | 54 / 58 | 144 / 48 |
| Selected sparse, reused validation | −32.048219 | −32.071529 | −32.048219 | 54 / 58 | 144 / 48 |

Selected recipe:
`mixed_diagonal_long_K12_r0e+00_b0:0,1:1,2:2,22:22,23:23,24:24`.
Its fit design is full rank (`72/72`), augmented condition number `78.57`,
minimum nonzero feature support `8` samples, maximum coefficient magnitude
`1.3282`, and maximum causal memory `24` samples. Exact analytical inference
count is 54 real MUL, 58 real ADD, 6 magnitude nonlinear operations, 24
comparisons, 12 LUT accesses, 144 stored real coefficients and 48 state reals.

The quality decision is unambiguous:

- versus matched MP: loss `5.024318 dB` full and `5.011702 dB` common;
- versus matched GMP: loss `6.315399 dB` full and `6.662276 dB` common;
- worst OOF-fold loss versus GMP: `6.678227 dB` full and `6.712301 dB`
  common;
- the preregistered cheap-Pareto limit was 3 dB versus MP, so the candidate is
  classified `neither_evaluator_nor_cheap_pareto` and Gate A→B remains closed.

Residual analysis was used to preregister the next bounded hypothesis. The
selected model's
train-OOF residual has pooled common NMSE `−32.08825 dB`; the strongest causal
proper correlation is at lag 9 (`0.69064`), reproduced at `0.69131` on reused
validation. Radial/tangential envelope correlations are much smaller (largest
radial-envelope value about `0.140` at lag 2), and the slow-state branch remains
ineligible because independent-capture count is zero. Thus the present result
does not justify adding a state or more knots; it justifies preregistering a
new branch dictionary containing lag-9 neighborhoods, with all gates frozen
before fitting. That lag-9 run is now reported in Section 10.8; no further
local delay expansion is authorized on this capture.

Machine-readable evidence is the immutable directory
`experiments/results/pa_sparse_spline_memory_apa200_selection/`:
`selection_manifest.json`, `staged_trials.json`, `predictions.npz`, the two
residual reports, `selected_sparse_pa.npz` and `execution_record.json`.
The manifest records runtime `33.5888 s`, exact streaming/reset equivalence,
hash re-verification before publication, and `test_split_accessed=false`.

## 10.8 Residual-guided lag-9 sparse PA — completed cheap-Pareto result

The next experiment was preregistered before candidate fitting in
`experiments/configs/pa_sparse_spline_memory_lag9_apa200.json`. It allowed nine
explicit topology families, `K={8,12,16}`, four ridge values and at most 66
OOF fit calls. The parent model and all residual/config/source/data hashes were
recorded as immutable evidence; test access remained forbidden.

The selected phase-equivariant model is:

```text
y_hat[n] = sum_b x[n-m_b] C_b(|x[n-d_b]|)
(m,d) = (0,0),(1,1),(2,2),(22,22),(23,23),(24,24),(8,0),(9,0),(10,0)
K = 12, ridge = 1e-8
```

| Candidate / split | Full pooled NMSE, dB | Common-interior NMSE, dB | OpenDPD-compatible, dB | MUL / ADD | Stored real coeff. / state |
|---|---:|---:|---:|---:|---:|
| Frozen parent sparse, train OOF | −32.030011 | −32.088250 | — | 54 / 58 | 144 / 48 |
| Lag-9 sparse, train OOF | **−37.792478** | **−37.852832** | — | **72 / 82** | **216 / 48** |
| Lag-9 sparse, full-train refit | −37.866643 | −37.927296 | −37.833864 | 72 / 82 | 216 / 48 |
| Lag-9 sparse, reused validation | −37.860728 | −37.898605 | −37.860728 | 72 / 82 | 216 / 48 |

The incremental gate passed in every OOF fold:
`+5.7624669/+5.7645826 dB` full/common pooled gain over the frozen parent,
with minimum fold gains `+5.7178445/+5.7313385 dB`. The candidate also passes
the internal cheap-Pareto rule versus matched MP by
`0.7381497/0.7528810 dB`, while it remains worse than matched GMP by
`0.5529319/0.8976938 dB`. Classification is therefore
`cheap_pareto_only`, not `evaluator_candidate`.

Exact inference bookkeeping uses the `4 real MUL + 2 real ADD` complex-product
convention: 72 real MUL, 82 real ADD, 6 magnitude nonlinear operations,
24 comparisons, 18 LUT accesses, 54 reads, 2 writes, 216 stored real
coefficient values, 23 constants and 48 state real values. The selected design
has rank `108/108`, augmented condition `2427.39`, maximum coefficient
`1.15093`, and exact streaming/reset equivalence. Runtime before publication
was `62.5693 s` on the recorded Python/NumPy host.

The two `(0,d)` envelope-only variants were rejected before ranking because
their designs were rank deficient (`105/108` and `141/144`): local spline
partition of unity duplicates the same `x[n]` linear component when the signal
delay is unchanged. The `K=16` selected topology was also rank deficient in
OOF folds, so the hard identifiability gate correctly excluded it.

Residual analysis after the selected model no longer shows the lag-9 peak:
the largest causal proper correlation in train OOF is `0.07106` at lag 32,
and validation has the same low-magnitude region. This is still
within-capture evidence. The immutable bundle is
`experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/`; all six
payload hashes and input hashes were reverified, and `test_split_accessed=false`.

## 10.9 APA_200MHz → APA_200MHz_b capture transfer

Перед запуском target fit был зафиксирован
`experiments/configs/pa_transfer_apa200_to_b.json`. Source и target inputs
`train`/`val` побитно идентичны, а measured outputs имеют разные hashes.
Поэтому это same-excitation capture transfer; metadata не позволяет назвать
изменение power, bias или temperature.

Protocol не менял source topology/knots/coefficients для zero-shot режима:

```text
source frozen PA model -> target input -> y_hat_target
compare with target measured y
```

Target coefficient adaptation refit-ит только complex coefficients на
`[0:N)` prefix каждой из трёх target-train frames. `N` и solver axes были
зафиксированы заранее; validation загружена только после завершения всех
prefix fits. После выбора `N=16384` для обеих families topology, coefficients,
metrics и release hashes были повторно заморожены до held-out evaluation.

| Model / mode | N per frame | Target val full NMSE | Common NMSE | Fit time, s | MUL / ADD | FP32 model+state |
|---|---:|---:|---:|---:|---:|---:|
| GMP zero-shot | 0 | −23.794841 | −23.793859 | 0 | 954 / 947 | 4532 B |
| lag-9 sparse zero-shot | 0 | −23.701383 | −23.703027 | 0 | 72 / 82 | 1148 B |
| GMP coefficient-only | 256 | −30.019598 | −30.029903 | 0.076 | 954 / 947 | 4532 B |
| GMP coefficient-only | 1024 | −36.884942 | −36.927799 | 0.148 | 954 / 947 | 4532 B |
| GMP coefficient-only | 4096 | −37.646230 | −37.696794 | 0.772 | 954 / 947 | 4532 B |
| GMP coefficient-only | 16384 | **−37.890764** | **−37.961563** | 6.860 | 954 / 947 | 4532 B |
| lag-9 sparse coefficient-only | 512 | −26.727371 | −26.729178 | 0.034 | 72 / 82 | 1148 B |
| lag-9 sparse coefficient-only | 2048 | −31.575753 | −31.593238 | 0.117 | 72 / 82 | 1148 B |
| lag-9 sparse coefficient-only | 8192 | −35.076113 | −35.138897 | 0.628 | 72 / 82 | 1148 B |
| lag-9 sparse coefficient-only | 16384 | **−35.358475** | **−35.446027** | 1.513 | 72 / 82 | 1148 B |

GMP `N=64/128` были записаны как infeasible: 3 prefixes не дают
overdetermined 444-complex-column solve. Sparse `N=64/128/256` технически
full-rank, но дают хуже zero-shot из-за малого support/conditioning; это
отрицательный calibration result, а не основание скрывать эти точки.

Nuisance diagnostic, fitted only on target train, нашёл integer delay `0` и
complex LS gain magnitude `1.152146`. Strict score намеренно не менялся
post-prediction gain fit; diagnostic не участвовал в выборе model/N. Streaming
chunk and reset equivalence remained exact for every published model.

Интерпретация ограничена capture transfer. Source model не переносится
zero-shot на B capture (потеря около 14.9 dB относительно source GMP control),
но coefficient-only calibration восстанавливает большую часть fidelity с
`6.86 s` fit для GMP или `1.51 s` для sparse. При этом sparse остаётся на
`2.53 dB` хуже GMP после полной preregistered calibration, хотя дешевле по
MUL и памяти. Это не Gate A→B и не DPD evidence.

Machine-readable bundle и независимый verifier:

- `experiments/results/pa_transfer_apa200_to_b_pretest/`;
- `experiments/verify_pa_transfer_bundle.py`;
- verifier проверяет 20 metric records, payload/source/data hashes и sealed
  held-out boundary.

### 10.9.1 Frozen held-out release

Target test использовался только в направлении forward identification:

```text
target measured x_test
    -> frozen zero-shot or target-train-calibrated PA model
    -> y_hat_test
    -> compare with target measured y_test
```

Ни topology, ни `N`, ни coefficients, ни metric protocol по test не
выбирались. Результаты:

| Model / mode | Target test full NMSE | Common NMSE | Relative error power | Fit time, s | Inference batch, s | MUL / ADD |
|---|---:|---:|---:|---:|---:|---:|
| GMP zero-shot | −23.795441 | −23.800907 | 4.1731e−3 | 0 | 0.03616 | 954 / 947 |
| GMP coefficient-only, N=16384 | **−37.895152** | **−38.003839** | **1.6236e−4** | 6.860 | 0.03637 | 954 / 947 |
| lag-9 sparse zero-shot | −23.695838 | −23.700933 | 4.2699e−3 | 0 | 0.00620 | 72 / 82 |
| lag-9 sparse coefficient-only, N=16384 | −34.801474 | −35.437986 | 3.3102e−4 | 1.513 | 0.00547 | 72 / 82 |

Для GMP validation и held-out test практически совпали:
`−37.890764 -> −37.895152 dB`. У sparse full-record score изменился
`−35.358475 -> −34.801474 dB`, но на общем steady-state support
`−35.446027 -> −35.437986 dB`. Следовательно, примерно `0.55 dB` различия
full-record связано главным образом с causal startup/reset boundary, а не с
разрушением steady-state transfer. Оба представления публикуются; удобный
common score не заменяет primary full-record score.

Sparse вариант в `13.25x` дешевле GMP по real MUL, примерно в `4.53x`
быстрее калибруется и в этом host batch был примерно в `6.65x` быстрее, но
проигрывает GMP `3.094 dB` по primary held-out NMSE (`2.566 dB` на common
support). Поэтому это cost/quality Pareto point, а не evaluator replacement.

Release audit не является идеальным single-open execution. Первый process
загрузил target test pair, затем остановился до model inference из-за
ошибочной проверки train-frame lengths: код ожидал
`19662+19662+19662`, тогда как frozen train framing равен
`19662+19662+19656`. Ни test metric, ни prediction тогда не вычислялись и
никакое selection decision не менялось. Исправлен только guard, после чего
тот же frozen protocol был выполнен со вторым доступом. Поэтому:

- held-out access count: `2`;
- first access: metric-free, before inference;
- `strict_single_open_execution=false`;
- test used for selection/coefficient fit: `false`.

Артефакты:

- release config:
  `experiments/configs/pa_transfer_apa200_to_b_release.json`;
- incident:
  `experiments/results/pa_transfer_apa200_to_b_release_incident_001.json`;
- immutable release:
  `experiments/results/pa_transfer_apa200_to_b_test_release/`;
- independent verification:
  `experiments/results/pa_transfer_apa200_to_b_test_release_verification.json`;
- verifier reproduced 4 metric records, test/data/source/pretest hashes,
  incident linkage and the two-access audit.

## 11. Что пока неизвестно или не выполнено

- Не установлено официальное определение Huawei `error < 10^-5`.
- Не установлено, означает ли `<1000 multipliers` operations/sample,
  физических DSP blocks или amortized update cost.
- Нет официальных DUT, carrier/power/backoff, waveform masks, feedback-path
  calibration и verification splits Huawei.
- A0/A1 fractional-delay sensitivity выполнена, но independent feedback-path
  calibration/de-embedding всё ещё отсутствует; A0 frozen без correction.
- Нет independent long captures для thermal/trapping state.
- Нет captures с известными и контролируемыми power levels/operating points
  для adaptation claim; held-out `APA_200MHz_b` release завершён, но
  measurement B остаётся нерасшифрованным capture transfer.
- Нет runnable bundled OpenDPD neural checkpoint.
- Нет locally rerun OpenDPD PA backbone в нашем frozen evaluator.
- Widely-linear, proper long-FIR и standalone SPH отклонены по quality gates.
  Первый non-factorized sparse PA также отклонён, а lag-9 sparse family
  проходит cheap-Pareto gate, но всё ещё уступает GMP по fidelity и не может
  служить независимым evaluator.
- Нет bit-accurate 16/14/12-bit PA-model evaluation.
- Нет measured latency/throughput на FPGA/ASIC/DSP target.
- Нет physical-PA remeasurement с predistorted waveform.

Поэтому текущая корректная формулировка результата:

> Validation-selected causal factorized GMP воспроизводимо моделирует
> held-out measured DPA/APA captures на −35.385/−38.608 dB full-record pooled
> NMSE при 766/954 counted real multiplications/sample. Он улучшает локальный
> MP, особенно на APA, но не достигает возможной −50 dB цели и не обеспечивает
> 10 dB evaluator margin для текущего surrogate-only DPD residual. Release
> gates, one-shot source tests и target capture-transfer release завершены.
> Target-calibrated GMP достиг −37.895 dB на B-capture held-out test, но
> normalized error power 1.62e−4 всё ещё выше возможного требования 1e−5.
> Gate A→B остаётся закрытым, поэтому DPD optimization приостановлена до
> controlled physical-PA и/или более точного independent PA evidence.
