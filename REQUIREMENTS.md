# Требования и границы проекта

Дата фиксации: 2026-07-29.

Этот документ разделяет три разных источника требований: два предоставленных
пользователем слайда Huawei, фактическое поведение OpenDPD и пока неизвестные
критерии заказчика. Слайды не содержат номера документа, даты, владельца
требований или полного технического задания, поэтому их нельзя считать
исчерпывающей формулировкой задачи Huawei.

## 1. Что явно следует из предоставленных слайдов

### 1.1 Постановка

На слайдах задача сформулирована прежде всего как идентификация нелинейной
RF-системы с памятью. Для complex time sequences \(X\) и \(Y\) требуется
подобрать отображение \(f\), минимизирующее ошибку

\[
\mathcal E(f)=\lVert f(X)-Y\rVert_2^2.
\]

Слайды также связывают идентификацию с построением inverse-компонента, который
создаёт искажения противоположного знака и тем самым компенсирует аналоговую
нелинейность. Из этого следуют две связанные, но разные задачи:

1. forward identification: \(x \rightarrow \widehat P \rightarrow \hat y\);
2. inverse/predistortion:
   \(x_\mathrm{desired}\rightarrow D\rightarrow P\rightarrow y\approx gx\).

Слайд не разрешает объединять их в один круговой reconstruction score.

### 1.2 Явно названные модельные классы

Слайды называют:

- Volterra series;
- Memory Polynomial;
- neural-network behavioral models;
- canonical piecewise-linear / CPWL models с локальным нелинейным базисом.

Нейросети описаны как модели с высокой fitting capability, но большой
ресурсоёмкостью, неудобной для real-time online coefficient solving. CPWL
предлагается как локальная альтернатива глобальному polynomial basis. Это
обосновывает исследование локальных spline/CPWL-базисов, но не доказывает
заранее их превосходство.

### 1.3 Явно названные физические эффекты

В разделе challenges указаны:

- частотно-зависимая нелинейность на широкой полосе;
- thermal memory, связанная с junction temperature;
- charge-trapping memory, зависящая от истории напряжения и захвата/эмиссии
  зарядов в semiconductor/2DEG.

Следовательно, чисто memoryless PA model недостаточен как окончательная модель.
State-conditioned/slow-memory variant следует добавлять только после
остаточного анализа, подтверждающего соответствующий time scale.

### 1.4 Явные цели

Модель должна иметь:

- сильную способность описывать неидеальные особенности PA;
- низкую сложность, на слайде грубо связываемую с числом multipliers;
- коэффициенты, пригодные для real-time calculation;
- обучение на training sets реальной схемы;
- проверку сначала на публичных воспроизводимых данных, затем на real-world
  service data.

На слайде явно указаны два численных ограничения:

- modeling error \(\mathcal E(f)<10^{-5}\) на verification set;
- менее 1000 real multipliers.

### 1.5 Что нельзя автоматически заключить из этих формулировок

Формула на слайде использует ненормированную squared \(L_2\)-норму. Поэтому
\(10^{-5}\) нельзя без уточнения объявить ни MSE, ни normalized error power, ни
NMSE \(<-50\) dB.

Если заказчик имеет в виду

\[
\frac{\sum_n|\hat y[n]-y[n]|^2}{\sum_n|y[n]|^2}<10^{-5},
\]

то это действительно эквивалентно pooled complex NMSE \(<-50\) dB. Если это
обычный MSE или сумма squared error, порог зависит от масштаба и числа samples.

Так же не определено, что означает «1000 multipliers»: число сохранённых
коэффициентов, физические multiplier units, peak real multiplications/sample,
средняя temporal activity или число операций за calibration block. В проекте
до уточнения используется консервативная software convention:

```text
1 complex multiplication = 4 real multiplications + 2 real additions
FMA = 1 real multiplication + 1 real addition
```

Square root, division, activation, comparison, LUT access и memory traffic
считаются отдельно.

## 2. Что явно следует из OpenDPD

Проверенная версия: `lab-emi/OpenDPD` commit
`7426bbf8a47624b59bd7f045a86641b403023f3c`.
Полный аудит находится в
[`research/opendpd_audit.md`](research/opendpd_audit.md).

### 2.1 OpenDPD решает две отдельные задачи

1. PA behavioral modeling обучается supervised в направлении
   \(x\rightarrow\widehat P(x)\approx y\)
   (`vendor/OpenDPD/steps/train_pa.py:17-49,83-99`,
   `vendor/OpenDPD/project.py:262-270`).
2. Neural DPD использует direct learning:
   desired \(x\rightarrow D(x)\rightarrow\widehat P(D(x))\), target \(gx\)
   (`vendor/OpenDPD/project.py:201-215`,
   `vendor/OpenDPD/models.py:172-185`,
   `vendor/OpenDPD/steps/run_dpd.py:79-99`).
3. MP/GMP DPD калибруются через ILA, но на validation/test получают desired
   `X_val/X_test`, а не measured `y_test`
   (`vendor/OpenDPD/benchmark/benchmark_volterra.py:899-940`).

Следовательно, наш проект также обязан иметь раздельные forward-PA и DPD
benchmark loops.

### 2.2 Dataset/evaluator contract

Built-in CSV split уже зафиксирован как train/validation/test = 60/20/20.
Для основных 200 MHz данных:

| Dataset | Train | Validation | Test | Fs | `nperseg` | Waveform |
|---|---:|---:|---:|---:|---:|---|
| DPA_200MHz | 23,040 | 7,680 | 7,680 | 800 MS/s | 2,560 | 10×20 MHz, 64-QAM |
| APA_200MHz | 58,980 | 19,662 | 19,662 | 983.04 MS/s | 19,662 | 5 carriers, 256-QAM metadata |

CSV loader не выполняет normalization/alignment. OpenDPD target gain — real
peak ratio

\[
g_\mathrm{peak}=\max|y_\mathrm{train}|/\max|x_\mathrm{train}|,
\]

а не complex least-squares gain
(`vendor/OpenDPD/utils/util.py:26-33`). Поэтому `opendpd_peak` и
`complex_ls` должны оставаться разными явно названными protocols.

Длительность train/validation records составляет лишь 28.8/9.6 µs для DPA и
примерно 60/20 µs для APA. Этого достаточно для проверки кратковременной
electrical memory, но недостаточно, чтобы по этим данным заявлять
идентификацию миллисекундной thermal drift. State-conditioned model допустим
только при воспроизводимой residual correlation на наблюдаемом time scale;
длинная thermal-memory проверка требует отдельного capture.

### 2.3 Метрики OpenDPD требуют compatibility-label

В repository:

- NMSE усредняет dB отдельных segments;
- “EVM” является spectral-bin error, а не demodulated constellation EVM;
- ACLR/ACPR нормируется на самый мощный in-band subchannel.

Источники:
`vendor/OpenDPD/utils/metrics.py:42-187`. Поэтому проект публикует одновременно:

- primary pooled complex NMSE;
- OpenDPD-compatible segment NMSE;
- time-domain RMS sample EVM;
- OpenDPD spectral EVM;
- OpenDPD-compatible left/right/average ACLR;
- при возможности стандартный integrated-channel ACLR и demodulated EVM.

### 2.4 Ограничения OpenDPD как reference

- Checkout не содержит готовых PA/DPD checkpoints; локальное воспроизведение
  требует полного обучения.
- Repository benchmark DPD оценивается через learned PA surrogate, а physical
  PA results статьи являются отдельным evidence layer.
- Stateful models сбрасывают state по segments; TRes содержит right context и
  wraparound feature, поэтому его нельзя считать causal streaming baseline без
  отдельного режима.
- 999 parameters не равны operations/sample; для TRes-DeltaGRU-H15 строгая
  arithmetic lower bound уже превышает 1000 real multiplications/sample.
- Included temporal sparsity kernels остаются dense matrix-vector execution;
  fake quantization не является bit-accurate fixed-point model.

Эти свойства не отменяют OpenDPD как quality baseline, но требуют одинакового
PA checkpoint, gain/alignment/framing и evaluator для честного сравнения.

## 3. Какие критерии Huawei остаются неизвестными

Следующие вопросы нельзя восстановить из двух слайдов и OpenDPD. До ответа они
считаются открытыми, а принятые в экспериментах значения — только внутренним
research protocol.

### 3.1 Определение качества

- Что именно есть \(\mathcal E(f)<10^{-5}\): SSE, MSE, normalized MSE,
  relative error power, maximum error или другая величина?
- Ошибка считается до или после gain/delay/phase alignment?
- Усреднение идёт по samples, frames, captures, carriers или operating points?
- Каковы обязательные NMSE, EVM и ACLR/ACPR limits и channel masks?
- Нужны ли worst-case/percentile gates, а не только mean?

### 3.2 Объект идентификации

- Требование относится к forward PA model, inverse model, полному DPD cascade
  или ко всем трём?
- Какой PA/DUT: topology, carrier frequency, bandwidth, sampling rate, output
  power/backoff, waveform, PAPR, antenna/array configuration?
- Включён ли measurement feedback path в модель или он предварительно
  equalized?
- Как представлены slow thermal/trapping states и достаточно ли capture
  duration для их наблюдения?

### 3.3 Complexity и real-time

- «<1000 real multipliers» означает operations per complex sample или
  количество одновременно размещённых hardware multiplier blocks?
- Complex multiply считается как 4M+2A или разрешена 3M+5A реализация?
- Можно ли amortize операции между samples/carriers/antennas?
- Каковы clock, throughput, maximum latency, batch/chunk size и допустимый
  look-ahead?
- Каковы budgets DSP/LUT/BRAM/SRAM/power/area и external-memory bandwidth?
- Входит ли coefficient-update/calibration engine в тот же multiplier budget?

### 3.4 Calibration и adaptation

- Что означает «real-time calculation of coefficients»: update каждый sample,
  frame, slot, burst, секунду или при смене operating point?
- Сколько calibration samples и wall-clock времени допустимо?
- Доступен ли clean desired/reference signal во время online adaptation?
- Какой feedback SNR/dynamic range и допустима ли closed-loop excitation?
- Как быстро должен отслеживаться temperature, bias, aging и load mismatch?

### 3.5 Fixed point и deployment

- Обязательные word lengths для input, coefficients, state и accumulators?
- Правила rounding, saturation, overflow и scaling?
- Разрешены ли sqrt/division, CORDIC, LUT, piecewise approximations?
- Требуется FPGA, ASIC, DSP/CPU или несколько targets?
- Какие bit-accurate reference vectors и tolerances использует заказчик?

### 3.6 Данные и acceptance

- Какие public и private service datasets являются официальными?
- Как зафиксированы train/validation/verification/test и guard intervals?
- Требуется ли generalization между waveform/power/bandwidth, между экземплярами
  одного PA или между разными PA?
- Сколько seeds/captures необходимо для confidence intervals?
- Какая physical-PA процедура является окончательным acceptance test?

## Рабочая интерпретация до получения ответов

Проект временно ведётся как двухконтурная задача:

```text
Контур A: x -> low-complexity PA model -> y_hat, compare with measured y
Контур B: desired x -> low-complexity DPD -> frozen independent PA -> g*x
```

Architecture selection использует только train/validation. Test открывается
после freeze. DPA_200MHz и APA_200MHz являются разными физическими PA и не
смешиваются как обычный train/test split.

Внутренний provisional gate для surrogate-based DPD:

- validation error power PA evaluator должен быть минимум в 10 раз меньше
  ожидаемого DPD residual, то есть иметь не менее 10 dB fidelity margin;
- ranking должен сохраняться на втором independently fitted surrogate;
- иначе DPD result помечается как evaluator-limited и не оптимизируется дальше.

Это не требование Huawei, а защитный исследовательский критерий против
эксплуатации ошибок surrogate. Окончательное доказательство требует
предикции на measured physical-PA output после подачи predistorted waveform.
