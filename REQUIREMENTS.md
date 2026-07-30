# Требования и границы проекта

Дата фиксации: 2026-07-30.

Этот документ разделяет три разных источника требований: два предоставленных
пользователем слайда Huawei, фактическое поведение OpenDPD и пока неизвестные
критерии заказчика. Слайды не содержат номера документа, даты, владельца
требований или полного технического задания, поэтому их нельзя считать
исчерпывающей формулировкой задачи Huawei.

## 0. Уточнение научного руководителя от 2026-07-30

Пользователь передал следующие разъяснения научного руководителя:

1. Ограничение, сформулированное как «1000 вещественных умножений», относится
   **только к исполняемому модулю DPD**, который формирует predistorted samples.
   Оно не относится к behavioral PA model: в конечной системе за DPD находится
   реальный усилитель.
2. Формулы на предоставленных слайдах являются иллюстративными и не должны
   использоваться как нормативное определение loss, model class или
   architecture.
3. Качество предполагается оценивать по затуханию паразитных гармоник.
4. Число 1000 задаёт не разрешённое число операций одного типа, а
   **эквивалентный временной бюджет**: все операции DPD вместе не должны
   исполняться дольше эталонной тысячи вещественных умножений.

Это более сильное рабочее уточнение, чем выводы из изображений слайдов, и далее
имеет приоритет над ними. Оно передано через пользователя, но не содержит
официального Huawei document ID/revision или bit-exact acceptance procedure,
поэтому точная спектральная метрика и способ измерения времени всё ещё должны
быть подтверждены владельцем требований.

Практические последствия:

- PA surrogate/evaluator выбирается прежде всего по fidelity, независимости и
  корректности ranking; его стоимость публикуется, но не проверяется против
  DPD latency gate;
- hard complexity gate применяется к deployment DPD datapath;
- `sqrt`, division, activation, LUT/addressing, comparisons и memory traffic
  нельзя считать бесплатными только потому, что они не являются
  умножениями;
- до появления target platform используется полный operation vector, а не
  только `MUL/sample`;
- окончательная проверка должна сравнивать измеренное время DPD с эталоном на
  одном target, с одинаковыми word length, streaming contract и правилами
  parallelism:

  \[
  T_\mathrm{DPD/sample}\ \leq\
  T_\mathrm{reference}(1000\ \text{real multiplications}).
  \]

## 1. Что явно следует из предоставленных слайдов

### 1.1 Постановка

На слайдах задача сформулирована прежде всего как идентификация нелинейной
RF-системы, для которой memory effects отдельно названы среди challenges.
\(X\), \(Y\) и \(D\) описаны как one-dimensional complex time sequences.
Обозначения на слайде неоднозначны: mapping записан как \(f:X\to D\), но ниже
ошибка сравнивает \(f(X)\) с \(Y\):

\[
\mathcal E(f)=\lVert f(X)-Y\rVert_2^2.
\]

По уточнению научного руководителя эта формула декоративная: она не определяет
acceptance loss и не является основанием трактовать `10^-5` как NMSE. Слайды
также связывают identification/inversion с построением нелинейного компонента,
поведение которого противоположно нелинейности аналоговой схемы. Однако они не
задают формальный forward/inverse interface, эксплуатационный вход
predistorter, target gain или test path. Двухконтурная постановка и запрет
круговой DPD-оценки ниже являются рабочим контрактом из постановки пользователя
и проверенного OpenDPD pipeline, а не восстановленным требованием слайда.

### 1.2 Явно названные модельные классы

Слайды называют:

- Volterra series;
- Memory Polynomial;
- Generalized Memory Polynomial (GMP), показанный формулой с cross-memory
  terms;
- neural-network behavioral models;
- canonical piecewise-linear / CPWL models с локальным нелинейным базисом.

Нейросети описаны как модели с высокой fitting capability, но большой
ресурсоёмкостью, неудобной для real-time online coefficient solving. CPWL
показан как локальная альтернатива global polynomial basis. По последующему
уточнению формулы и перечисленные на слайде модели не являются предписанной
архитектурой. Поэтому spline/CPWL остаётся исследовательским кандидатом только
по измеренному Pareto trade-off, а не потому, что он изображён на слайде.

### 1.3 Явно названные физические эффекты

В разделе challenges указаны:

- частотно-зависимая нелинейность на широкой полосе;
- thermal memory, связанная с junction temperature;
- charge-trapping memory, зависящая от истории напряжения и захвата/эмиссии
  зарядов в semiconductor/2DEG.

Следовательно, memory behavior входит в область исследования. Однако по
слайдам нельзя установить, какие именно memory mechanisms наблюдаемы в
конкретном capture и требуется ли для них отдельное slow state.

### 1.4 Явные цели

Модель должна иметь:

- сильную способность описывать неидеальные особенности PA;
- низкую сложность, на слайде грубо связываемую с числом multipliers;
- коэффициенты, пригодные для real-time calculation;
- обучение на training sets реальной схемы;
- self-verification на конкретных public data и parameter settings с
  воспроизводимым output;
- последующую verification на real-world service data, подтверждающую
  performance, complexity и real-time computing indicators.

На слайде визуально указаны два численных ограничения:

- modeling error \(\mathcal E(f)<10^{-5}\) на verification set;
- менее 1000 real multipliers.

Последующее уточнение меняет их рабочую трактовку: формула error не задаёт
acceptance metric, качество оценивается по затуханию паразитных гармоник, а
временной эквивалент 1000 real multiplications относится только к DPD.

### 1.5 Мотивация на слайде, но не нормативные критерии

Background slide утверждает, что RF nonlinearity ухудшает signal quality и
вызывает spectral spreading. В качестве ожидаемых benefits названы улучшение
эффективности PA более чем на 5%, снижение power consumption digital chips и
ожидаемое улучшение integration на 50% для 5G massive MIMO.

На слайде нет baseline, определения метрик или acceptance procedure для этих
чисел. Поэтому они фиксируются только как motivation claims, а не как
подтверждённые требования или результаты проекта.

### 1.6 Что нельзя автоматически заключить из этих формулировок

Декоративную формулу на слайде нельзя использовать для определения метрики.
Поэтому \(10^{-5}\) нельзя без отдельного подтверждения объявить ни MSE, ни
normalized error power, ни NMSE \(<-50\) dB и нельзя считать текущим hard
acceptance gate.

Если заказчик имеет в виду

\[
\frac{\sum_n|\hat y[n]-y[n]|^2}{\sum_n|y[n]|^2}<10^{-5},
\]

то это действительно эквивалентно pooled complex NMSE \(<-50\) dB. Если это
обычный MSE или сумма squared error, порог зависит от масштаба и числа samples.
Это условная математическая конверсия для дополнительной диагностики, не
восстановленное требование.

Объект ограничения теперь известен: online DPD sample datapath. Не определены
target platform и эталонный timing kernel, то есть как именно сравнивать
parallel/pipelined implementation, memory stalls и nonlinear primitives с
тысячей вещественных умножений. В проекте сохраняется operation convention:

```text
1 complex multiplication = 4 real multiplications + 2 real additions
FMA = 1 real multiplication + 1 real addition
```

Square root, division, activation, comparison, LUT access и memory traffic
считаются отдельно, а финальный pass/fail требует target timing, а не
подстановки произвольных весов операций.

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
примерно 60/20 µs для APA. Она ограничивает доступный для идентификации time
horizon и позволяет искать residual structure только внутри наблюдаемого
record, но сама по себе не доказывает достаточность данных даже для всех видов
short electrical memory. По этим captures нельзя делать выводы об эффектах на
более длинных time scales; их проверка требует отдельного более длинного
capture.

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

### 3.0 Статус и владелец требований

- Какой authoritative specification, revision и owner определяют acceptance?
- Являются ли предоставленные слайды нормативным документом, summary или
  предварительной постановкой?
- Какие требования являются hard gates, а какие только optimization goals?

### 3.1 Определение качества

- Что именно есть \(\mathcal E(f)<10^{-5}\): SSE, MSE, normalized MSE,
  relative error power, maximum error или другая величина?
- Остаётся ли `10^-5` дополнительным требованием после перехода к спектральной
  оценке или это только декоративный текст слайда?
- Под «паразитными гармониками» имеются в виду true RF harmonics около
  \(2f_c,3f_c,\ldots\), in-band intermodulation products, adjacent-channel
  spectral regrowth или точки заданной emission mask?
- Затухание измеряется в dBc относительно carrier/occupied-channel power или
  как improvement `no DPD -> DPD`? Каковы exact integration bands, resolution
  bandwidth, window/FFT/Welch settings и требуемый порог?
- Ошибка считается до или после gain/delay/phase alignment?
- Усреднение идёт по samples, frames, captures, carriers или operating points?
- Каковы обязательные NMSE, EVM и ACLR/ACPR limits и channel masks?
- Нужны ли worst-case/percentile gates, а не только mean?
- Как разрешается Pareto trade-off между fidelity, multiplier count,
  coefficient-update latency, memory, power и area?

### 3.2 Объект идентификации

- Требование относится к forward PA model, inverse model, полному DPD cascade
  или ко всем трём?
- Какова точная семантика \(X\), \(Y\), \(D\) и mapping \(f:X\to D\), если
  приведённый loss сравнивает \(f(X)\) с \(Y\)?
- Какой PA/DUT: topology, carrier frequency, bandwidth, sampling rate, output
  power/backoff, waveform, PAPR, antenna/array configuration?
- Включён ли measurement feedback path в модель или он предварительно
  equalized?
- Как представлены slow thermal/trapping states и достаточно ли capture
  duration для их наблюдения?

### 3.3 Complexity и real-time

- Подтверждено: latency-equivalent budget относится к deployment DPD, не к PA
  behavioral model.
- Как определяется эталон времени 1000 real multiplications: serial dependency
  chain, independent vector, pipelined stream или target-specific reference
  kernel?
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
- Какие exact dataset versions, parameter settings, seeds, reference outputs и
  numerical tolerances образуют public self-verification contract?
- Как зафиксированы train/validation/verification/test и guard intervals?
- Требуется ли generalization между waveform/power/bandwidth, между экземплярами
  одного PA или между разными PA?
- Сколько seeds/captures необходимо для confidence intervals?
- Какая physical-PA процедура является окончательным acceptance test?

### 3.7 Заявленные benefits

- Являются ли `>5%` PA-efficiency improvement, снижение digital-chip power и
  `50%` integration improvement реальными acceptance targets?
- Если да, как определены efficiency, digital power и integration, относительно
  какого baseline, на каких operating points и каким способом они измеряются?

## Рабочая интерпретация до получения ответов

На основании пользовательской постановки, уточнения научного руководителя и
проверенного OpenDPD pipeline, а не только двух слайдов, проект ведётся как
двухконтурная задача:

```text
Контур A: x -> high-fidelity PA evaluator -> y_hat, compare with measured y
Контур B: desired x -> low-latency DPD -> frozen independent PA/real PA -> g*x
```

Контур A может иметь больше 1000 real MUL/sample: это offline research
instrument, а не блок передатчика. Его complexity всё равно измеряется для
воспроизводимости и evaluator deployment, но не является Huawei DPD gate.
Контур B обязан пройти spectral-quality gate и измеренный
1000-real-MUL-equivalent latency gate.

Architecture selection использует только train/validation. Test открывается
после freeze. DPA_200MHz и APA_200MHz являются разными физическими PA и не
смешиваются как обычный train/test split.

В честном DPD test на вход DPD подаётся desired \(x_\mathrm{test}\), а measured
\(y_\mathrm{test}\) не используется как DPD input. Путь
\(y/g\rightarrow\widehat P^{-1}\rightarrow\hat x\rightarrow\widehat P
\rightarrow\hat y\) допустим только как явно обозначенный ILA/inverse-forward
diagnostic и не доказывает predistortion нового desired input.

Memoryless model остаётся обязательным baseline. State-conditioned/slow-memory
variant добавляется только после residual evidence соответствующего time scale
на train/validation или после получения подходящего длинного capture.

Внутренний provisional gate для surrogate-based DPD:

- validation error power PA evaluator должен быть минимум в 10 раз меньше
  ожидаемого DPD residual, то есть иметь не менее 10 dB fidelity margin;
- ranking должен сохраняться на втором independently fitted surrogate;
- иначе DPD result помечается как evaluator-limited и не оптимизируется дальше.

Это не требование Huawei, а защитный исследовательский критерий против
эксплуатации ошибок surrogate. Окончательное доказательство требует
предикции на measured physical-PA output после подачи predistorted waveform.
