# Итог глубокого исследования следующего поколения DPD

Дата фиксации: 2026-08-02
Статус: research-only recommendation; новые алгоритмы и frozen results не
изменялись.

## 1. Короткий ответ

Наиболее обоснованный следующий путь проекта — сохранить текущий дешёвый,
причинный и фазово-эквивариантный spline-memory DPD как основной тракт и
добавлять к нему только те немногочисленные residual-ветви, необходимость
которых воспроизводимо обнаружена observation receiver и подтверждена
независимой спектральной проверкой.

Рекомендуемая последовательность:

1. **Observer-only:** ничего не менять в DPD, а определить, содержит ли остаток
   устойчивую пропущенную структуру.
2. **Advisor:** предложить не более одной причинной группы признаков и сравнить
   выбор с random, exhaustive, DOMP и group-LASSO на validation.
3. **Shadow validation:** проверить новые коэффициенты на независимом capture
   или evaluator, не применяя их в рабочем тракте.
4. **Safe apply/rollback:** включать обновление только после спектральных,
   амплитудных, EVM и timing gates; хранить предыдущий проверенный bank.
5. **Physical PA:** считать результат доказанным только после парных измерений
   реального усилителя.

Рабочее название научной гипотезы:

> **PA-sensitivity-aware, cost-aware sparse residual adaptation for a
> phase-equivariant low-complexity DPD with independent spectral validation and
> transactional rollback.**

Это потенциально новая комбинация известных механизмов. Сама корреляция
остатка, OMP, spline DPD, decorrelation learning или rollback по отдельности
новизной не являются.

## 2. Что исследовано

Исследование опирается на фактический код и frozen evidence проекта, аудиты
OpenDPD и DPD_for_PA, а также на литературную матрицу:

- **89 уникальных проверенных первичных источников**;
- **27 работ изучены подробно** по полному тексту, коду или supplementary
  material;
- 48 источников опубликованы в 2020–2026 годах;
- 18 источников опубликованы в 2024–2026 годах;
- 52 строки матрицы содержат явное evidence с физическим PA;
- у всех 89 строк есть DOI или официальный адрес;
- программная симуляция, replay измеренных данных, физический PA и аппаратный
  результат помечаются разными классами evidence.

Полные данные находятся в:

- `modern_dpd_literature_review.md` — содержательный обзор;
- `literature_matrix.csv` — 33 поля на каждую работу;
- `ai_methods_transfer_to_dpd.md` — проверка AI/LLM/SSM/control идей;
- `residual_observer_and_controller.md` — проект feedback controller;
- `proposed_next_generation_dpd.md` — восемь архитектур;
- `experiment_plan_modern_dpd.md` — gates и последовательность экспериментов;
- `risk_and_evidence_register.md` — что доказано и что не доказано;
- `executive_recommendation.md` — краткая рекомендация.

Результаты разных PA, полос, форм сигналов и способов измерения не объединялись
в общий рейтинг. Опубликованное число из другой статьи не считается ожидаемым
результатом нашего усилителя.

## 3. Фактическая задача проекта

Конечный объект — только быстрый DPD перед физическим PA:

\[
x[n]\;\longrightarrow\;D_\theta\;\longrightarrow\;z[n]
\;\longrightarrow\;PA\;\longrightarrow\;y[n].
\]

Желаемый выход:

\[
y_{\mathrm{target}}[n]=g x[n].
\]

Основная цель — подавить паразитный спектр после физического PA без
недопустимого снижения основной мощности, роста EVM, PAPR или peak drive.

PA model является вспомогательным evaluator. Ограничение, условно названное
«1000 вещественных умножений», относится к рабочему DPD и означает временной
бюджет, а не буквальный счёт только операторов умножения. Поэтому итоговая
стоимость должна включать MUL, ADD, нелинейные операции, LUT, память,
ветвления, state update, latency и sustained throughput.

Формула \(E(f)<10^{-5}\) со слайда не является подтверждённым критерием Huawei.
Порог −50 dB NMSE также не является требованием заказчика.

## 4. Что установлено по текущему проекту

Локально подтверждены кодом, тестами или frozen manifests:

- DPD test получает desired \(x\), а не известный measured \(y_{test}\);
- PA modeling и DPD evaluation разделены;
- train, validation и test разделены и защищены hash manifests;
- spline-memory core причинный, потоковый и фазово-эквивариантный;
- reset/state/frame semantics проверяются тестами;
- selected three-branch spline DPD имеет аналитическую стоимость
  **21 real MUL + 24 ADD + одно вычисление амплитуды на sample** по принятой
  software convention;
- его frozen спектральное улучшение подтверждено только через текущий
  surrogate;
- circular inverse→forward test и R² не доказывают линеаризацию нового desired
  сигнала;
- пара dense reservoirs размера 600 требует порядка 728 тысяч real MUL/sample
  и не соответствует требуемому классу рабочего времени.

Пока не доказаны:

- подавление спектра на нашем физическом PA;
- apples-to-apples превосходство над OpenDPD;
- соответствие временному бюджету на целевой FPGA/DSP/ASIC;
- перенос между power, temperature, waveform и PA instances;
- подавление RF harmonics около \(2f_c\) и \(3f_c\);
- достаточная точность текущего PA evaluator для тонкой оптимизации DPD;
- соответствие Huawei, пока не определена acceptance procedure.

Текущий evaluator отстоит от остаточной ошибки DPD только примерно на
5.5–5.9 dB, тогда как внутренний диагностический gate требует 10 dB. Это не
Huawei threshold, а защита от surrogate exploitation. Gate дальнейшей тонкой
оптимизации через один evaluator поэтому **закрыт**.

## 5. Главный технический вывод из литературы

### 5.1 Что имеет наиболее сильное основание

Для дешёвого широкополосного DPD наиболее устойчивый набор идей образуют:

- локальные spline/CPWL basis с малым числом активных коэффициентов;
- короткие phase-equivariant memory branches;
- GMP/DDR-подобная причинная библиотека;
- разреженный выбор групп признаков;
- linear/ridge/RLS calibration;
- observation receiver и block-rate update;
- отдельная проверка spectrum, main-band power, EVM и peak;
- fixed coefficient bank с атомарным переключением и rollback.

Фундаментальные GMP, spline, piecewise, ILC и decorrelation работы, а также
современные sparse и hardware-oriented DPD дают этому направлению и
физический, и инженерный prior. Однако их числа нельзя напрямую переносить на
наши captures.

### 5.2 Что полезно только при наличии дополнительных данных

- Slow-state conditioning полезен, если controlled captures подтверждают
  зависимость от мощности или температуры.
- Expert bank полезен, если режимы дискретны и повторяемы.
- Hypernetwork/low-rank update полезен, только если коэффициенты разных режимов
  действительно лежат в низкоразмерном подпространстве.
- Tiny neural residual оправдан, только если при равной measured cost он
  превосходит одну-две structured branches.
- Compact SSM оправдан, только если он превосходит FIR и one-pole states.

### 5.3 Что не следует делать первым

- Transformer или attention на каждом sample;
- full Mamba/selective SSM без structured baseline;
- soft mixture нескольких experts на sample;
- большая GRU/LSTM или reservoir;
- RL для задачи, остающейся линейной по коэффициентам;
- unstructured pruning без hardware zero-skipping;
- differentiable spectral tuning через единственный недостаточно точный
  surrogate.

Эти методы либо переносят модный термин без нужного механизма, либо добавляют
memory traffic, nonlinear gates и трудно ограничиваемое состояние в тракте с
сотнями миллионов samples/s.

## 6. Рекомендуемая архитектура

### 6.1 Быстрый основной тракт

Базовый DPD сохраняется в форме

\[
z_0[n]=\sum_{b\in\mathcal B_0}x[n-m_b]
C_b\!\left(|x[n-d_b]|\right),
\]

где \(C_b(r)\) — комплексная кусочно-линейная функция с локальными basis.
У sample активны только два соседних knot. Такая форма удовлетворяет

\[
D(xe^{j\phi})=D(x)e^{j\phi}.
\]

### 6.2 Разреженное residual-расширение

После observer analysis допускается

\[
z[n]=z_0[n]+\sum_{j\in\mathcal S}\sum_k c_{j,k}\phi_{j,k}[n],
\qquad |\mathcal S|\ll |\mathcal D|,
\]

где \(\mathcal D\) — заранее замороженная причинная библиотека, а
\(\mathcal S\) содержит не более нескольких выбранных групп.

Примеры групп:

\[
x[n-m]B_k(|x[n-d]|),
\]

\[
x[n-m]|x[n-d]|^p,
\]

\[
x[n-m]q_\tau[n],
\qquad
q_\tau[n]=\beta_\tau q_\tau[n-1]+(1-\beta_\tau)|x[n]|^2,
\]

где

\[
\beta_\tau=\exp\!\left(-\frac{1}{f_s\tau}\right).
\]

Для thermal claim состояние должно использовать физически осмысленный
\(\tau\), сохраняться между кадрами и проверяться против температуры, drive
power и frame-position artifacts. Сам по себе фильтр огибающей не доказывает
тепловую память.

### 6.3 Ориентировочная стоимость

Current selected core: 21 MUL + 24 ADD + magnitude/sample. Одна дополнительная
common-envelope spline branch ориентировочно добавляет:

- 6 real MUL;
- 8 real ADD;
- 2 LUT/coefficient accesses;
- delay/state storage.

Это аналитическая оценка, а не измеренная target latency. Стоимость
address generation, magnitude implementation, memory ports, saturation и
control должна измеряться отдельно.

## 7. Почему простая корреляция остатка недостаточна

После калибровки observation path измеряется

\[
e[n]=y_{pa,aligned}[n]-g x[n].
\]

Raw score \(\phi^H e\) показывает зависимость, но не является градиентом
качества каскада. Ветвь изменяет вход PA, поэтому для малого изменения

\[
\delta y\approx J_P(z)\,\delta z,
\]

где \(J_P(z)\) — локальная чувствительность PA. Ранжировать advisor-кандидаты
нужно в output space:

\[
\widetilde\Phi_G=J_P(z)\Phi_G,
\]

после QR/SVD residualization относительно уже активных ветвей. \(J_Pv\) можно
получить через независимый differentiable evaluator, конечную разность или
малый безопасный physical probe. Raw correlation остаётся только baseline и
diagnostic.

Это различие критично: сильная raw correlation может рекомендовать признак,
который PA почти не преобразует в нужное спектральное изменение.

## 8. Проект observation receiver и контроллера

Feedback path:

\[
PA\ output\ copy\rightarrow observation\ receiver\rightarrow alignment
\rightarrow residual\ analysis\rightarrow candidate\ bank.
\]

До residual analysis обязательны:

1. integer и fractional delay alignment;
2. complex gain/phase alignment;
3. DC и IQ-imbalance test;
4. de-embedding feedback frequency response;
5. feedback noise/SNR estimate;
6. проверка нелинейности observation receiver;
7. фиксация alignment parameters без подгонки под каждый кандидат.

Контроллер работает по кадрам, а не на каждом sample. Он может быть тяжёлым,
поскольку не входит в fast path.

Режимы зрелости:

| Режим | Что делает | Что менять разрешено |
|---|---|---|
| Observer-only | измеряет residual groups и drift | ничего |
| Advisor | предлагает одну branch group | только offline candidate |
| Shadow | оценивает новый bank | рабочий bank не меняется |
| Safe adaptive | принимает/откатывает bounded update | atomic bank swap |

Новые коэффициенты сначала проходят integrity/version check, bit-true
peak/PAPR/clipping check, независимый evaluator, dummy/low-power test и только
затем controlled physical validation. При любой деградации активируется
последний known-good bank.

## 9. Первый эксперимент максимальной информационной ценности

Первым выполняется **read-only observer diagnostic**. Он не меняет frozen DPD
и поэтому не загрязняет уже полученный результат.

До просмотра остатка validation frames делятся на:

- `observer_discovery`;
- `advisor_select`;
- sealed `advisor_shadow`.

Если существующая validation уже исследовалась, shadow должен быть новым
capture.

На `observer_discovery`:

1. проверить hashes, split, alignment и frame semantics;
2. вычислить aligned complex residual;
3. построить малую заранее замороженную causal feature library;
4. residualize error и candidate output-sensitivity groups через QR/SVD;
5. рассчитать raw и \(J_P\)-transformed scores по целым кадрам;
6. построить whole-frame/capture permutation или circular-shift null;
7. внутри каждой permutation повторять **всю selection procedure**;
8. использовать max-statistic/FDR и block bootstrap confidence intervals;
9. проверить stability между frames/evaluators;
10. проверить segment position, conjugate/IQ и observation artifacts;
11. сформировать максимум одну рекомендацию с MUL/ADD/LUT/state cost;
12. не fit-ить новую DPD на discovery split.

### Gate перехода к advisor

Переход разрешён, только если группа:

- устойчиво выше block-null;
- сохраняет знак/физический смысл по кадрам;
- не объясняется observation path или frame boundary;
- имеет согласованное ranking минимум через два независимых evaluators либо
  через physical probe;
- укладывается в заранее установленную incremental cost budget.

Если gate не пройден, structured DPD не усложняется. Ресурс направляется на PA
evaluator и physical feedback fidelity. Это полезный отрицательный результат.

## 10. Спектральная цель и защита от ложного улучшения

Временная и спектральная training objective может иметь вид

\[
L=\lambda_tL_{time}+\lambda_LL_L+\lambda_RL_R+
\lambda_eL_{EVM}+\lambda_pL_{peak},
\]

но cost модели дискретен и должен использоваться через Pareto/constraint, а не
как фиктивный постоянный differentiable term.

При выборе модели отдельно публикуются:

- adjacent leakage/ACLR слева и справа;
- absolute и dBc band powers;
- integrated OOB;
- main-band power change;
- PSD и worst spectral bin/mask violation;
- EVM и вспомогательный NMSE;
- max \(|z|\), PAPR и clipping count;
- mean, confidence interval, high percentile и worst frame.

Обязательные hard constraints предотвращают три ложных результата:

1. улучшение dBc простым снижением основной мощности;
2. улучшение integrated leakage при появлении узкого spike;
3. улучшение surrogate spectrum ценой опасного peak drive.

Huawei spectral gate остаётся `BLOCKED` до получения точных band definitions,
reference и thresholds.

## 11. Три рекомендуемых направления

### 1. Основное — sensitivity-aware sparse residual spline/GMP

Почему первое:

- минимальное изменение уже проверенного fast path;
- линейная и быстрая калибровка;
- фазовая эквивариантность;
- стоимость растёт по одной branch group;
- отрицательный observer-result позволяет быстро остановиться;
- наилучшее основание для quality/cost Pareto.

Главный риск: ranking обусловлен ошибкой evaluator, а не физическим PA.

### 2. Безопасное запасное — slow-state spline или hard expert bank

Выбор зависит от данных:

- непрерывный подтверждённый drift → slow state;
- дискретные повторяемые operating points → один expert per frame с hysteresis.

Soft per-sample mixture не нужен. При одинаковой topology frame-static смесь
коэффициентов может быть рассчитана вне fast path.

Главный риск: state/router кодирует capture position, а не физический режим.

### 3. Высокорисковое — slow low-rank coefficient generator или compact SSM

Тяжёлая модель работает редко и генерирует только коэффициенты дешёвого DPD:

\[
\theta_{t+1}=\Pi_\Theta\left(\theta_0+U h_\varphi(s_t)\right).
\]

Альтернатива — 4–8 устойчивых diagonal states с quantized pole margin и
ограниченным состоянием.

Главный риск: обычный ridge, coefficient LUT или one-pole bank окажется проще,
быстрее и надёжнее.

## 12. Hardware и fixed-point решение

Для каждого кандидата нужен единый ledger:

- real MUL и ADD;
- magnitude/sqrt/activation;
- LUT и memory accesses;
- comparisons/branches;
- coefficient и state memory;
- accumulator width и saturation;
- critical recurrence path;
- samples/s, batch=1 latency, sustained throughput;
- p50, p95, p99.9 и worst measured time;
- chunk equivalence и causality;
- FP32, FP16 diagnostic и bit-true 16/14/12-bit.

Сравнение с «1000 умножениями» возможно только после получения эталонного ядра
Huawei: target, clock, bit width, parallelism, load/store convention и deadline.
До этого analytical MUL count — полезный proxy, но не acceptance result.

## 13. Безопасность и смена режима

Drift detector должен наблюдать не одну метрику, а набор:

- distribution shift амплитуды и средней мощности;
- residual group scores;
- left/right leakage, EVM и worst bin;
- peak/PAPR/clipping;
- disagreement независимых PA evaluators;
- feedback SNR/alignment error;
- temperature/bias, если доступны.

При смене режима:

1. заморозить online update;
2. перейти на conservative known-good bank;
3. ограничить drive amplitude;
4. собрать новый sealed calibration block;
5. проверить bounded candidate в shadow;
6. применить атомарно или отклонить;
7. сохранить audit log и возможность rollback.

## 14. Научная новизна

Уже известны:

- decorrelation и residual correlation;
- OMP/DOMP/group-LASSO и sparse pruning;
- spline/CPWL/GMP branches;
- adaptive/closed-loop DPD;
- operating-point switching и coefficient prediction;
- safe RL с threshold/recovery;
- teacher–student и meta-learning DPD.

Поэтому честный потенциальный вклад находится не в названии одного блока, а в
полной проверяемой системе:

1. local phase-equivariant spline core;
2. PA-sensitivity-aware residual group score;
3. cross-frame block-null significance;
4. cost-aware selection нескольких branches;
5. независимый spectral shadow capture;
6. hard main-band/EVM/peak/PAPR/worst-bin gates;
7. transactional apply/rollback;
8. physical drift validation под measured timing constraint.

До сравнения с DOMP, decorrelation, exhaustive selection, fixed GMP/spline и
safe adaptive prior art это следует называть **новой комбинацией известных
методов**, а не доказанным новым методом.

## 15. Что необходимо получить от Huawei и стенда

### От Huawei

1. Что считается паразитной составляющей: adjacent regrowth, discrete IM,
   RF \(2f_c/3f_c\) или mask violation.
2. Точные band edges, RBW/VBW, window, detector и averaging.
3. Reference: dBm, dBc или improvement относительно no-DPD.
4. Minimum suppression и допустимые main-band/EVM/peak/PAPR изменения.
5. Target sample rate, clock, FPGA/DSP/ASIC и numeric format.
6. Эталонное «1000-MUL» ядро и правила учёта памяти/параллелизма.
7. Максимальная latency, sustained throughput и update cadence.
8. Observation receiver bandwidth, ADC rate, SNR и допустимые probes.
9. Рабочие диапазоны power, temperature, carrier, bandwidth и waveform.

### От физического стенда

1. Парные no-DPD/DPD captures при одинаковой output power.
2. Повторы и confidence intervals.
3. Логи temperature, bias, drive/output power и waveform.
4. Калибровка observation path.
5. Controlled dwell и transition captures.
6. RF bandwidth, достаточная для заявляемых harmonics.
7. Несколько power/temperature points и, позднее, PA instances.

## 16. Окончательное решение

Сейчас разрешён только Stage 1: read-only observer analysis. Новую ветвь,
hypernetwork, SSM или neural residual нельзя считать обоснованной до него.

Если observer обнаруживает устойчивую output-sensitivity residual group, первой
реализуется **одна sparse spline/GMP branch**, после чего выполняется
independent spectral shadow validation. Если такой структуры нет или ranking
меняется между evaluators, модель не усложняется: сначала повышается fidelity
PA evaluator и выполняется physical PA measurement.

Именно этот путь максимизирует шанс получить реальное подавление паразитного
спектра, сохраняя дешёвый фиксированно-точечный fast path, и одновременно
минимизирует риск «победить» только ошибку программной модели усилителя.

До физического apples-to-apples эксперимента корректная формулировка результата:

> Проект имеет воспроизводимый low-cost DPD baseline и обоснованный протокол
> поиска следующей полезной ветви. Превосходство над OpenDPD, соответствие
> Huawei и физическая линеаризация пока не доказаны.
