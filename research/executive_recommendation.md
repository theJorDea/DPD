# Итоговая рекомендация: следующий шаг DPD

Дата: 2026-07-30
Статус: рекомендация по результатам аудита кода, frozen evidence и современной
литературы. Новое превосходство экспериментально не заявляется.

## Решение

Основной исследовательский путь:

> **phase-equivariant spline-memory DPD + correlation-guided sparse residual
> branches + независимая spectral shadow validation + accept/rollback.**

Это не предложение сразу построить ещё одну большую модель. Первый шаг —
observer-only анализ frozen validation residual. Он ничего не меняет в
рабочем DPD и отвечает, существует ли воспроизводимая пропущенная структура.

## Почему именно этот путь

1. Текущий spline core уже causal, streaming-compatible, fixed-point-oriented
   и имеет аудированную стоимость 21 real MUL + 24 ADD + `sqrt` на sample.
2. Physical spline/SPH/SMP, GMP/DOMP и decorrelation literature показывает,
   что локальные basis и sparse branches дают сильный quality/cost trade-off.
3. Дополнительная spline branch с уже общей envelope добавляет ориентировочно
   только 6 MUL + 8 ADD и 2 LUT accesses.
4. Complex ridge/QR calibration быстрее и интерпретируемее neural retraining.
5. Observer остаётся вне fast path, поэтому может использовать более тяжёлую
   статистику.
6. Negative result тоже ценен: если residual correlations неустойчивы,
   проект не тратит время на ненужную сеть.

## Три наиболее перспективных подхода

### 1. Основной: sparse residual spline/GMP

\[
z[n]=z_0[n]+\sum_{j\in\mathcal S}\sum_kc_{j,k}\phi_{j,k}[n],
\qquad |\mathcal S|\ \text{мало}.
\]

Raw correlation используется только как diagnostic baseline. Ветвь
ранжируется по QR/SVD-whitened partial score после преобразования через PA
sensitivity \(J_P(z)\Phi_G\), проверяется по кадрам и сравнивается с
random/exhaustive/DOMP/group-LASSO. Применение разрешается только после
отдельной spectral validation с peak, PAPR, main-band и worst-bin gates.

Ожидаемая ценность: наибольшая вероятность улучшить leakage при минимальном
росте рабочего времени.

### 2. Безопасный резерв: slow-state spline или frame-rate expert bank

Если residual устойчиво связан с

\[
q_\beta[n]=\beta q_\beta[n-1]+(1-\beta)|x[n]|^2,
\]

используется state-conditioned spline. Если режимы дискретны, один из
нескольких coefficient banks выбирается раз в кадр с hysteresis.

Ожидаемая ценность: адаптация к power/temperature/bandwidth без нескольких
experts на sample.

Ограничение: нужны controlled multi-condition captures. APA и APA_B с
неизвестными изменившимися осями недостаточны для thermal claim.

### 3. Высокорисковый: slow coefficient generator / compact diagonal state

Hypernetwork или low-rank update генерирует только коэффициенты дешёвого DPD:

\[
\theta_t=\Pi_\Theta(\theta_0+UH_\phi(s_t)).
\]

Альтернатива — 4–8 устойчивых diagonal states. Оба пути сохраняют маленький
fast path, но требуют доказать преимущество над direct ridge, coefficient LUT
и one-pole states.

Ожидаемая ценность: самостоятельная научная тема быстрой адаптации.

Риск: высокая вероятность, что обычный bank/ridge окажется лучше и безопаснее.

## Что не рекомендуется первым

- full Transformer или attention по samples;
- full Mamba/selective SSM без дешёвого linear-state baseline;
- soft mixture нескольких DPD experts на каждый sample;
- большая GRU/LSTM только потому, что она точнее surrogate;
- reservoir computing: dense recurrence уже существенно превышает бюджет;
- RL для коэффициентов при наличии linear regression/closed-loop methods;
- spectral fine-tuning через текущий единственный surrogate;
- unstructured sparsity без zero-skipping target engine.

## Ближайший эксперимент

До просмотра residual целые frozen validation frames разделяются на
`observer_discovery`, `advisor_select` и запечатанный `advisor_shadow`. Если
validation уже исследовалась, независимый shadow требует нового capture.

На `observer_discovery`, без изменения DPD:

1. вычислить aligned residual;
2. сформировать малую causal phase-equivariant feature library;
3. residualize candidates и error через QR/SVD active subspace;
4. посчитать raw и PA-sensitivity-transformed group scores по кадрам;
5. построить whole-frame/capture permutation null, повторяя всю selection
   procedure, и max-statistic/FDR;
6. проверить frame-position и widely-linear artifacts;
7. выбрать не более одной **рекомендации**, не fit;
8. указать её incremental MUL/ADD/LUT/state cost;
9. сравнить ranking между доступными independent evaluators.

Gate:

- стабильная feature group;
- score выше null;
- одинаковый физический смысл по кадрам;
- не observation artifact;
- доступен независимый evaluator/physical PA для следующего шага.

При отсутствии последнего условия результат — только observer hypothesis;
переход к advisor закрыт.

## Что уже доказано

- правильный desired-input DPD path;
- разделение PA evaluator и DPD;
- deterministic splits/hash protection;
- causal streaming spline core;
- аналитическая стоимость;
- surrogate-only spectral improvement frozen spline DPD;
- ошибки кругового теста Егора и неприемлемая стоимость dense reservoir;
- отсутствие достаточного evidence для claims о physical/Huawei/OpenDPD
  superiority.

## Что не доказано

- physical spectral suppression текущей DPD;
- apples-to-apples превосходство над OpenDPD;
- перенос на power/temperature/waveform/PA instance;
- соответствие target timing;
- подавление RF harmonics \(2f_c/3f_c\);
- научная новизна residual correlation как таковой;
- достаточная точность текущего PA evaluator для дальнейшей fine optimization.

## Нужные ответы и данные

От Huawei:

- exact unwanted spectral components и band edges;
- absolute/dBc reference и minimum improvement;
- main-band/EVM/peak/PAPR limits;
- RBW/VBW/window/detector/averaging;
- target sample rate, clock, architecture и numeric format;
- exact reference kernel, соответствующий «1000 умножений»;
- допустимая latency и update frequency;
- observation receiver specification.

Пока эти ответы не получены, numerical spectral/timing acceptance gates имеют
статус `BLOCKED`; внутренние thresholds не называются Huawei criteria.

От physical stand:

- paired no-DPD/DPD captures;
- logged output power, temperature, bias и waveform;
- repeats и transition captures;
- observation-path calibration;
- RF bandwidth для harmonic claims.

## Оценка научной новизны

Residual correlation, decorrelation learning, OMP/DOMP, sparse pruning,
piecewise DPD, expert selection и coefficient prediction уже опубликованы.
Safe-RL DPD с ACLR threshold и recovery/rollback также опубликован
([Spano et al., 2025](https://doi.org/10.3390/s25196102)).

Потенциальный вклад — только их строгая комбинация:

- local phase-equivariant spline main path;
- cross-frame/null-tested residual selection;
- cost-aware group choice;
- independent spectral shadow score;
- safety constraints;
- transactional apply/rollback;
- physical drift validation под hard measured timing budget.

До сравнения с ближайшими baselines это называется **новой комбинацией
известных методов**, не доказанной новой архитектурой.

## Критерий остановки

Если advisor не выбирает ветвь стабильнее random, не приближается к exhaustive
best и меняет ranking между evaluators, DPD не усложняется. Следующий ресурс
направляется на high-fidelity evaluator и physical PA measurement.

Такой outcome не является неудачей: он предотвращает оптимизацию ошибок
surrogate и сохраняет главную цель — реальное подавление паразитного спектра
при дешёвом рабочем DPD.
