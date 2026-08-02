# Кандидаты следующего поколения DPD

Дата: 2026-07-30
Статус: архитектурный design study; реализации и новые DPD-результаты не
создавались.

## 1. Исходная точка и конвенция стоимости

Текущий выбранный low-cost baseline проекта:

\[
z_0[n]=\sum_{b=1}^{B}x[n-m_b]C_b(|x[n-d_b]|),
\qquad B=3,\quad (m_b,d_b)\in\{(0,0),(1,0),(2,0)\}.
\]

Каждый \(C_b\) — complex linear spline. На отсчёте активны два соседних
control points. Текущий аудированный float schedule для трёх ветвей:

- 21 real MUL;
- 24 real ADD;
- 1 `sqrt`;
- 6 coefficient LUT reads;
- 144/48 real coefficient values для текущих DPA/APA конфигураций;
- 4 real state values;
- фазовая эквивариантность;
- causal streaming и chunk equivalence подтверждены тестами.

Это аналитический operation count, не аппаратная latency. Ограничение Huawei
трактуется как временной бюджет, приблизительно эквивалентный 1000 real
multiplications, поэтому для каждого кандидата ниже отдельно учитываются
MUL, ADD, nonlinear functions, LUT/memory, state и critical path.

Принята конвенция

\[
1\ \text{complex MUL}=4\ \text{real MUL}+2\ \text{real ADD}.
\]

Gauss 3-MUL schedule рассматривается только как отдельная hardware ablation,
поскольку он увеличивает число сложений и динамический диапазон.

## 2. Общие требования ко всем кандидатам

Кандидат не переходит к test, пока не выполнены:

- causal desired-input path;
- fixed train/validation/test split;
- независимый frozen PA evaluator или physical PA;
- одинаковый spectral evaluator;
- peak/PAPR/main-band constraints;
- streaming chunk equivalence;
- bit-accurate 16/14/12-bit replay;
- аналитический count и измеренный target timing;
- отсутствие выбора topology по test.

Сложность добавляется только при доказанном validation improvement.

## 3. A — Structured spline-memory + sparse residual branches

### 3.1 Уравнения

\[
z[n]=z_0[n]+\sum_{j\in\mathcal S}\sum_{k=1}^{K_j}
      c_{j,k}\phi_{j,k}[n],
\qquad |\mathcal S|\le B_\max,
\]

где causal phase-equivariant library содержит группы:

\[
\phi^\mathrm{spline}_{m,d,k}[n]
=x[n-m]B_k(|x[n-d]|),
\]

\[
\phi^\mathrm{GMP}_{m,d,p}[n]
=x[n-m]|x[n-d]|^{p-1},
\]

\[
\phi^\Delta_{m,d}[n]
=x[n-m]\bigl(|x[n-d]|^2-|x[n-d-1]|^2\bigr).
\]

Spline group включает все control points одной ветви; отдельные knots не
выбираются как независимые hardware features. Отдельного trainable scalar
\(a_j\) поверх trainable spline control points нет: иначе появляется
масштабная неидентифицируемость.

### 3.2 Выбор и обучение

1. observer residualizes feature groups относительно active dictionary;
2. raw correlation используется как diagnostic baseline, а advisor ранжирует
   группы после преобразования через PA sensitivity
   \(J_P(z)\Phi_G\), полученную JVP/finite difference независимого evaluator
   или безопасным малым probe;
3. fit выполняется complex ridge/QR на train/calibration;
4. сравниваются correlation-guided, exhaustive small search, OMP и
   group-LASSO;
5. модель принимается только после independent spectral shadow validation.

### 3.3 Стоимость

При уже вычисленной envelope amplitude новая spline branch с существующим
`envelope_delay` добавляет:

- 6 real MUL;
- 8 real ADD с учётом complex accumulation;
- 2 complex coefficient reads = 4 real reads;
- 2 LUT accesses.

Новый уникальный `envelope_delay` дополнительно требует 3 real MUL, 2 ADD,
1 `sqrt`, interval comparisons и delayed-state storage. Поэтому выбирать
несколько signal delays с общей envelope выгоднее, чем много независимых
envelope taps.

Для \(B_\max=4\) worst incremental arithmetic относительно baseline —
приблизительно 24 MUL + 32 ADD, если envelope общий. Это далеко ниже
арифметического proxy 1000, но фактический timing определяется LUT/memory и
pipeline.

### 3.4 Оценка

- Причинность: да.
- Fixed-point: структура благоприятна для regular LUT/MAC datapath; результат
  требует bit-accurate и target-timing проверки.
- Калибровка: linear-in-parameters после freeze словаря.
- Адаптация: block QR/RLS/regularized LS; update не на sample path.
- Ожидаемое преимущество: минимальная добавочная стоимость именно для
  обнаруженной residual structure.
- Главный риск: корреляция вызвана ошибкой evaluator/feedback, а не PA.
- Новизна: гипотеза о полезной системной комбинации; correlation/OMP и safety
  primitives сами по себе известны, а вклад требует physical comparison.

**Приоритет: основной путь.**

## 4. B — Spline/GMP со slow-state conditioning

### 4.1 Уравнения

\[
q_\ell[n]=\beta_\ell q_\ell[n-1]+(1-\beta_\ell)s[n-1],
\]

где дешёвый proxy использует \(s[n]=|x[n]|^2\), а physically closer
вариант — уже сформированный drive \(s[n]=|z[n]|^2\), measured output power
либо sensor statistic. Задержка на один отсчёт исключает algebraic loop.

Минимальная conditioned spline:

\[
C_b(r,q)=C_b^{(0)}(r)+(q-q_0)C_b^{(1)}(r),
\]

\[
z[n]=\sum_bx[n-m_b]C_b(|x[n-d_b]|,q[n]).
\]

Набор \(\beta_\ell\) должен соответствовать физически наблюдаемым time
constants:

\[
\beta_\ell=\exp[-1/(f_s\tau_\ell)].
\]

Для thermal hypothesis предпочтительны actual drive \(z\), measured output
power и temperature/bias sensors. При очень больших \(\tau_\ell\) состояние
обновляется на frame/control rate и не сбрасывается на обычной границе кадра;
его reset semantics задаются отдельно. Добавлять множество произвольных
\(\beta\) нельзя: состояния становятся коллинеарны и плохо
идентифицируемы.

### 4.2 Стоимость

Для proxy \(|x|^2\) общий квадрат уже нужен baseline для amplitude и
переиспользуется; только при этом итог ниже равен 35 MUL. Вариант
\(|z|^2\) добавляет ещё 2 MUL + 1 ADD после формирования output и влияет на
следующий sample. Одно состояние: 2 MUL + 1 ADD, state read/write,
accumulator width и saturation policy.
Conditioned branch требует вторую spline interpolation, умножение
complex correction на real \(q-q_0\) и сложение:

- примерно 4 дополнительных real MUL на branch;
- примерно 6 дополнительных ADD на branch;
- ещё 2 complex coefficient reads на активный interval;
- один state value на \(\beta\).

Для baseline из трёх ветвей с одним \(q\): ориентир 35 MUL/sample до
hardware-specific addressing, то есть всё ещё очень дёшево.

### 4.3 Обучение и адаптация

При fixed \(\beta\) модель linear-in-parameters и обучается complex ridge.
\(\beta\) выбирается только на validation из малого логарифмического grid.
Альтернатива — оценить time constants по decay residual autocorrelation.

- Причинность: да.
- Fixed-point: структура благоприятна; для \(\beta=1-2^{-s}\) update
  реализуется shift/add, но state bound, rounding, saturation и limit cycles
  всё равно проверяются bit-accurate.
- Ожидаемое преимущество: thermal/bias/envelope memory.
- Главный риск: доступный capture не содержит смены slow state, и branch
  подгоняет position in frame.
- Gate: стабильная residual correlation с \(xq_\beta\) после контроля frame
  position и на разных captures.

**Приоритет: второй путь только при evidence медленного состояния.**

## 5. C — Банк маленьких экспертов с frame-rate routing

### 5.1 Уравнения

Имеются \(E\) coefficient banks одинаковой дешёвой структуры:

\[
z[n]=D_{\theta_{r_t}}(x)[n],\qquad n\in\text{frame }t,
\]

Router работает один раз на кадр:

\[
r_t=\arg\max_e\left(
a_e^T[s_t^\mathrm{known\ before\ frame},\,s_{t-1}^\mathrm{feedback}]+b_e
\right),
\]

где metadata текущего кадра разрешены только если они заранее известны
scheduler-у; measured residual текущего кадра влияет не раньше следующего.
Router statistics rotation-invariant, а все banks имеют одну
phase-equivariant topology.

Mixture внутри отсчёта не рекомендуется: он вычисляет несколько DPD и
умножает стоимость на число heterogeneous experts при sample-rate routing.
Если topology одинакова и mixture weights постоянны весь кадр, можно заранее
смешать коэффициенты и выполнить одну модель; это отдельный coefficient-bank
interpolation baseline. Hard routing сохраняет стоимость одного baseline.

### 5.2 Switching safety

- выбор только на границе кадра;
- hysteresis margin;
- минимальное dwell time;
- коэффициенты обоих banks доступны до switch;
- для stateless одинаковых spline banks предпочтительна интерполяция
  coefficients на границе кадра;
- output cross-fade выполняет два experts и допускается только после
  peak/timing analysis;
- неизвестный режим направляется в conservative fallback.

### 5.3 Стоимость

Для \(E=4\), \(S=6\) linear router — 24 MUL, примерно 24 ADD и 3 argmax
comparisons **на кадр**, а не на sample. Отдельно считаются feature
statistics, coefficient transfer и optional interpolation. Рабочий per-sample
count равен одному spline/GMP expert. Coefficient memory возрастает в \(E\)
раз. Для stateful expert требуется один заранее выбранный policy: общий state,
reset/warm-up, проверенный state transfer либо обновление всех states
(последнее умножает recurrence cost на \(E\)). Для stateless spline banks
этой проблемы нет.

- Причинность: да.
- Fixed-point: структура благоприятна; требуется bit-accurate switching test.
- Адаптация: coefficient-only recalibration per regime.
- Ожидаемое преимущество: discontinuous operating modes лучше одной smooth
  модели.
- Главный риск: переключательная нестабильность и недостаток labeled regimes.
- Сравнение: обязательно против обычной coefficient table по power/temperature;
  «MoE» не даёт преимущества сам по себе.

**Приоритет: безопасный резервный путь при наличии нескольких режимов.**

## 6. D — Slow hypernetwork / low-rank coefficient generator

### 6.1 Уравнения

Deployed DPD остаётся структурированным:

\[
z[n]=D_{\theta_t}(x)[n].
\]

Раз в кадр или реже:

\[
h_t=H_\phi(s_{\le t}),\qquad
\tilde\theta_{t+1}=\Pi_\Theta\left(\tilde\theta_0+Uh_t\right),
\]

где для \(N\) complex coefficients
\(\tilde\theta=[\Re\theta;\Im\theta]\in\mathbb R^{2N}\),
\(U\in\mathbb R^{2N\times r}\), \(r\ll2N\), а \(\Pi_\Theta\) ограничивает
коэффициенты, drive gain и smoothness. Feedback кадра \(t\) влияет не раньше
\(t+1\); bank сменяется атомарно. Statistics генератора rotation-invariant,
а структура \(D_\theta\) phase-equivariant.

Это low-rank adaptation коэффициентов, а не low-rank sample-level neural
layer. Если \(C\) и так мало, LoRA не экономит deployed inference; её смысл
только в ограничении безопасного drift subspace.

### 6.2 Стоимость

- fast path: ровно стоимость `D_theta`;
- slow path: стоимость \(H_\phi\), \(2Nr\) MUL,
  \(2N(r-1)\) ADD, projection, coefficient writes и bank transfer на update;
- memory: \(\theta_0\), \(U\), параметры \(H_\phi\), rollback bank;
- critical path DPD не меняется.

### 6.3 Обучение

Нужны controlled captures по power/temperature/bandwidth. Сначала для каждого
режима независимо оценивается \(\theta^\star_c\), затем SVD проверяет, является
ли variation действительно low-rank. Только после этого обучается generator.

- Причинность: fast path да; update delayed.
- Fixed-point: fast path структурно благоприятен, но требует bit-accurate
  проверки; generator может работать на CPU/DSP.
- Ожидаемое преимущество: быстрая coefficient adaptation с несколькими
  calibration samples.
- Главный риск: extrapolation в неизвестном режиме создаёт опасный drive.
- Gate: generator должен превосходить nearest-bank interpolation и direct
  limited-sample ridge на held-out condition.

**Приоритет: высокий исследовательский риск; не первый эксперимент.**

## 7. E — Structured DPD + tiny phase-equivariant residual network

### 7.1 Уравнения

Сеть получает только rotation-invariant causal features

\[
u[n]=[|x[n]|^2,|x[n-1]|^2,\Delta|x[n]|^2,q_\beta[n],\ldots].
\]

Она выдаёт complex scalar correction:

\[
h[n]=\sigma(W_1u[n]+b_1),
\qquad
c[n]=W_2h[n]+b_2\in\mathbb R^2,
\]

\[
z[n]=z_0[n]+x[n]\left(c_R[n]+jc_I[n]\right).
\]

Так сохраняется глобальная phase-equivariance. Произвольная сеть над I/Q без
tying не используется.

### 7.2 Пример стоимости

Для \(F=6,H=8\):

- first layer: 48 MUL;
- output layer: 16 MUL;
- final complex multiply: 4 MUL;
- итого network core около 68 MUL/sample **плюс** создание
  \(|x|^2,\Delta|x|^2,q_\beta\), state read/write и addressing;
- ADD: примерно 68 плюс accumulation;
- 8 activation evaluations;
- параметры: \(6\cdot8+8+8\cdot2+2=74\) real;
- 74 parameter reads/sample без caching/packing assumptions;
- state задаётся только feature delays и \(q_\beta\).

PWL/hard-tanh предпочтительнее sigmoid/tanh для fixed-point.

### 7.3 Обучение

Сначала freeze structured path, затем fit residual на train. Joint fine-tuning
разрешается отдельной ablation. Loss обязан включать spectral, main-band,
EVM и peak constraints. Сеть сравнивается с добавлением structured ветвей
той же измеренной latency.

- Причинность: да при causal features.
- Fixed-point: структурно возможна при PWL activation и QAT; пригодность
  требует bit-accurate и target-timing проверки.
- Ожидаемое преимущество: compact approximation остатка, не покрытого
  словарём.
- Главный риск: surrogate exploitation и дорогая activation/memory traffic.
- Gate: residual network должна превзойти sparse branches при равном timing.

## 8. F — Lightweight diagonal state-space DPD

### 8.1 Уравнения

Полные S4/Mamba блоки слишком тяжелы для sample-rate порядка сотен MS/s.
Проверяется минимальная diagonal state model:

\[
u[n]=[|x[n]|^2,\ |x[n]|^2-|x[n-1]|^2]^T,
\]

\[
h[n]=a\odot h[n-1]+Bu[n],
\]

\[
c[n]=Ch[n]+Du[n],
\qquad
z[n]=z_0[n]+x[n](c_R[n]+jc_I[n]).
\]

\(h\) real, \(c\) complex pair. Для точной арифметики требуется
\(|a_i|\le1-\epsilon\), bounded input и рассчитанный state bound. При
quantization pole margin проверяется повторно: sigmoid на float training не
гарантирует, что coefficient не округлится в \(a_i=1\).

### 8.2 Стоимость

Для \(H=8,F=2\):

- diagonal recurrence: 8 MUL;
- input projection: 16 MUL;
- complex-pair output: 16 MUL;
- direct term: 4 MUL;
- multiplication by \(x\): 4 MUL;
- всего около 48 MUL/sample плюс structured baseline;
- около 38 core ADD, 44 coefficient reads, 8 state reads и 8 state writes
  до учёта invariant-feature generation;
- state: 8 real values;
- nonlinear operations в deployed recurrence отсутствуют.

Можно выбрать \(a_i=1-2^{-s_i}\), заменив только diagonal recurrence MUL на
shift/add; матрицы \(B,C,D\) всё ещё требуют MAC. Критический путь проходит
через recurrent update, поэтому общий MUL count не доказывает initiation
interval.

### 8.3 Оценка

- Причинность: да.
- Fixed-point: структура благоприятна, но обязательны quantization-aware pole
  margin, accumulator-width/state-bound analysis, saturation/rounding policy
  и zero-input limit-cycle test.
- Ожидаемое преимущество: длинная память без длинного FIR/GRU gates.
- Главный риск: линейное invariant-state пространство недостаточно, а full
  selective SSM разрушает cost advantage.
- Новизна: инженерная адаптация state-space principle; не «Mamba-DPD» без
  доказательства.
- Gate: превосходство state-conditioned spline и short FIR при равном timing.

## 9. G — Teacher–student distillation

### 9.1 Уравнения

Teacher формирует корректирующее действие \(z_T[n]\) через physical ILC,
heavy OpenDPD model либо явный optimizer/DLA через high-fidelity evaluator.
Сам evaluator без optimizer не задаёт \(z_T\). Student:

\[
\min_\theta
\lambda_z\|D_\theta(x)-z_T\|^2+
\lambda_y L_\mathrm{cascade}(P(D_\theta(x)),gx)+
\lambda_\mathrm{spec}L_\mathrm{spectral}+
\lambda_\mathrm{peak}L_\mathrm{peak}.
\]

Student — spline/sparse GMP/малый SSM. Teacher никогда не попадает в
deployment path.

### 9.2 Стоимость и риски

- deployed cost равен student;
- calibration cost может быть высокой и публикуется отдельно;
- ILC teacher требует physical feedback и повторных воспроизведений;
- surrogate teacher передаёт student свои ошибки;
- imitation \(z_T\) не гарантирует одинаковый PA output вне training
  distribution.

Gate: distilled student сравнивается с прямым complex ridge/DLA student на
том же capture, с одинаковой architecture и без test tuning. Causal student
проверяется на новом waveform/capture; noncausal teacher допустим только если
student inputs остаются строго causal.

## 10. H — Frequency-selective sparse nonlinear FIR

### 10.1 Уравнения

\[
\phi_j[n]=x[n]|x[n]|^{p_j-1},
\qquad
z[n]=g_0x[n]+\sum_{j\in\mathcal S}\sum_{\ell=0}^{L_j-1}
h_{j,\ell}\phi_j[n-\ell].
\]

Это short FIR на нескольких выбранных nonlinear basis streams. Branch можно
выбирать по sub-band residual correlation, если Huawei определит конкретные
нежелательные полосы.

Чтобы избежать линейной зависимости, \(p_j=1,\ell=0\) исключается из суммы:
линейный current-sample term представлен только \(g_0x[n]\).

### 10.2 Стоимость

Каждый complex FIR tap — 4 MUL + 4 ADD с accumulation по принятой convention;
basis magnitude powers считаются отдельно и должны переиспользоваться между
taps. Для двух nonlinear streams по четыре taps — не менее 32 real MUL только
на FIR, плюс basis generation.

- Причинность: да.
- Fixed-point: regular MAC структурно благоприятен; bit-accurate и target
  verification обязательны.
- Ожидаемое преимущество: направленная компенсация frequency-dependent memory.
- Риск: улучшение одной полосы создаёт spike в другой; нельзя оптимизировать
  только selected bins.

## 11. Спектральная целевая функция

Для differentiable evaluator time/spectral terms дают training objective, но
safety terms являются constraints, а не взаимозаменяемыми штрафами:

\[
\begin{aligned}
L_\mathrm{train}={}&\lambda_tL_\mathrm{time}
+\lambda_LL_\mathrm{left}
+\lambda_RL_\mathrm{right}
+\lambda_oL_\mathrm{OOB}
+\lambda_eL_\mathrm{EVM},
\end{aligned}
\]

\[
P_\mathrm{main}\ge P^\star_\mathrm{main}-\delta_\mathrm{main},\quad
\max|z|\le z_\mathrm{safe},\quad
\mathrm{PAPR}(z)\le p_\mathrm{safe},\quad
\mathrm{PSD}(f)\le M(f).
\]

Band power считается через windowed FFT с frozen masks. Чтобы DPD не улучшал
dBc простым снижением полезной мощности:

\[
L_\mathrm{main}=
\left(P_\mathrm{main}(y)-P^\star_\mathrm{main}\right)^2
\]

либо, предпочтительно, main-band power задаётся явным constraint.

Для защиты от редких плохих кадров используются:

\[
L_\mathrm{tail}=\operatorname{CVaR}_{0.95}
\{L_{\mathrm{adj},t}\},
\]

а для узких spikes:

\[
L_\mathrm{mask}=\max_{t,f}
\operatorname{softplus}(\mathrm{PSD}_t(f)-M(f)).
\]

Отдельно публикуется hard maximum по кадрам: 95-й процентиль не называется
worst case. \(L_\mathrm{cost}\) постоянен внутри fixed topology и потому не
даёт полезного gradient; стоимость участвует в дискретном выборе/Pareto
frontier между architectures.

Один FFT loss недостаточен:

- window leakage может быть принята за PA leakage;
- mean PSD скрывает worst frames;
- dBc улучшается при падении main-band power;
- оптимизация только соседних полос переносит энергию дальше;
- surrogate gradient может вести к неphysical adversarial waveform.

Практический порядок:

1. time-domain warm start;
2. frozen physical band masks;
3. constrained spectral fine-tune только на train;
4. validation Pareto frontier;
5. independent evaluator;
6. physical PA.

## 12. Сводный shortlist

| Кандидат | Fast-path MUL, ориентир | State | Coefficient memory | Calibration | Fixed-point | Научный статус | Решение |
|---|---:|---:|---:|---|---|---|---|
| A sparse residual branches | 21 + 6 на common-envelope branch | delays | линейно с branches | ridge/OMP/group-LASSO | structurally favorable; unverified | гипотеза о новой safe combination | первый |
| B slow-state spline | около 35 с reused \(|x|^2\), либо около 37 с \(|z|^2\) | 1+ | примерно 2× spline coeff | ridge при fixed beta | structurally favorable; unverified | инженерное расширение | после evidence |
| C frame expert bank | стоимость одного expert | router/frame | \(E\times\) | per-regime fit | structurally favorable; unverified | известная идея, полезная система | backup |
| D hypernetwork/low-rank | base DPD | slow stats | generator + basis | meta/offline | fast path needs verification | высокий риск | позже |
| E tiny residual NN | base + ~68 core + features/state | small | ~74 real example | SGD/QAT | requires QAT/bit-true proof | известный residual hybrid | только ablation |
| F diagonal SSM | base + ~48 core + features | 8 real example | ~44+ real | ridge/gradient | requires state-bound proof | адаптация SSM principle | research |
| G distillation | student cost | student | student | teacher expensive | зависит от student | известный training mechanism | useful option |
| H nonlinear FIR | base + ≥32 example | FIR | taps | LS/adaptive | structurally favorable; unverified | классическое FS-DPD | targeted backup |

## 13. Рекомендация

Основной путь — кандидат A: observer-only residual analysis, затем одна
cost-aware sparse branch и безопасная spectral validation. Он наиболее
естественно продолжает текущий проверенный phase-equivariant spline core,
сохраняет линейную калибровку и regular hardware datapath.

Безопасный запасной путь — C либо B:

- C, если есть дискретные, повторяемые operating points;
- B, если residual analysis показывает непрерывное медленное состояние.

Высокорисковый путь — D/F: slow coefficient generator или compact diagonal
state. Для D нужны controlled multi-condition captures. F можно сначала
сравнить с FIR/one-pole state на существующем capture, но robustness claim
также требует нескольких режимов. Full Transformer, full Mamba и
sample-level soft MoE не
рекомендуются: их routing/gating/state/memory cost не соответствует
sample-rate DPD и не имеет доказанного преимущества над структурированным
baseline в текущем apples-to-apples pipeline.
