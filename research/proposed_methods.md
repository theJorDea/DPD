# Предлагаемые low-complexity DPD-кандидаты

Дата: 2026-07-28. Это архитектурный shortlist, а не заявление о
превосходстве. Любой кандидат попадает на Pareto frontier только после
одинакового dataset, PA/checkpoint, split, gain/alignment и RF evaluator.

## 1. Общая постановка

Для desired complex baseband \(x[n]\) требуется

\[
x[n]\xrightarrow{D}z[n]\xrightarrow{P}y[n]\approx g x[n].
\]

Основной test path всегда начинается с \(x_{\rm test}\). Нормированный measured
output \(y_{\rm test}/g\) разрешён как input только для ILA-fit или отдельно
помеченного postdistorter diagnostic; он не является эксплуатационным DPD test.

Фазово-эквивариантные модели предпочтительны, если измеренный feedback path не
показывает значимый IQ imbalance:

\[
D(xe^{j\phi})=D(x)e^{j\phi}.
\]

Если IQ imbalance существует, его следует сначала измерить и либо
скомпенсировать отдельным widely-linear блоком, либо явно сравнить constrained и
unconstrained варианты. Нарушать symmetry двумя произвольными I/Q-моделями без
такого evidence не следует.

## 2. Кандидат 0: memoryless complex linear spline

\[
z[n]=x[n]C(|x[n]|),
\]

\[
C(r)=(1-t)c_k+t c_{k+1},\qquad r\in[r_k,r_{k+1}].
\]

Для sample активны ровно две соседние complex control points. ILA regression:

\[
u[n]=y[n]/g,\qquad
\Phi_{n,k}=u[n]B_k(|u[n]|),
\]

\[
\hat{\mathbf c}=
\arg\min_{\mathbf c}
\frac1N\|\Phi\mathbf c-\mathbf x\|_2^2+
\lambda\|\mathbf c\|_2^2+
\mu\|D_2\mathbf c\|_2^2.
\]

Это одно комплексное решение. Эквивалентная real block matrix допустима, но две
независимые I/Q regression не нужны.

### Knot placement

- `uniform_amplitude`: равномерные \(r_k\);
- `uniform_power`: \(r_k=\sqrt{q_k}\) для равномерных \(q_k=r^2\), но
  interpolation по-прежнему линейна по amplitude;
- `quantile`: одинаковое приблизительное число calibration samples на interval;
- `compression_aware`: интервалы сжимаются к большому \(r\);
- adaptive knots допускаются позднее, только с validation-only objective и
  penalty за дополнительную calibration complexity.

Отдельный вариант, линейный непосредственно по \(r^2\), экономит square root, но
имеет другую basis function. Его нельзя выдавать за бесплатную реализацию
amplitude-linear spline: это отдельная ablation.

### Стоимость

Основная конвенция: complex multiply = 4 real multiplications + 2 real
additions; FMA разложен на MUL+ADD.

Для amplitude-linear spline с precomputed reciprocal interval width:

| Узел | Real MUL | Real ADD | Other |
|---|---:|---:|---|
| \(r=\sqrt{I^2+Q^2}\) | 2 | 1 | 1 square root |
| interval coordinate \(t\) | 1 | 1 | binary comparisons или direct index |
| complex control-point interpolation | 2 | 4 | 2 LUT reads |
| complex \(xC\) | 4 | 2 | — |
| **Итого** | **9** | **8** | 1 sqrt, comparisons, 2 lookups |

При Gauss complex multiply итог — 8 MUL и 11 ADD. Stored trainable coefficients:
\(2K\) real scalars; runtime arithmetic почти не зависит от \(K\), но binary
search, LUT size и cache/address width зависят.

### Ожидания и failure modes

Плюсы:

- AM/AM и AM/PM корректируются совместно;
- точное phase equivariance;
- deterministic convex calibration;
- streaming без state, future context и frame reset;
- удобный LUT/fixed-point datapath;
- \(K=8\ldots64\) намного ниже gate 1000 real MUL/sample.

Ограничения:

- не моделирует electrical/thermal memory и hysteresis;
- ILA может быть biased в глубокой compression/non-injective области;
- quantile knots могут недоразрешать редкие, но критичные high-amplitude peaks;
- endpoint extrapolation и maximum \(|z|\) должны контролироваться;
- на 160–200 MHz достижение −50 dB NMSE заранее маловероятно.

Baseline следует считать успешным даже при худшем NMSE, если он честно
устанавливает нижнюю границу стоимости. Усложнение разрешено только после
измеренного residual-versus-delay evidence.

## 3. Минимальные memory extensions

### A. Sparse spline memory branches

\[
z[n]=\sum_{b=1}^{B}x[n-m_b]C_b(|x[n-d_b]|).
\]

Начальный nested sequence:

1. \((m,d)=\{(0,0)\}\);
2. delays \(\{0,1\}\);
3. delays \(\{0,1,2\}\);
4. далее greedy/group selection.

Ожидаемая точность: выше memoryless при envelope-dependent electrical memory.
Стоимость: приблизительно \(B\) spline paths плюс \(2(B-1)\) real additions;
envelope calculation можно разделять для одинаковых \(d_b\). Calibration —
одна complex group-ridge regression. Риски: correlated branches, poor
conditioning, много control points при слепом добавлении delays.

Gate добавления branch:

- validation NMSE улучшается не менее чем на заранее установленный margin;
- либо left/right ACLR улучшается значимо по segments;
- improvement сохраняется после coefficient quantization;
- peak drive/stability не ухудшаются сверх лимита.

### B. Spline-based Hammerstein (SPH)

\[
v[n]=x[n]C(|x[n]|),\qquad
z[n]=\sum_{\ell=0}^{L-1}h_\ell v[n-\ell].
\]

Ожидаемая точность: хорошо ловит linear memory после nonlinear compression.
Стоимость: memoryless spline + \(L\) complex FIR taps, то есть примерно
\(9+4L\) real MUL/sample по основной конвенции. Regular datapath очень удобен
для FPGA.

Риски:

- scale ambiguity между spline и FIR;
- простой Hammerstein не ловит envelope-dependent memory до nonlinear block;
- alternating fit может зависеть от initialization.

Mitigation: фиксировать один control point или FIR DC gain, сравнить joint
linearized/alternating regression и ILA iterations.

### C. Sparse spline memory polynomial

Dictionary состоит из локальных spline features по выбранным pairs
\((m,d)\). Выбор:

- greedy forward selection с validation-only score;
- group LASSO по всем \(K\) control points branch;
- group OMP/OLS;
- pruning по validation NMSE и ACLR.

Плюсы: local support и MAC-friendly branches. Риски: selection wall-clock может
доминировать calibration, а irregular sparse delays усложняют buffering.
Следует хранить полный selection trace и считать offline feature-search cost
отдельно от inference.

### D. State-conditioned spline

\[
q_\ell[n]=\beta_\ell q_\ell[n-1]+(1-\beta_\ell)|x[n]|^2,
\]

\[
z[n]=g_0x[n]+\sum_bx[n-m_b]
C_b(|x[n-d_b]|,q_\ell[n]).
\]

Кандидат предназначен для slow thermal/bias effects. При фиксированных
\(\beta_\ell\) control points остаются линейными по coefficients; 2-D local
interpolation активирует четыре соседних points.

Плюсы: несколько дешёвых one-pole states, естественная streaming semantics.
Риски: \(\beta\), static spline и memory branches плохо идентифицируются на
коротком stationary capture; thermal timescale может отсутствовать в dataset.
Добавлять только если residual correlation/operating-point experiment это
подтверждает.

### E. Sparse CPWL

Complex canonical piecewise-linear features могут описывать локальные regions и
IQ imbalance. Phase-equivariant radial CPWL предпочтителен как первый вариант;
полный Cartesian CPWL — только если feedback audit показывает необходимость.

Плюсы: additions/comparisons, fixed-point friendly. Риски: number of regions,
boundary discontinuities derivative, routing irregularity, phase-equivariance
loss.

### F. Feature-selected GMP/CPWL dictionary

Большой offline dictionary строится из MP/GMP, cross-memory и CPWL features,
после чего выбираются groups. Это вероятный кандидат для качества при
\(<1000\) real MUL/sample.

Плюсы: linear/convex final calibration, понятный operation count, easy fixed
point. Риски: огромный initial dictionary, leakage при selection по test,
selection instability между waveforms и power levels. Необходимо nested
validation или фиксированный dictionary, минимум bootstrap/stability-selection.

### G. Cheap main path + tiny residual TCN

\[
z[n]=D_{\rm spline/MP}(x)[n]+\epsilon_\theta(x)[n].
\]

Residual network получает только ошибку, оставшуюся после дешёвого structured
path. Его receptive field и channels увеличиваются лишь до Pareto improvement.

Плюсы: потенциально достигает neural quality меньшей сетью. Риски:
noncausal look-ahead, activation/quantization cost, surrogate exploitation,
сложное совместное обучение. Разрешён последним, если structured dictionaries
не достигают acceptance gate.

## 4. Сравнительная таблица до экспериментов

Это priors, а не результаты.

| Candidate | Expected quality | Nominal inference | Calibration | Fixed-point/FPGA | Главный риск |
|---|---|---|---|---|---|
| Complex memoryless spline | Низкая–средняя на wideband PA | 9 real MUL + 8 ADD, 1 sqrt | one \(K\times K\) complex solve | Отлично | memory отсутствует |
| 2–3 spline branches | Средняя | примерно 18–27 MUL + sums | small group ridge | Отлично | correlated branches |
| SPH + short FIR | Средняя–высокая для Hammerstein-like PA | \(9+4L\) MUL | alternating/ridge/LMS | Отлично | block identifiability |
| Sparse spline memory polynomial | Средняя–высокая | \(O(B)\), local LUT | group selection + ridge | Хорошо | selection cost/instability |
| State-conditioned spline | Выше при slow drift | LUT + 4 MUL/state + 2-D interpolation | grid \(\beta\)+ridge | Хорошо | state not identifiable |
| Sparse radial CPWL | Средняя–высокая | comparisons + selected MAC | convex/greedy | Хорошо | region explosion |
| Feature-selected GMP/CPWL | Высокая при правильном dictionary | selected-term dependent | потенциально тяжёлая selection | Хорошо | generalization |
| Structured path + tiny TCN | Наивысший potential | десятки–сотни MAC + activations | E2E gradient training | Средне | latency/surrogate bias |
| EnhancedESN_FAN R600 I/Q | Не установлена честным test | 728,622 MUL + 726,152 ADD | eigensolve+state scan+ridge | Плохо | >729× cost gate |

## 5. Decision ladder

1. Зафиксировать evaluator и gain/alignment definitions.
2. Fit \(K=\{8,12,16,24,32,48,64\}\), четыре knot strategies и заданный
   regularization grid; выбирать только по validation.
3. Test запускать один раз после freeze.
4. Если memoryless residual имеет значимую lag structure, сравнить nested
   branches \(m=0\), \(0,1\), \(0,1,2\).
5. SPH сравнить при том же operation budget.
6. Только затем feature selection/state/tiny residual.
7. Каждый новый блок должен добавлять отдельную точку к Pareto frontier хотя бы
   по одной quality metric без нарушения stability/fixed-point gates.

Ни один surrogate-only результат не называется «лучше OpenDPD» до физической
проверки на том же PA или, как минимум, до общего frozen PA checkpoint с
идентичными splits и evaluator.
