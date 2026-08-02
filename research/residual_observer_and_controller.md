# Residual observer и безопасный контроллер DPD

Дата: 2026-07-30
Статус: исследовательская спецификация; алгоритмы и frozen-результаты проекта
не изменены.

## 1. Роль модуля

Предлагаемый модуль находится не в основном тракте передачи и не является
«вторым DPD». Основной тракт остаётся коротким и детерминированным:

```text
desired x -> deployed DPD D_theta -> physical PA -> antenna
```

Небольшая копия выхода снимается ответвителем:

```text
PA output copy
  -> observation receiver
  -> de-embedding and alignment
  -> residual observer
  -> advisor / shadow evaluator
  -> bounded coefficient proposal
  -> accept or rollback
```

Это разделяет два масштаба времени:

- `D_theta` вычисляется для каждого отсчёта и должен укладываться в аппаратный
  временной бюджет;
- observer может обрабатывать блок раз в кадр или только после обнаружения
  дрейфа; его стоимость не входит в стоимость рабочего DPD, но должна быть
  измерена как стоимость калибровки.

Цель observer — не минимизировать произвольную ошибку программной модели, а
обнаруживать воспроизводимую структуру физического остатка, которую текущий
DPD не исправляет.

## 2. Наблюдаемая величина

Пусть

\[
z[n]=D_\theta(x)[n],\qquad y_\mathrm{PA}[n]=P(z)[n].
\]

Observation receiver выдаёт не сам \(y_\mathrm{PA}\), а

\[
v[n]=h_\mathrm{obs}*y_\mathrm{PA}[n-\tau]
    +k_\mathrm{obs}*y_\mathrm{PA}^*[n-\tau]
    +d_\mathrm{DC}+\eta[n]+\nu_\mathrm{nl}[n].
\]

Здесь \(h_\mathrm{obs}\) — обычная АЧХ/ФЧХ, \(k_\mathrm{obs}\) — image/IQ
leakage, \(\tau\) — задержка, \(\eta\) — шум, а \(\nu_\mathrm{nl}\) —
нелинейность самого приёмника. LTI- и widely-linear части оцениваются
раздельно. После их калибровки получаем \(\tilde y\). Подтверждённую
receiver compression нельзя «обратить» scalar FIR: observer должен перейти в
fail-closed состояние и запретить coefficient update.

Желаемый выход фиксируется как

\[
y_\mathrm{target}[n]=g\,x[n].
\]

Остаток

\[
e[n]=\tilde y[n]-g\,x[n]
\]

имеет смысл только после корректного выравнивания. Коэффициент \(g\), delay и
de-embedding нельзя независимо переоценивать для каждого DPD-кандидата так,
чтобы алгоритм скрывал потерю мощности или фазовую ошибку.

## 3. Калибровка observation path

### 3.1 Последовательность

1. Удалить DC отдельно по I и Q на calibration capture.
2. Найти грубую integer delay по максимуму нормированной комплексной
   cross-correlation на линейном или малосигнальном capture.
3. Оценить fractional delay по фазовому наклону cross-spectrum либо
   band-limited fractional-delay FIR.
4. Оценить линейный complex gain

   \[
   g=\frac{\sum_n x^*[n]\tilde y[n]}{\sum_n|x[n]|^2}
   \]

   на заранее определённом calibration interval.
5. Оценить частотную характеристику observation path на малосигнальном
   multitone/OFDM capture и применить регуляризованный inverse FIR только в
   надёжно наблюдаемой полосе.
6. Проверить widely-linear leakage:

   \[
   \tilde y[n]\simeq h*x[n]+\bar h*x^*[n].
   \]

   Сильный \(\bar h\) сначала считается дефектом observation path/IQ, а не
   новой нелинейной ветвью DPD.
7. Проверить собственную компрессию observation receiver двумя уровнями
   ответвления. Изменение residual spectrum с уровнем при неизменном PA output
   означает, что observer нельзя использовать для обновления.

### 3.2 Acceptance checks для выравнивания

- задержка стабильна между повторными captures в пределах заданного допуска;
- остаточный фазовый наклон после correction статистически неотличим от
  calibration noise floor;
- inverse observation filter не усиливает шум вне полосы;
- оценка \(g\) на независимых calibration blocks согласована;
- метрики без DPD воспроизводимы до включения adaptive loop.

## 4. Причинная библиотека признаков

В рабочий DPD разрешены только причинные признаки. Для observer можно считать
диагностики шире, но noncausal признак не может автоматически стать deployed
branch.

### 4.1 Phase-equivariant кандидаты

Для глобального фазового поворота \(x\mapsto xe^{j\phi}\) желательна
эквивариантность \(\psi\mapsto\psi e^{j\phi}\):

\[
\psi^\mathrm{MP}_{m,p}[n]
  =x[n-m]|x[n-m]|^{p-1},
\]

\[
\psi^\mathrm{GMP}_{m,d,p}[n]
  =x[n-m]|x[n-d]|^{p-1},
\]

\[
\psi^\mathrm{spline}_{m,d,k}[n]
  =x[n-m]B_k(|x[n-d]|),
\]

\[
q_\beta[n]=\beta q_\beta[n-1]+(1-\beta)|x[n]|^2,
\qquad
\psi^\mathrm{slow}_{m,\beta}[n]=x[n-m]q_\beta[n].
\]

Это envelope-memory proxy, не измерение температуры. Thermal hypothesis
использует \(\beta=\exp[-1/(f_s\tau)]\), delayed \(|z|^2\)/output-power и
temperature/bias sensors; состояние с большим \(\tau\) нельзя сбрасывать на
обычной границе кадра.

Для spline local support на одном отсчёте активны только две соседние basis
functions. Observer должен группировать пару control-point features как одну
аппаратную ветвь, а не выбирать отдельный knot.

Полезна также dynamic-deviation ветвь

\[
\psi^\Delta_{m,d}[n]
  =x[n-m]\left(|x[n-d]|^2-|x[n-d-1]|^2\right).
\]

### 4.2 Widely-linear диагностика

\[
\psi^\mathrm{WL}_m[n]=x^*[n-m]
\]

нарушает обычную phase-equivariance. Она нужна как индикатор I/Q imbalance,
image leakage или неправильного feedback correction. Включать такую ветвь в
DPD можно только после отдельного физического доказательства, что источник
находится в transmit path, а не в observation receiver.

### 4.3 Сегментные признаки

Observer сохраняет статистики по кадрам:

- mean и percentile мощности \(|x|^2\);
- PAPR и максимальный \(|z|\);
- долю отсчётов в compression region;
- температуру, bias и output power, если датчики доступны;
- left/right leakage, integrated out-of-band power и worst spectral bin;
- residual RMS и residual-feature correlations;
- position within capture/frame.

Корреляция с position признаком важна: она отличает физический медленный state
от ошибки reset/warm-up/framing.

## 5. Комплексная корреляционная диагностика

Raw score \(\psi^He\) является только observer diagnostic. Добавление
признака меняет вход PA, тогда как \(e\) измерен на его выходе:

\[
\delta y \simeq J_P(z)\,\delta z.
\]

Поэтому локальная полезность branch для time loss определяется
\((J_P\psi)^He\), а для band loss — корреляцией после соответствующего
spectral operator. \(J_P\psi\) оценивается JVP независимого evaluator,
симметричной finite difference либо безопасным малым physical probe.
Непреобразованная корреляция допустима только как baseline с явно указанным
приближением PA постоянным complex gain.

Partial group score вычисляется устойчиво через QR/SVD. Пусть \(Q_A\) —
ортонормированный базис output-sensitivity columns уже активного словаря, а
\(\Xi_G=J_P(z)\Phi_G\) — output-sensitivity candidate group. Тогда

\[
e_\perp=(I-Q_AQ_A^H)e,\qquad
R_G=(I-Q_AQ_A^H)\Xi_G,
\]

\[
R_G=U_G\Sigma_GV_G^H,\qquad
s_G=\frac{\|U_G^He_\perp\|_2}
          {\|e_\perp\|_2+\epsilon},
\]

где сохраняются только singular directions выше заранее заданного rank
threshold. Это residualizes обе стороны и сравнивает энергию ортогональной
group subspace, а не произвольный масштаб/поворот её columns. Ridge
используется позднее при fit coefficients, но regularized hat matrix не
называется ортогональным проектором.

Без residualization сильно коррелированные GMP/spline ветви многократно
«обнаруживают» одну и ту же структуру.

В score добавляется стоимость:

\[
s_G^\mathrm{cost}
=
\frac{s_G}{1+\alpha_\mathrm{mul}M_G+\alpha_\mathrm{add}A_G+
\alpha_\mathrm{lut}L_G+\alpha_\mathrm{state}S_G}.
\]

Это только ранжирование гипотез. Даже sensitivity-aware score не является
доказательством улучшения ACLR: его обязана подтвердить отдельная spectral
shadow capture.

## 6. Защита от ложных корреляций

Для каждого кандидата нужны одновременно:

1. block bootstrap confidence interval по целым кадрам;
2. permutation целых кадров/captures; circular shifts разрешены только после
   исключения OFDM symbol/CP/frame periodicities;
3. проверка знака/фазы корреляции между кадрами;
4. holdout validation blocks, не участвовавшие в выборе;
5. повторение **всей** selection procedure в каждой null-репликации;
6. max-statistic либо обоснованный FDR для зависимых feature groups;
7. минимальный effect-size, а не только маленькое `p`;
8. воспроизводимость минимум на нескольких captures одного режима.

Обычное случайное перемешивание отдельных samples разрушает временную
структуру OFDM и даёт слишком оптимистичный null. Phase randomization может
также разрушить именно envelope dependence, которую проверяет branch.
Поэтому основной null формируется целыми кадрами/captures, а не отдельными
samples.

## 7. Четыре режима работы

### 7.1 Observer-only

Контроллер публикует residual report, но не меняет коэффициенты. Это первый и
единственный разрешённый следующий шаг при текущем недостаточно точном PA
evaluator.

Выход:

- ранжирование causal feature groups;
- confidence interval и null percentile;
- устойчивость по кадрам;
- оценка добавочной рабочей стоимости;
- диагностика observation-path/IQ/frame artifacts.

### 7.2 Advisor

Observer предлагает одну ветвь \(G^\star\), но исследователь вручную запускает
fit только на train/calibration data. `advisor_select` используется для
topology/regularization, а заранее запечатанный `advisor_shadow` — один раз
для сравнения с:

- случайной ветвью той же стоимости;
- лучшей ветвью полного малого перебора;
- OMP/group-LASSO;
- неизменённой frozen architecture.

Test остаётся закрытым.

### 7.3 Safe adaptive controller

Обновление \(\theta'\) сначала работает в shadow path:

1. fit на calibration block;
2. integrity/CRC/version check и projection коэффициентов в допустимое
   множество;
3. bit-accurate генерация \(z'\), accumulator/saturation, peak, PAPR и
   clipping guards;
4. offline replay через независимый evaluator;
5. только затем dummy-load или ступенчатый low-power physical probe с
   hardware watchdog;
6. validation capture на прежнем operating point и проверка всех gates;
7. атомарная смена coefficient bank на границе кадра.

Candidate привязывается к hash/version observation calibration, operating
point, исходному coefficient bank и времени capture. Изменение режима либо
устаревшая calibration автоматически отменяют update.

Принятие:

\[
\Delta L_\mathrm{adj}<-\delta_\mathrm{adj},
\quad
\Delta P_\mathrm{main}\ge-\delta_\mathrm{main},
\quad
\Delta\mathrm{EVM}\le\delta_\mathrm{EVM},
\]

\[
\max|z'|\le z_\mathrm{safe},
\quad
\mathrm{PAPR}(z')\le p_\mathrm{safe},
\quad
\max_f\{\mathrm{PSD}'(f)-\mathrm{mask}(f)\}\le 0.
\]

Улучшение среднего ACLR не компенсирует новый узкий spectral spike или
чрезмерный drive.

Все \(\delta\), confidence level, capture count, left/right no-regression,
main-band/EVM/peak/PAPR/worst-bin limits задаются до capture. Пока Huawei не
определила bands и acceptance thresholds, этот gate имеет статус `BLOCKED`
для customer acceptance; internal thresholds не выдаются за Huawei criteria.

### 7.4 Sparse residual adaptation

Активно не более \(B_\max\) дополнительных групп. На каждом update можно
добавить или удалить не более одной группы. Используется hysteresis:

- `add_threshold` выше `keep_threshold`;
- минимальное dwell time;
- ограничение нормы \(\|\theta'-\theta\|\);
- forgetting factor только внутри заданного диапазона;
- rollback bank хранится до нескольких подтверждённых кадров.

## 8. Drift и смена режима

Сигнал тревоги формируется не одной метрикой, а несколькими независимыми
признаками:

\[
D_t=w_1Z_\mathrm{amp}+w_2Z_\mathrm{power}+w_3Z_\mathrm{residual}
    +w_4Z_\mathrm{spectrum}+w_5Z_\mathrm{sensor},
\]

где каждый \(Z\) нормирован относительно verified baseline distribution и
noise floor. Threshold задаётся через целевой false-alarm rate и persistence
по нескольким кадрам. Observation-health fault является отдельным veto и не
может быть компенсирован хорошими значениями других слагаемых.

Возможные detector:

- CUSUM/Page–Hinkley для среднего residual power;
- change point по frame-level ACLR/EVM;
- divergence amplitude histogram;
- рост disagreement двух PA evaluators;
- рост widely-linear residual, указывающий на observation fault.

Состояния:

```text
NORMAL -> SUSPECT -> SHADOW_CALIBRATION -> CANDIDATE
   ^          |              |                |
   |          v              v                v
 SAFE_FALLBACK <--------- REJECT/ROLLBACK <- VERIFY
```

При неисправности observer основной DPD не должен получать неконтролируемые
updates. Safe fallback — последний физически проверенный coefficient bank либо
консервативный low-drive профиль.

## 9. Варианты архитектуры контроллера

| Вариант | Уже известная основа | Возможное отличие проекта | Рабочая стоимость | Главный риск | Минимальный эксперимент |
|---|---|---|---|---|---|
| Observer-only | residual/decorrelation analysis | frame-stability + null tests + cost annotation | ноль | feedback artifact принимается за PA memory | frozen validation residual report |
| Advisor | OMP/DOMP, feature selection | group spline/GMP/slow-state library | только принятая ветвь | validation overfit | one-branch recommendation vs random/exhaustive |
| Safe controller | closed-loop/decorrelation DPD | spectral shadow gates + atomic rollback | bounded | unstable coefficient update | two-bank replay and forced rejection tests |
| Sparse residual adaptation | sparse Volterra/GMP pruning | stable cross-frame selection under hard timing budget | \(O(B_\max)\) | basis collinearity | OMP/group-LASSO/correlation ablation |
| Operating-point experts | model switching/MoE | frame-rate hard routing with hysteresis | cost одного expert | switching transient | two known captures, blinded transition |
| Hypernetwork coefficients | hypernetwork/meta-learning | slow generator, projected cheap DPD coefficients | base DPD only | unsafe extrapolation | held-out condition coefficient prediction |
| Low-rank update | adaptive coefficient subspace | identified physical drift subspace | base + small update | drift not low-rank | SVD across controlled conditions |
| Teacher–student | ILC/neural distillation | physical feedback teacher, structured student | student only | student inherits teacher/surrogate bias | direct fit vs distilled fit |

## 10. Научная новизна

Корреляция остатка с базисными функциями сама по себе не нова. Ближайшие
семейства prior art:

- decorrelation-based concurrent DPD;
- piecewise decorrelation и self-orthogonalization;
- OMP/DOMP и sparse Bayesian selection;
- dynamic selection активных параметров;
- pruning basis functions по residual contribution.

Кроме того, safe update/recovery тоже имеет прямой prior art:
[Spano et al., 2025](https://doi.org/10.3390/s25196102) проверили в Huawei
Milan safe deep-RL DPD, ограничивающий действие около ACLR threshold и
выполняющий recovery через обратную последовательность сохранённых действий.
Следовательно, ни spectral safety threshold, ни rollback сами по себе не
являются новизной проекта.

Поэтому формулировку
`Correlation-guided sparse residual adaptation for low-complexity DPD`
пока следует считать **потенциально новой комбинацией известных механизмов**,
а не доказанной новой теорией.

Самостоятельный вклад возможен, если одновременно показаны:

1. complex, causal, phase-aware group dictionary;
2. cross-frame stability и корректный correlated null;
3. cost-aware selection под измеренным timing budget;
4. независимая spectral shadow validation;
5. peak/main-band/EVM/worst-bin safety constraints;
6. atomic accept/rollback;
7. controlled physical drift experiments;
8. преимущество над DOMP/group-LASSO/decorrelation при равной рабочей
   стоимости;
9. сравнение с safe-RL prior по числу physical updates, auditability и
   recovery, а не только final ACLR.

Возможное отличие от safe RL следует формулировать узко: детерминированный
active-set proposal, 1–несколько shadow fits вместо сотен exploration updates,
block-null statistical evidence, multi-metric/worst-bin gates и атомарная
смена coefficient bank. Пока это гипотеза, а не опубликованное преимущество.

## 11. Первый эксперимент

Первый эксперимент не меняет frozen DPD:

1. до любого анализа разделить целые frozen validation frames на
   `observer_discovery`, `advisor_select` и запечатанный `advisor_shadow`;
   если validation уже исследовалась, независимый shadow требует нового
   capture;
2. восстановить ровно тот alignment/gain, которым сформирован frozen result;
3. вычислить residual для неизменённой spline-memory DPD;
4. построить causal library с ограниченными delays и несколькими
   \(\beta\);
5. residualize относительно активных ветвей;
6. посчитать raw и PA-sensitivity-transformed complex group scores по кадрам;
7. сравнить с whole-frame/capture null, повторяя полную selection procedure,
   и применить max-statistic либо обоснованный FDR;
8. потребовать одинаковую полезную группу в большинстве кадров;
9. опубликовать ожидаемую добавочную MUL/ADD/LUT/state стоимость;
10. использовать только `observer_discovery`, не выполнять fit и не открывать
    `advisor_shadow` или test.

Gate для перехода к advisor:

- найден хотя бы один causal group с confidence interval выше null threshold;
- эффект устойчив по кадрам и не объясняется frame position;
- это не widely-linear observation artifact;
- дополнительная ветвь помещается в рабочий timing envelope;
- независимый evaluator либо физический PA доступен для последующей
  спектральной проверки; без него результат остаётся observer hypothesis и
  gate закрыт.

Текущий проект удовлетворяет не последнему условию: PA evaluator пока слишком
близок по ошибке к остаточной DPD-ошибке. Поэтому сейчас обоснован
observer-only diagnostic, но не автоматическое обновление DPD.
