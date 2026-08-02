# Реестр доказательств, рисков и незакрытых утверждений

Дата: 2026-07-30
Статус: живой реестр. `Доказано` означает только указанный класс evidence.

## 1. Шкала evidence

| Код | Смысл |
|---|---|
| `CODE` | подтверждено исходным кодом, тестом или hash manifest |
| `LOCAL-SUR` | локальный результат через PA surrogate |
| `PUBLISHED-PHY` | опубликованный результат на физическом PA, локально не воспроизведён |
| `LOCAL-PHY` | локальный контролируемый physical PA result |
| `HW-ANALYTIC` | аналитический operation/storage count |
| `HW-HOST` | host timing diagnostic; не target hardware |
| `HW-POST` | post-synthesis/post-layout simulation |
| `HW-TARGET` | измерение на целевом FPGA/ASIC/DSP |
| `UNKNOWN` | данных недостаточно |

Ни один `LOCAL-SUR` не повышается до `LOCAL-PHY` из-за хорошего NMSE.

## 2. Фактические утверждения текущего проекта

| Утверждение | Evidence | Данные/evaluator | Статус | Что не доказано | Эксперимент, закрывающий пробел |
|---|---|---|---|---|---|
| DPD test использует desired \(x\), а не measured \(y_\mathrm{test}\) | `CODE` | tests и spline evaluator | доказано локально | physical cascade | capture `x -> DPD -> PA -> y` |
| Train/validation/test разделены | `CODE` | DPA/APA manifests | доказано для текущего pipeline | независимость будущих physical captures | заранее sealed capture manifest |
| Frozen artifacts защищены hashes | `CODE` | result/model manifests | доказано | внешний storage/chain of custody | signed acquisition manifest |
| Spline-memory DPD phase-equivariant | `CODE` + математика | complex form | доказано с точностью арифметики | fixed-point equivariance under saturation | randomized bit-true rotation test |
| Spline-memory causal и streaming-compatible | `CODE` | chunk/reset tests | доказано для software core | target pipeline latency | RTL/DSP streaming test |
| Float analytical cost = 21 MUL + 24 ADD + sqrt для selected 3-branch DPD | `HW-ANALYTIC` | current operation counter | доказано по принятой convention | elapsed target time, memory stalls | reference-kernel target benchmark |
| Fixed-point core bit-accurate относительно собственной integer semantics | `CODE` | sealed runner/tests | доказано локально | RF degradation на physical PA | replay + physical 16/14/12-bit |
| Frozen validation DPD улучшает baseband adjacent leakage через current surrogate | `LOCAL-SUR` | DPA/APA frozen evaluator | доказано только на surrogate | физическое spectral suppression | physical paired captures |
| Current DPD лучше OpenDPD | нет | разные evaluators/results | **не доказано** | всё apples-to-apples | same PA/capture/evaluator/physical run |
| Current DPD соответствует Huawei | нет | точная metric/timing неизвестны | **не доказано** | bands/reference/threshold/hardware | written acceptance spec + target measurement |
| PA evaluator достаточно точен для дальнейшей тонкой DPD optimization | `LOCAL-SUR` | evaluator-vs-DPD error margin | **опровергнуто текущим gate** | независимое ranking | high-fidelity evaluator/physical PA |
| Gate A→B открыт | internal 10 dB margin | DPA/APA | **нет** | evaluator margin | retrain independent neural PA or physical test |
| Host timing укладывается в target | `HW-HOST` | pinned CPU diagnostics | **не доказано** | target clock/kernel/end-to-end | target implementation |
| 12/14/16 bit достаточно | `CODE` частично | integer core/runner | не завершено как RF claim | sealed metric table + physical | fixed-point sweep и physical replay |

## 3. Ограничения PA evaluator

| Риск | Наблюдение | Последствие | Severity | Mitigation |
|---|---|---|---:|---|
| Surrogate exploitation | residual DPD error близок к PA-model fidelity | optimizer может искать adversarial drive | critical | независимый evaluator и physical PA |
| Один evaluator для fit и score | common model bias | ложное ranking | critical | train/evaluate разными architectures/captures |
| Input/output sensitivity mismatch | raw \(\psi^He\) игнорирует \(J_P(z)\) | branch score не предсказывает изменение cascade output | critical | JVP/finite difference/малый probe; raw score только baseline |
| Gain/alignment refit | можно скрыть power loss | ложный ACPR/NMSE gain | high | freeze gain/alignment protocol |
| Feedback response mismatch | residual содержит receiver distortion | неверная branch recommendation | high | de-embedding + small-signal calibration |
| Frame reset artifact | error коррелирует с segment position | ложная long-memory ветвь | high | continuous capture и position diagnostic |
| Unseen drive amplitude | DPD выходит за train support | surrogate extrapolation | critical | amplitude guard + OOD flag |
| Shadow reused after discovery | validation уже повлияла на dictionary/null | optimistic advisor estimate | critical | split before analysis or acquire new sealed capture |

Текущий корректный вывод: surrogate пригоден для regression/unit diagnostics,
но недостаточен для заявления о физической линеаризации и для агрессивного
spectral fine-tuning.

## 4. Риски residual observer

| Гипотеза | Что уже известно | Что неизвестно | False-positive mechanism | Gate |
|---|---|---|---|---|
| Residual содержит пропущенную short-memory branch | sparse PA analysis обнаруживал lag structure | сохраняется ли она после DPD и на physical PA | evaluator bias | stable group correlation + independent score |
| \(xq_\beta\) отражает thermal memory | literature подтверждает thermal drift | есть ли thermal variation в captures | выбранные \(\beta\) описывают только ns–µs и/или frame position | \(\beta=\exp[-1/(f_s\tau)]\), persistent state, temperature sensor + repeated dwell |
| Widely-linear residual требует DPD branch | IQ-related correlations измеримы | находится дефект в TX или observer | observation IQ imbalance | two-path calibration before branch |
| Correlation ranking предсказывает spectral gain | OMP/decorrelation prior art | работает ли на spline groups | collinearity и отсутствие PA/spectral sensitivity | QR/SVD-whitened \(J_P\Phi_G\) score vs exhaustive/random/DOMP |
| One-branch update стабилен | bounded small update | physical peak/transient | gain overcompensation | shadow spectrum + peak gates |
| Drift detector обнаружит режим | frame metrics доступны | thresholds/false alarm | normal waveform variability | controlled transitions and null runs |

## 5. Риски архитектур

### A. Sparse residual spline/GMP

- Evidence: spline/SPH/SMP, DOMP, sparse Volterra и decorrelation имеют
  physical priors.
- Gap: их безопасная комбинация в нашем pipeline не проверена.
- Failure: feature selection нестабилен между кадрами/evaluators.
- Kill criterion: выбранная ветвь не лучше random/exhaustive tolerance либо
  ухудшает worst spectral bin/peak.

### B. Slow-state conditioning

- Evidence: temperature/power adaptive DPD и APNRRU envelope states.
- Gap: текущие captures не имеют доказанного slow-state axis.
- Failure: state кодирует sample position вместо температуры.
- Kill criterion: benefit исчезает при holdout frame positions или не
  переносится между repeated dwell captures.

### C. Expert bank

- Evidence: real-time model switching, gated dynamic NN, sparsely gated MoE.
- Gap: число реальных operating points и transition behavior.
- Failure: rapid switching создаёт modulation/spurs.
- Kill criterion: coefficient LUT с hysteresis равна или лучше learned router.

### D. Hypernetwork/low-rank update

- Evidence: coefficient predictors/meta-learning существуют.
- Gap: low-rank structure коэффициентов нашего PA.
- Failure: extrapolation создаёт unsafe drive.
- Kill criterion: SVD не показывает compact subspace либо limited-sample ridge
  лучше на held-out condition.

### E. Tiny residual neural path

- Evidence: physical residual/phase-normalized networks.
- Gap: advantage при равной target latency.
- Failure: activation/memory traffic дороже structured branch.
- Kill criterion: не доминирует sparse spline/GMP на validation Pareto.

### F. Lightweight SSM

- Evidence: general SSM theory; APNRRU/slow-state DPD.
- Gap: physical DPD advantage именно diagonal state.
- Failure: это дорогая запись нескольких one-pole filters.
- Kill criterion: не лучше state-conditioned spline/short FIR при равной
  measured cost.

### G. Distillation

- Evidence: ILC teacher и DPD knowledge-distillation papers.
- Gap: student benefit против direct fit.
- Failure: teacher/surrogate bias и excessive teacher peak.
- Kill criterion: direct fit student не хуже на unseen physical blocks.

## 6. Спектральные риски

| Риск | Почему метрика может обмануть | Обязательная защита |
|---|---|---|
| Падение основной мощности | dBc leakage улучшается без линеаризации | absolute и relative band powers; main-band constraint |
| Усреднение left/right | скрывает асимметрию | left и right отдельно |
| Узкий новый spike | integrated OOB может уменьшиться | worst-bin/spectral mask |
| Welch/window leakage | измеряется evaluator artifact | frozen window/FFT/overlap/masks |
| Несогласованный sample rate | неправильные границы полос | dataset metadata hash |
| Среднее по кадрам | редкие unsafe кадры скрыты | percentile и worst frame |
| Peak drive | хороший spectrum ценой опасного PA input | max \(|z|\), PAPR, clipping count |
| Surrogate adversarial spectrum | модель не видит реальное поведение | second evaluator + physical PA |
| RF harmonics 2fc/3fc | complex baseband capture их не содержит | RF analyzer/observation bandwidth |

До определения Huawei отдельно публикуются left/right adjacent leakage,
integrated OOB, main-band power, PSD, EVM, NMSE, peak, PAPR и worst spike.
Customer spectral gates остаются `BLOCKED`, пока не определены bands,
reference, numerical deltas, confidence level и required capture count.

## 7. Hardware/timing риски

| Утверждение/риск | Статус | Почему недостаточно MUL count | Закрывающий тест |
|---|---|---|---|
| `<1000 MUL` означает соответствие | неверно | ADD/LUT/sqrt/memory/control/clock | compare target reference kernel |
| Parameter count = operation count | неверно | recurrence/convolution reuse различны | explicit schedule |
| Sparse weights бесплатны | неверно без zero skipping | indices/router/cache | sparse target kernel |
| Average temporal sparsity гарантирует deadline | неверно | worst frame может быть dense | p99.9/worst latency |
| Host NumPy timing = embedded timing | неверно | vectorization/cache/OS | FPGA/DSP/RTL target |
| FP16 = fixed-point | неверно | saturation/scaling/accumulator различны | bit-true integer simulation |
| LUT access дешевле MUL | target-dependent | BRAM ports/addressing critical | measured synthesis/timing |
| `sqrt` — одна операция | target-dependent | iterative/LUT approximation | compare magnitude/power knots |

## 8. Robustness risks

| Ось | Текущее evidence | Запрещённое утверждение | Нужные данные |
|---|---|---|---|
| DPA vs APA | два разных физических PA/datasets | cross-PA generalization | не смешивать; отдельные experiments |
| APA vs APA_B | capture transfer, hidden axes unknown | temperature/power robustness | labeled controlled operating points |
| Output power | отдельные literature priors | robust locally | repeated captures at fixed temperature |
| Temperature | локальных данных нет | thermal adaptation | sensor + dwell/transition protocol |
| Bandwidth/waveform | metadata различна между datasets | waveform generalization | same PA, controlled waveforms |
| PA instance | нет | manufacturing robustness | several units |
| Aging | нет | long-term adaptation | longitudinal captures |
| Observation SNR | нет controlled sweep | feedback-noise robustness | attenuation/noise sweep |

## 9. Novelty register

| Идея | Ближайший prior art | Честная оценка |
|---|---|---|
| Residual correlation | decorrelation DPD, OMP/DOMP | стандартный метод |
| Sparse branch selection | DOMP, LASSO, sparse Bayesian | стандартный/инженерный |
| Piecewise/local spline | spline LUT, CPWL, piecewise DPD | известный |
| Operating-point experts | model switching, MoE/GDNN | известный |
| Hypernetwork coefficients | coefficient predictor/meta-learning | известная идея |
| Slow envelope states | thermal models, APNRRU | известный механизм |
| Safe shadow accept/rollback | safe RL DPD Spano et al. 2025 уже имеет threshold/recovery | известный safety pattern |
| Cross-frame null-tested cost-aware selection + spectral rollback | отдельные части известны | потенциально новая комбинация |

До формального systematic prior-art review и physical experiment слово
`novel` в названии результата не используется.

## 10. Decision log и gates

### Gate R0 — observer-only

Разрешён сейчас. Не меняет frozen DPD.

Успех:

- воспроизводимая causal residual group;
- confidence interval выше block-null;
- стабильность по кадрам;
- не observation/frame artifact.

### Gate R1 — advisor

Разрешён после R0 и при независимом evaluator.

Успех:

- рекомендация не хуже exhaustive-best-within-tolerance;
- лучше random при равной стоимости;
- validation spectrum улучшен без peak/main-band penalty.

### Gate R2 — adaptive coefficients

Разрешён после shadow/rollback tests.

Успех:

- bounded update;
- bit-true path;
- forced bad candidate отклоняется;
- rollback восстанавливает прежние metrics.

### Gate R3 — physical claim

Требует sealed physical protocol:

- одинаковый output power/temperature/waveform;
- no-DPD и DPD captures;
- repeated runs и confidence intervals;
- observation-path calibration;
- RF bandwidth, включающий требуемые spurs/harmonics.

### Gate R4 — Huawei/timing claim

Требует:

- точного определения spectral bands/reference/threshold;
- target hardware/clock/data format;
- reference 1000-MUL kernel;
- measured throughput, latency и high quantiles.

## 11. Ответы, которые ещё нужны от Huawei

1. Что именно называется «паразитной гармоникой»: adjacent regrowth, discrete
   IM products, RF harmonics около \(2f_c/3f_c\) или spectral mask violations?
2. Точные band edges, resolution bandwidth, window/detector и averaging.
3. Reference: absolute dBm, dBc относительно main-band/channel power или
   improvement относительно no-DPD?
4. Minimum suppression и допустимая потеря основной мощности/EVM.
5. Максимальные \(|z|\), PAPR и clipping policy.
6. Target sample rate, clock, FPGA/ASIC/DSP, numeric format и memory hierarchy.
7. Определение эталонной «1000 real MUL» программы: serial/parallel, bit width,
   complex convention, inclusion of loads/stores.
8. Допустимая latency и требуемая sustained throughput.
9. Разрешённая observation receiver bandwidth/SNR/ADC rate.
10. Частота обновления коэффициентов и допустимые calibration probes.
11. Рабочие диапазоны power, temperature, carrier, bandwidth, waveform.
12. Нужны ли гарантии при смене PA экземпляра/старении.

## 12. Следующий эксперимент максимальной информационной ценности

Не меняя frozen DPD, выполнить observer-only анализ на frozen validation:

- complex residual-feature group correlations;
- residualization относительно active branches;
- block bootstrap/circular-shift null;
- frame consistency и segment-position control;
- cost annotation;
- comparison ranking между независимыми evaluators.

Положительный результат обосновывает ровно одну branch proposal. Отрицательный
результат предотвращает ненужное усложнение и указывает, что следующий ресурс
следует потратить на observation/evaluator fidelity, а не на новую сеть.
