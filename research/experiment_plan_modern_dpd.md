# План экспериментов для следующего поколения DPD

Дата: 2026-07-30
Статус: план; новые model runs в рамках данного research-коммита не
выполнялись.

## 1. Принцип последовательности

Нельзя одновременно менять evaluator, architecture и spectral metric: источник
улучшения станет неидентифицируем. Порядок:

1. зафиксировать требования и evidence;
2. проанализировать остаток, не меняя frozen DPD;
3. получить независимый evaluator либо physical feedback;
4. проверить advisor на validation;
5. добавлять по одной ветви;
6. открыть test только после freeze;
7. провести physical и target-hardware validation.

Текущий Gate A→B закрыт: точность имеющегося PA evaluator недостаточно
превосходит остаточную ошибку DPD. Поэтому немедленно разрешён только
observer-only diagnostic. Новая DPD optimization через тот же evaluator
запрещена.

## 2. Неизменяемые объекты

До завершения исследования не изменяются:

- существующие frozen DPD artifacts;
- frozen test results;
- split manifests;
- spectral replay configurations;
- hashes datasets/checkpoints;
- released test decisions.

Любой новый result получает отдельный config, output directory, execution
record, input hashes и Git commit.

### 2.1 Реестр порогов до запуска

До любого `PASS/FAIL` фиксируются:

- primary metric, units, exact left/right bands и reference;
- \(\delta\) improvement/no-regression отдельно для left и right;
- main-band, EVM, peak, PAPR, worst-bin и worst-frame limits;
- confidence level, число независимых captures и multiplicity correction;
- fixed-point degradation и timing limits.

Пока Huawei не определила bands/reference/thresholds, customer acceptance
gates имеют статус `BLOCKED`. Внутренние diagnostic thresholds можно
пререгистрировать, но их нельзя называть Huawei acceptance.

## 3. Stage 0 — reproducibility preflight

### Цель

Подтвердить, что текущий repository state воспроизводимо читает frozen
artifacts до добавления новых diagnostics.

### Существующие проверки

```bash
python -m unittest discover -s tests -v
```

```bash
python experiments/evaluate_frozen_dpd_spectrum.py \
  --config experiments/configs/dpd_spectral_replay_dpa200_validation.json
```

```bash
python experiments/evaluate_frozen_dpd_spectrum.py \
  --config experiments/configs/dpd_spectral_replay_apa200_validation.json
```

Команды replay нельзя направлять в уже sealed output без встроенной
verification/overwrite policy. На preflight достаточно hash verification;
численные files не перегенерируются в research-only этапе.

### Gate S0

- чистый git status до run;
- совпали commit/data/model/config hashes;
- tests проходят;
- frozen outputs не изменены.

## 4. Stage 1 — observer-only residual analysis

### 4.1 Данные

До первого просмотра residual целые validation frames детерминированно
делятся и hashes публикуются:

- `observer_discovery` — только Stage 1;
- `advisor_select` — topology/regularization selection;
- `advisor_shadow` — открывается один раз после freeze.

Если существующая validation уже многократно исследовалась, её части нельзя
переименовать в независимый shadow: нужен новый capture.

Используются только frozen **validation** waveforms:

- DPA_200MHz отдельно;
- APA_200MHz отдельно;
- datasets не объединяются;
- legacy test artifacts не используются для feature selection.

Входы:

- desired \(x\);
- frozen DPD drive \(z\);
- frozen evaluator output \(\hat y\);
- fixed target \(gx\);
- frame boundaries и warm-up policy;
- существующий alignment/gain manifest.

### 4.2 Feature library v1

Ограниченный causal grid:

- signal delays \(m=0,\ldots,M_s\);
- envelope delays \(d=0,\ldots,M_e\);
- odd MP/GMP orders \(p\in\{1,3,5,7\}\);
- spline groups на существующих knots;
- dynamic deviation \(\Delta |x|^2\);
- fast envelope states задаются через
  \(\beta=\exp[-1/(f_s\tau)]\) для заранее объявленных
  \(\tau\in\{10\,\mathrm{ns},100\,\mathrm{ns},1\,\mu\mathrm{s},
  10\,\mu\mathrm{s}\}\) и не называются thermal;
- thermal-scale state допускается только при capture duration/sensors,
  достаточных для соответствующего \(\tau\); он использует delayed
  \(|z|^2\), output power или sensors, сохраняется между кадрами и может
  обновляться на control rate;
- conjugate/widely-linear features только как observation/IQ diagnostic;
- frame-position controls.

\(M_s,M_e\) сначала ограничиваются текущим short-memory range. Большой
dictionary не открывается до evidence, чтобы multiple-testing burden не стал
основным результатом.

### 4.3 Статистика

Для каждого group:

- complex normalized residual correlation;
- QR/SVD residualization относительно active branches;
- raw score как baseline и output-sensitivity score после
  \(J_P(z)\Phi_G\), оценённого независимым evaluator/JVP, finite difference
  или безопасным малым probe;
- frame-wise magnitude и phase;
- block-bootstrap 95% interval;
- whole-frame/capture permutation null; периодические OFDM/CP shifts
  исключаются;
- full selection rerun per null replicate и max-statistic либо обоснованный
  FDR для зависимых groups;
- consistency across frames;
- analytical incremental MUL/ADD/LUT/state cost.

### 4.4 Baselines

- raw correlation;
- partial/residualized correlation;
- случайное ranking;
- correlation после sample shuffle — только как демонстрация неверного null;
- whole-frame/capture null;
- existing PA residual analyzer where semantics match.

### 4.5 Результат

Один report на dataset:

```text
feature_group
score
null_percentile
confidence_interval
phase_consistency
frame_hit_rate
incremental_cost
suspected_artifact
```

Frozen DPD coefficients не меняются; `advisor_select`, `advisor_shadow` и test
не открываются.

### Gate S1

Переход разрешён, если одна causal group:

- выше заранее заданного null/FDR threshold;
- стабильна по кадрам;
- не коррелирует только с frame position;
- не является widely-linear observation artifact;
- повторяется через второй evaluator либо physical capture.

Без такого подтверждения результат остаётся observer hypothesis, Gate S1
закрыт. Если ranking сильно меняется между evaluators, следующий шаг —
evaluator, не DPD.

## 5. Stage 2 — независимый PA evaluator

### 5.1 Сравниваемые evaluators

- текущий MP PA;
- текущий GMP PA;
- current sparse spline PA;
- OpenDPD neural PA backbone, переобученный без DPD cost cap;
- второй neural/structured evaluator с отличным inductive bias.

PA cost не ограничивается 1000-operation proxy.

### 5.2 Protocol

- те же alignment, gain, framing и splits;
- train architecture/hyperparameters только по train/validation;
- pooled complex NMSE;
- OpenDPD-compatible NMSE отдельно;
- error PSD;
- AM/AM и AM/PM residual;
- residual correlations;
- error by amplitude/frame position;
- model disagreement на proposed DPD drive.

Существующий preflight:

```bash
python experiments/train_opendpd_pa.py \
  --config experiments/configs/opendpd_pa_cpu_preflight_apa200.json
```

Полный GPU config должен быть создан отдельной задачей после фиксации
environment/seed/budget; CPU preflight не считается converged evaluator.

### Gate S2

Для fine DPD optimization требуется:

- pooled fidelity существенно лучше ожидаемого DPD residual; внутренний
  ориентир 10 dB является только необходимым diagnostic, не достаточным
  gate;
- bandwise error PSD и absolute leakage-prediction error ниже заранее
  заданных limits именно на predistorted-drive distribution;
- coverage/uncertainty не ухудшаются на proposed drive;
- branch ranking согласован как минимум у двух evaluators;
- отсутствует рост disagreement в диапазоне proposed drive;
- evaluator checkpoint выбирается не по test.

Два evaluators, обученные на одном capture, проверяют model dependence, но не
являются независимым physical evidence.

Если Gate S2 не пройден, перейти к Stage 6 physical PA, а не усложнять
surrogate.

## 6. Stage 3 — advisor: одна ветвь

### 6.1 Разбиение validation

Используется partition, запечатанный до Stage 1. `advisor_select` применяется
для branch recommendation; `advisor_shadow` открывается ровно один раз после
freeze candidate. Нельзя выбирать ветвь и оценивать её на тех же samples.

### 6.2 Методы выбора

При одинаковом candidate set и cost:

1. random branch, много фиксированных seeds;
2. exhaustive single-branch fit;
3. OMP/DOMP;
4. group-LASSO;
5. proposed stable correlation-guided advisor.

### 6.3 Fit

- complex ridge/QR;
- regularization выбирается на calibration/train inner split; если для этого
  нужен `advisor_select`, внутри него заранее фиксируются непересекающиеся
  fit/select blocks;
- existing active coefficients можно freeze либо refit; это две отдельные
  ablations;
- amplitude/peak guard включён;
- test закрыт.

### 6.4 Метрики

- left/right/average adjacent leakage;
- absolute adjacent power;
- integrated OOB;
- main-band power change;
- worst spectral bin и worst frame;
- EVM и NMSE;
- max \(|z|\), PAPR, clipping count;
- operation and memory delta;
- calibration samples/time.

### Gate S3

Advisor успешен, если:

- его ветвь находится в **до запуска заданном** tolerance от exhaustive best;
- improvement над random имеет заранее заданный effect size, confidence
  level и capture count;
- shadow spectral objective улучшен на preregistered \(\delta\);
- ни один safety constraint не нарушен;
- вывод согласован между evaluators либо подтверждён physical PA.

Если numerical thresholds из §2.1 не определены, Gate S3 остаётся `BLOCKED`.

## 7. Stage 4 — sparse residual adaptation

### 7.1 Ablation ladder

\[
B_\mathrm{extra}=0\rightarrow1\rightarrow2\rightarrow3\rightarrow4.
\]

На каждом шаге:

- не более одной новой группы;
- re-evaluate all safety metrics;
- publish marginal quality per marginal measured time;
- stop при малом/нестабильном validation gain.

### 7.2 Архитектуры

- fixed delays \(\{0\}\);
- \(\{0,1\}\);
- \(\{0,1,2\}\);
- correlation-guided extra spline;
- selected GMP/dynamic-deviation;
- state branch только после slow-state evidence;
- tiny residual NN только после structured Pareto saturation.

### 7.3 Selection baselines

- fixed architecture;
- exhaustive small search;
- OMP/DOMP;
- group-LASSO/sparse Bayesian if numerically feasible;
- proposed cost-aware correlation;
- learned routing only after multi-condition data.

### Gate S4

Freeze topology when:

- no candidate produces material validation spectral gain per timing cost;
- peak/PAPR begins to dominate;
- evaluator ranking diverges;
- selected group unstable across frames.

Only then create sealed test release decision. Test opened once.

## 8. Stage 5 — operating-point robustness

### 8.1 Required capture grid

Same physical PA:

- at least three output-power points;
- at least three stabilized temperatures;
- at least two bandwidths;
- at least two waveforms/modulations;
- repeated captures per cell;
- controlled transitions power/temperature;
- PA instance id and chronological order.

DPA_200MHz and APA_200MHz are different physical systems and remain separate
experiments.

### 8.2 Experiments

For each axis:

1. zero-shot fixed coefficients;
2. nearest verified coefficient bank;
3. limited-sample ridge/RLS recalibration;
4. slow-state spline;
5. frame expert router;
6. low-rank/hypernetwork only after 1–5.

Curves:

- quality vs calibration samples;
- quality vs acquisitions;
- quality vs wall-clock;
- quality vs distance from training operating point;
- forgetting/retention when returning to old point.

### Gate S5

Advanced adaptation is retained only if it beats warm-start/nearest-bank/direct
ridge at equal samples and time on a held-out operating condition.

## 9. Stage 6 — physical PA protocol

### 9.1 Acquisition order

Use заранее сгенерированный counterbalanced/randomized paired order while
controlling thermal history. Candidate не должен всегда следовать после
baseline. Для каждого candidate block:

1. stabilization/washout;
2. bracketing no-DPD reference;
3. randomized baseline/candidate order;
4. второй bracketing no-DPD reference;
5. повтор sequence в новом random order.

### 9.2 Controls

- equal defined main-band/output power, not just equal DAC scaling;
- same carrier, bandwidth, waveform and observation chain;
- logged temperature/bias/output power;
- fixed attenuators and receiver gain;
- calibration before/after;
- blind file names for metric computation;
- amplitude limiter and emergency fallback.

### 9.3 Measurements

Baseband:

- left/right adjacent leakage;
- integrated OOB;
- PSD/worst bin;
- EVM/NMSE;
- AM/AM, AM/PM;
- peak/PAPR.

RF analyzer:

- exact Huawei-defined bands when supplied;
- discrete intermodulation components;
- \(2f_c/3f_c\) harmonics only if RF front end covers them;
- RBW/VBW/detector/averaging recorded.

### 9.4 Statistics

- paired differences by capture;
- bootstrap confidence intervals by complete frames/captures;
- median and worst case;
- drift covariate;
- failed/aborted captures retained in log.

### Gate S6

Only `LOCAL-PHY` evidence permits physical linearization claims. Improvement
must hold without main-band loss or unsafe drive.

## 10. Stage 7 — fixed-point

Formats:

- FP32 reference;
- FP16-like diagnostic;
- signed 16-bit;
- signed 14-bit;
- signed 12-bit if range permits.

For each:

- input, coefficient, state, interpolation and output scale;
- rounding mode;
- accumulator width;
- saturation count;
- LUT/address behavior;
- state limit cycles;
- streaming chunk equivalence;
- spectral degradation and peak change.

Current sealed runner:

```bash
python experiments/evaluate_fixed_point_dpd.py \
  --config experiments/configs/dpd_fixed_point_dpa200_validation.json
```

```bash
python experiments/evaluate_fixed_point_dpd.py \
  --config experiments/configs/dpd_fixed_point_apa200_validation.json
```

Эти commands проверяют текущий core; новый candidate требует новых output
directories/config hashes и не может перезаписывать sealed artifacts.

### Gate S7

- no overflow outside declared saturation behavior;
- spectral degradation не превышает preregistered \(\delta_\mathrm{fixed}\);
- no new worst-bin violation;
- exact full-vs-chunk equality;
- fixed-point operation schedule frozen.

Без \(\delta_\mathrm{fixed}\) Gate S7 имеет статус `BLOCKED`, даже если
integer replay численно близок к float.

## 11. Stage 8 — временная стоимость

Измеряется только deployed DPD, observer/calibration отдельно.

### 11.1 Analytical ledger

- real MUL;
- real ADD;
- comparisons/branches;
- nonlinear functions;
- LUT reads;
- state/coefficient/input reads and writes;
- stored coefficients/constants/state;
- look-ahead and buffer.

### 11.2 Host diagnostic

Existing command:

```bash
python experiments/benchmark_dpd_timing.py --help
```

Измерять:

- batch/streaming;
- chunk 1, 8, 64, 512;
- pinned core;
- warm-up;
- median, p95, p99, maximum;
- samples/s и latency/sample.

Host diagnostic не является acceptance.

### 11.3 Target benchmark

На target hardware одна программа запускает:

1. reference kernel из 1000 real MUL с согласованной bit width и memory
   pattern;
2. DPD kernel;
3. empty/IO overhead.

Публикуются cycles/sample, initiation interval, clock, latency, throughput,
BRAM/DSP/LUT/FF, energy/sample при наличии. Также фиксируются dependency
graph, bit widths, memory placement и полный IO/state path: тысяча независимых
MUL может распараллеливаться иначе, чем зависимая цепочка LUT/sqrt/state.

### Gate S8

DPD проходит временной budget только если sustained throughput и high-quantile
latency не хуже согласованного reference при полном IO/state/LUT path. Пока
Huawei не предоставила reference kernel либо absolute cycle/sample deadline,
это только внутреннее proxy comparison и Gate S8 остаётся `BLOCKED` для
customer acceptance.

## 12. Stage 9 — безопасный online update

Тесты controller:

- known-good update accepted;
- deliberately bad spectrum rejected;
- excessive peak rejected;
- corrupted/NaN coefficients rejected;
- incomplete bank write cannot become active;
- rollback returns exact previous bank;
- rapid detector oscillation suppressed hysteresis;
- observation receiver failure invokes safe fallback.

Update выполняется атомарно на frame boundary. Coefficient banks versioned и
hash-checked.

## 13. Test opening policy

Test открывается только после:

- topology freeze;
- regularization freeze;
- spectral masks freeze;
- alignment/gain freeze;
- bit widths freeze;
- operation schedule freeze;
- signed release record.

После test:

- никакого повторного tuning;
- отрицательный result публикуется;
- новый architecture начинается с нового untouched test capture.

## 14. Общая comparison table

Для каждого:

- no DPD;
- MP-DPD;
- GMP-DPD;
- memoryless spline;
- spline \(\{0,1\}\);
- spline \(\{0,1,2\}\);
- sparse spline;
- residual advisor;
- slow-state spline;
- expert bank;
- tiny residual/SSM, если достигнут соответствующий gate;
- OpenDPD DPD.

Колонки:

```text
dataset
physical_pa_or_evaluator
evidence_class
left_leakage
right_leakage
average_leakage
absolute_suppression
relative_dbc_improvement
integrated_oob
worst_bin
main_band_change
evm
nmse
peak
papr
parameters
state
mul
add
nonlinear
lut
memory_access
measured_latency
throughput
coefficient_memory
calibration_samples
calibration_time
fixed_point_degradation
causal
look_ahead
evaluator_dependence
physical_evidence
```

Строки разных PA/datasets не ранжируются в одну quality league.

## 15. Ближайшие три задачи

1. Реализовать только read-only observer diagnostic и его statistical unit
   tests; frozen model не изменять.
2. Подготовить high-fidelity OpenDPD PA evaluator training с отдельным
   checkpoint selection и GPU environment.
3. Согласовать с Huawei physical spectral bands и target timing kernel.

Самый информативный ближайший эксперимент — Stage 1. Он дешёв, не загрязняет
test и отвечает, существует ли вообще воспроизводимая пропущенная структура,
ради которой стоит усложнять DPD.
