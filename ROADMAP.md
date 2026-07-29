# Roadmap: two-loop PA identification and DPD

Дата обновления: 2026-07-29.

Главное изменение относительно первого этапа: дальнейшее улучшение DPD
приостановлено, пока независимый PA evaluator не станет заметно точнее
оцениваемого DPD residual. Legacy memory-polynomial surrogate первого этапа
имел test fidelity около −30…−31 dB, а spline-memory DPD уже достигал до
−32.7 dB на том же evaluator. Новый validation-selected causal GMP достиг на
frozen test −35.385 dB на DPA и −38.608 dB на APA при 766/954 counted real
MUL/sample. Однако arithmetic margin относительно старых surrogate-only DPD
residuals равен лишь 5.521/5.867 dB. Поскольку true cascade через новые frozen
GMP evaluators ещё не запускался и нет второго independent evaluator/physical
PA measurement, Gate A→B остаётся закрытым.

Следующий standalone candidate теперь также проверен: APA SPH (`K=32,L=8`)
даёт 37 MUL/sample, но −30.4024 dB train-OOF NMSE — на 6.652 dB хуже
matched MP и на 7.943 dB хуже GMP. Поэтому он остаётся дешёвым отрицательным
control point, а не новым evaluator. Выполненный затем non-factorized sparse
spline-memory PA дал −32.0300 dB train-OOF при 54 MUL/58 ADD, но остался на
5.024/6.315 dB хуже MP/GMP (full) и также не стал evaluator. Его residual peak
смещает следующий hypothesis к lag 9, а не к дополнительным knots.

## Правила выполнения

- Каждая строка roadmap реализуется отдельным небольшим commit и сразу
  отправляется в `git@github.com:theJorDea/DPD.git`.
- Architecture/hyperparameter selection использует только train/validation.
- Test не используется для tuning и открывается после freeze.
- DPA_200MHz и APA_200MHz не объединяются: это разные PA/captures.
- Surrogate, cross-surrogate и physical-PA результаты маркируются раздельно.
- Parameter count никогда не подменяет operation, state-memory или latency.
- Model/evaluator code, experiment config, numerical result и report update
  остаются разными small tasks: один небольшой commit на один проверяемый
  результат, затем немедленный push. Formal sweep не объединяется с изменением
  evaluator или модели.

## Status snapshot

Текущий snapshot включает frozen GMP results и два завершённых APA
linear-residual ablations:

- MP forward baseline, frozen test и residual analysis для DPA/APA завершены;
- causal factorized GMP kernel, exact cost/state counter, ridge/truncated-SVD
  calibration и unit tests завершены;
- validation-only GMP selector завершён: full-record score является primary,
  common-interior score сохраняется отдельно, strict `<1000 MUL` применяется
  до fit;
- A0/A1 sensitivity выполнена; A0 integer-only/no-fractional-transform frozen
  для DPA и APA до formal GMP selection;
- formal GMP selection, coefficient-OOF residual audit и separate release
  gate выполнены для обоих datasets;
- оба release gates прошли все hard predicates и разрешили по одному frozen
  test; разрешения израсходованы, test refit/retry не выполнялся;
- DPA/APA frozen GMP test full-record pooled NMSE: −35.385/−38.608 dB;
- integrity-gated frozen runner поддерживает MP и GMP, включая проверку полного
  `gmp_config`, hashes, operation count и common cooldown до чтения test;
- APA GMP residual содержит воспроизводимую pseudo-correlation с
  `conj(x[n-d])` на каузальных лагах 0…3, но bounded widely-linear residual
  audit дал максимум 0.0273/0.0305 dB full/common OOF gain и не прошёл
  preregistered 0.1 dB gate; frozen decision — `no_correction`;
- widely-linear audit имеет evidence class
  `post_discovery_internal_resampling_only`: validation уже использовался
  как descriptive split, test не читался и не хэшировался;
- causal proper-complex correlation около лагов 43…48 была проверена nested
  sparse-FIR supports; лучший `{44,45,46}` дал лишь 0.0182/0.0201 dB
  full/common OOF gain при 966 MUL, поэтому снова выбран `no_correction`;
- long-FIR audit также не читал и не хэшировал test; выбранная Pareto-точка
  остаётся исходным APA GMP: 954 MUL / 947 ADD / 236 state reals;
- standalone APA SPH search завершён train-only: selected `K=32,L=8` имеет
  −30.4024 dB full OOF NMSE при 37 MUL/36 ADD, fails the ≤3 dB cheap-Pareto
  loss gate versus MP and is classified `neither_evaluator_nor_cheap_pareto`;
- SPH `K=48/64` raw-score variants были отклонены из-за rank-deficient
  control designs (`47/48` и `62–63/64`), поэтому knots дальше не расширяем;
- SPH residual proper-correlation peak 22–24 samples воспроизведён на train
  OOF и reused validation; slow-state branch остаётся запрещён при
  `independent_capture_count=0`;
- non-factorized sparse spline-memory PA search завершён train-only: selected
  `mixed_diagonal_long`, `K=12`, 6 branches, −32.0300/−32.0882 dB full/common
  OOF при 54 MUL/58 ADD; hard rank/support/condition/streaming gates прошли,
  но cheap-Pareto и evaluator gates не прошли;
- sparse residual воспроизводит causal proper-correlation peak lag 9
  (`0.69064` train OOF, `0.69131` reused validation); envelope/state evidence
  не оправдывает slow-state branch;
- versioned frame-safe fractional-alignment transform остаётся sensitivity
  tool, а не доказанным measurement-path de-embedding.

Следовательно, GMP forward-identification result завершён, но release-gate
PASS нельзя смешивать с Gate A→B. DPD quality, fixed-point, robustness и
physical-PA linearization этим результатом не доказаны.

## Этап 0. Requirements contract

Статус: завершён документально.

- [x] Отделить явные требования слайдов от интерпретации.
- [x] Зафиксировать фактический OpenDPD contract.
- [x] Составить список неизвестных критериев Huawei.
- [ ] Получить ответы владельца требований на вопросы из `REQUIREMENTS.md`.

Артефакт: `REQUIREMENTS.md`.

## Этап A1. Единый PA-model evaluator

Статус: завершён для floating-point forward PA baseline.

Один неизменяемый путь:

```text
x_split -> PA model -> y_hat_split -> compare with measured y_split
```

Общие для всех моделей:

- frozen integer alignment, estimated on train only;
- explicit complex-LS и OpenDPD-peak gain diagnostics;
- dataset `nperseg` framing и zero-state boundary policy;
- pooled NMSE, OpenDPD NMSE, error PSD, AM/AM и AM/PM residual;
- fit time, inference latency/throughput, real MUL/ADD/nonlinear/LUT;
- stored coefficients/state bytes;
- fixed-point interface and streaming chunk equivalence.

Реализовано:

- train-frozen alignment/gain/framing protocol;
- pooled и OpenDPD-compatible NMSE, spectral и AM/AM/AM/PM diagnostics;
- common-warmup score и отдельный validation-frozen cooldown-aware interior
  diagnostic для latency-bearing GMP;
- отдельный integrity-gated test command, который проверяет hashes до первого
  чтения test;
- MP/GMP model dispatch, проверка полного frozen topology/operation contract;
- unit tests на sealed test, frame reset и отсутствие post-hoc gain/delay fit.

Основные файлы: `baseline/pa_benchmark.py`,
`experiments/evaluate_frozen_pa.py`, `tests/test_pa_benchmark.py`,
`tests/test_evaluate_frozen_pa.py`.

## Gate A0/A1. Fractional-alignment sensitivity до formal GMP

Статус: завершён. Numerical comparison использовал только train/validation;
для обоих datasets frozen variant — A0. Decision artifact:
`experiments/results/pa_alignment_protocol_decision.json`.

Сравниваются две заранее определённые protocol variants:

- **A0** — текущий integer-only frame protocol без fractional transform;
- **A1** — тот же frame protocol с явно заданным и frozen по train residual
  delay, versioned frame-safe windowed-sinc transform и symmetric valid crop.

`baseline/fractional_alignment.py` не оценивает measurement-path delay и не
является автоматическим de-embedding. Он только детерминированно применяет
заданный frozen transform, независимо внутри каждого frame, без circular
convolution. Без independent feedback/loopback calibration A1 остаётся
sensitivity analysis.

Gate protocol:

1. До чтения validation зафиксировать источник delay, sign convention, FIR
   coefficients/hash, guard, один fixed PA recipe и solver.
2. Fit выполняется раздельно на train для A0 и A1; test не существует в control
   path sensitivity runner.
3. A0 и A1 сравниваются на validation на одном и том же допустимом support:
   A0 получает тот же symmetric guard crop, что A1.
4. Primary decision metric — pooled complex NMSE на common-interior support.
   Для каждой проверки заранее определяется
   \(\Delta=\mathrm{NMSE}_{A1}-\mathrm{NMSE}_{A0}\) в dB, поэтому отрицательное
   значение означает улучшение A1.
5. A1 принимается только если одновременно:
   - fixed GMP recipe даёт \(\Delta_\mathrm{OOF}\le-0.25\) dB и
     \(\Delta_\mathrm{validation}\le-0.25\) dB на common interior;
   - fixed MP corroboration даёт \(\Delta_\mathrm{OOF}\le0\) и
     \(\Delta_\mathrm{validation}\le0\) на common interior;
   - соответствующие full-record deltas имеют тот же знак улучшения
     (\(\Delta\le0\)).
   Любое нарушение, missing/non-finite metric или ничья вне этих условий
   детерминированно выбирает A0.
6. Сохранить pooled NMSE, OpenDPD-compatible NMSE, common-interior NMSE,
   boundary residual и rank/conditioning; DPA и APA анализировать отдельно.
7. Protocol decision и его hashes фиксируются отдельным commit до formal GMP.
   Уже pre-registered A0 configs нельзя незаметно переопределить; для A1 нужны
   новые versioned configs.

Observed result:

| Dataset | GMP OOF common Δ A1−A0 | GMP validation common Δ | Full-record caveat | Frozen |
|---|---:|---:|---|---|
| DPA_200MHz | −0.00114 dB | −0.00017 dB | same sign, но далеко от −0.25 dB gate | A0 |
| APA_200MHz | −0.00841 dB | −0.00654 dB | OOF sign reversal: +0.01070 dB | A0 |

Ни один dataset не прошёл preregistered A1 threshold. Это решение означает
«не вводить fractional transform», а не «measurement path откалиброван».

## Этап A2. PA baselines

Статус: MP, causal GMP, standalone APA SPH и non-factorized sparse PA
завершены; локального runnable OpenDPD neural checkpoint нет. SPH и sparse PA
отклонены как quality evaluators.

Порядок:

1. [x] Memory Polynomial с validation grid order/memory/ridge.
2. GMP:
   - [x] causal factorized kernel, exact operations/state и streaming tests;
   - [x] column-scaled ridge и rank-controlled truncated-SVD calibration;
   - [x] validation-only selector и explicit full-record/common-interior metrics;
   - [x] pre-registered DPA/APA causal configs;
   - [x] integrity-gated GMP frozen runner;
   - [x] пройти Gate A0/A1 и freeze protocol variant A0;
   - [x] выполнить formal validation sweep;
   - [x] выполнить residual/OOF stability checks;
   - [x] freeze winner и открыть test один раз после release-gate PASS.
3. [ ] OpenDPD PA backbone/checkpoint, только если checkpoint доступен или
   воспроизводимо обучен.
4. [x] Spline/CPWL memoryless nonlinearity + short complex FIR (APA SPH;
   отрицательный train-OOF result, не evaluator).
5. [x] Non-factorized sparse complex spline-memory PA с bounded branch/delay
   dictionary, мотивированным residual peak 22–24; APA run завершён, quality
   gate отклонён.
6. [ ] State-conditioned variant только при residual evidence slow state.

Каждый baseline разбивается минимум на отдельные commits: model + tests;
pre-registered config; numerical result artifact; report update. Нельзя менять
evaluator одновременно с моделью или объединять config и result.

Текущие completed results:

| Dataset | Selected model | Validation pooled NMSE | Frozen test pooled NMSE | MUL/sample |
|---|---|---:|---:|---:|
| DPA_200MHz | odd orders 1…9, 24 delays | −34.962 dB | −35.099 dB | 792 |
| DPA_200MHz | causal GMP `both_k4_m1`, `ka7/la24` | −35.366 dB | −35.385 dB | 766 |
| APA_200MHz | powers 1…5, 30 delays | −37.095 dB | −36.990 dB | 960 |
| APA_200MHz | causal GMP `both_k2_m2`, `ka7/la30` | −38.665 dB | −38.608 dB | 954 |

GMP configs `experiments/configs/pa_gmp_dpa200.json` и
`experiments/configs/pa_gmp_apa200.json` фиксируют bounded causal grid,
OpenDPD-compatible initial truncated-SVD `rcond=1e-4`, ridge/SVD refinement
axes, primary full-record validation score и strict exclusive multiplier
budget. Они были выполнены ровно по frozen variant A0; selection manifests и
one-shot test artifacts находятся в `experiments/results/pa_gmp_*`.

## Этап A3. Residual analysis

Статус: завершён для выбранных MP и GMP models; повторяется после каждого
нового frozen PA baseline до любого test release.

На validation residual \(e[n]=y[n]-\hat y[n]\) измерить:

- complex correlation с lagged I/Q;
- correlation с \(|x|\), \(|x|^2\) и nonlinear envelope terms;
- slow one-pole/block envelope averages на нескольких time scales;
- error power/phase против position inside segment;
- error в high-amplitude/compression quantiles;
- residual PSD и left/right adjacent regions;
- cross-correlation между residual и model output/input.

Результат должен выбирать следующий inductive bias:

- short lag structure → memory branches/GMP;
- linear post-memory → spline + FIR;
- slow envelope state → state-conditioned spline;
- unexplained Cartesian asymmetry → feedback IQ/widely-linear audit;
- отсутствие устойчивого pattern → не добавлять параметры.

Артефакты: code + `PA_MODEL_BENCHMARK.md`, отдельный residual JSON/CSV.

Matched GMP residual audit подтвердил improvement не только на validation, но
и coefficient-OOF: 0.295/0.301 dB full/common на DPA и 1.291/1.651 dB на APA.
Все folds full rank; support и OOF→validation preregistered limits выполнены.
На APA full/common OOF отличаются на 0.405 dB, поэтому boundary/reset semantics
остаются существенными. High-amplitude diagnostics не дают достаточного
основания для compression-only branch. `independent_capture_count=0`, поэтому
thermal/state-conditioned ветвь остаётся заблокированной.

Для APA strongest low-cost causal evidence после GMP — pseudo-complex
correlation ошибки с входом: её magnitude на лагах 0, 1, 2, 3
равна 0.0832, 0.0816, 0.0702, 0.0517 на train coefficient-OOF и
0.0678, 0.0663, 0.0559, 0.0387 на validation. Это мотивирует
минимальную conjugate-FIR correction, но не доказывает её PA origin:
такую же структуру может дать IQ imbalance или feedback-path asymmetry.
Отрицательные лаги в residual report означают future samples и запрещены
как inference features.

Важно: и train OOF residual, и validation residual уже просмотрены
при выборе widely-linear family. Поэтому выполненный запуск имеет статус
`post_discovery_internal_resampling_only`; validation не называется
independent confirmation, а test не читался. Все четыре supports дали лишь
0.0248…0.0273 dB full и 0.0296…0.0305 dB common OOF gain при 958…974
MUL/sample. Ни один не достиг 0.1 dB full/common threshold, поэтому выбран
`no_correction`. Для acceptance нужен новый capture/operating point.

Следующий nested ablation проверил отдельно proper-complex linear memory.
Train-OOF и reused validation показывали согласованный correlation peak около
causal lag 45, но explained error power снова оказался мал:

| Proper FIR support | OOF gain full/common | Minimum fold full/common | MUL |
|---|---:|---:|---:|
| `{45}` | 0.0133/0.0147 dB | 0.0083/0.0093 dB | 958 |
| `{44,45,46}` | **0.0182/0.0201 dB** | 0.0114/0.0127 dB | 966 |
| `{43,…,48}` | 0.0177/0.0199 dB | 0.0115/0.0144 dB | 978 |
| `{42,…,49}` | 0.0177/0.0201 dB | 0.0114/0.0144 dB | 986 |

Все folds дали положительный gain, были full rank и streaming/reset exact, но
ни один support не достиг 0.1 dB gate. Поэтому не следует расширять GMP ещё
большим linear delay grid или превращать эти же lags в spline branches без
отдельного nonlinear evidence. Такой standalone phase-equivariant
spline/CPWL + short-FIR PA (SPH) уже проверен: его 37-MUL factorized form
проиграла matched MP на 6.652 dB. Следующий experiment должен добавить только
недостающую degree of freedom — residual-guided lag-9 branch neighborhood —
с жёстким budget/OOF gate. Уже выполненный sparse run показал, что topology
только с lag 22–24 недостаточна.

### A2.5 Sparse non-factorized PA result

Preregistered config `experiments/configs/pa_sparse_spline_memory_apa200.json`
выбрал topology `(0,0),(1,1),(2,2),(22,22),(23,23),(24,24)`, `K=12`, ridge
`0`. Train OOF score — `−32.030011/−32.088250 dB` full/common; full-train
refit — `−32.049190 dB`; reused validation — `−32.048219 dB`. Exact cost:
`54 MUL`, `58 ADD`, 6 sqrt, 24 comparisons, 12 LUT accesses, 144 coefficient
reals and 48 state reals. Rank `72/72`, condition `78.57`, minimum feature
support `8`; streaming/reset checks exact.

The candidate improves SPH by about `1.628 dB`, but loses to matched MP by
`5.024/5.012 dB` and to matched GMP by `6.315/6.662 dB` full/common. Decision
`neither_evaluator_nor_cheap_pareto`; Gate A→B remains closed. The immutable
bundle is `experiments/results/pa_sparse_spline_memory_apa200_selection/` and
the execution took `33.5888 s` before publication without test access.

The next local experiment must be a separately preregistered lag-9 neighborhood
(`{8,9,10}` or an explicitly justified subset of signal/envelope pairs), not
an unbounded delay/K sweep. If that bounded family also misses the GMP gap,
stop local surrogate expansion and prioritize `APA_200MHz_b`/physical capture.

## Ближайшая точная последовательность

Каждый пункт ниже — отдельный small-task commit с тестом/проверкой и push:

1. [x] Обновить `PA_MODEL_BENCHMARK.md` и quantitative Gate A→B assessment
   после one-shot GMP tests.
2. [x] Синхронизировать normative `EXPERIMENT_PLAN.md`; устаревший
   `experiments/experiment_plan.md` явно пометить historical ledger либо
   обновить, чтобы не было двух противоречащих status sources.
3. [x] Создать living skeletons обязательных документов
   `DPD_BENCHMARK.md`, `ROBUSTNESS_AND_ADAPTATION.md`, `HARDWARE_COST.md` и
   `FINAL_GAP_ANALYSIS.md`, отделяя completed evidence от planned work.
4. [x] Зафиксировать residual-selected APA widely-linear/IQ hypothesis,
   exact support/cost limits и reused-validation status до candidate fit; test запрещён.
5. [x] Реализовать two-stage conjugate residual correction + unit tests: causality,
   frame reset, streaming equivalence, exact operation count и deterministic complex fit.
6. [x] Выполнить APA leave-one-frame-out internal audit при strict `<1000 MUL`;
   validation показать только как already-viewed descriptive split, test не читать.
7. [x] Применить preregistered decision: audit threshold не прошёл, поэтому
   feedback-path/IQ attribution не заявлять и DPA tuned conjugate support не
   запускать.
8. [x] Preregister, реализовать и выполнить nested proper-complex sparse FIR
   audit около causal lags 42…49 без test access.
9. [x] Применить frozen fallback: лучший gain 0.0182/0.0201 dB ниже 0.1 dB,
   поэтому не сохранять long-FIR correction.
10. [x] Preregister, реализовать и выполнить bounded standalone
   `spline/CPWL memoryless nonlinearity + short complex FIR` PA audit с exact
   cost, identifiability и streaming contract; APA SPH отклонён по OOF gate.
11. [x] Preregister, implement and execute bounded non-factorized sparse
   spline-memory PA; selected result fails quality/evaluator gates.
12. [ ] Preregister and run the residual-guided lag-9 neighborhood with the
   same train-only OOF protocol; stop if the cheap-Pareto/evaluator gate fails.
13. [ ] Проверить surviving family на reused validation и новом independent
   capture/operating point; не использовать старый APA test для selection.
14. [ ] Только после Gate A→B PASS preregister cross-evaluator DPD benchmark.

## Gate A→B

DPD optimization возобновляется, только если:

1. PA model выбран по validation и frozen до test;
2. PA fidelity имеет provisional margin не менее 10 dB по error power
   относительно DPD residual, который планируется различать;
3. минимум два independently fitted evaluator variants дают одинаковый ranking;
4. predistorted drive остаётся внутри проверенного PA-model support;
5. boundary/streaming semantics совпадают с deployment protocol.

Margin 10 dB — временный внутренний conservative research criterion, а не
восстановленное со слайдов требование Huawei. Он подлежит замене после
уточнения acceptance protocol или после physical-PA cross-validation.

Если gate не выполнен, DPD numbers остаются diagnostic surrogate-only.

Текущий projected margin — subtraction frozen forward GMP fidelity и старого
surrogate-only DPD residual; это не завершённый cascade experiment:

| Dataset | Frozen GMP PA test fidelity | Existing spline-DPD residual | Projection margin | Shortfall to 10 dB |
|---|---:|---:|---:|---:|
| DPA_200MHz | −35.385 dB | −29.864 dB | 5.521 dB | 4.479 dB |
| APA_200MHz | −38.608 dB | −32.741 dB | 5.867 dB | 4.133 dB |

Existing spline DPD ещё не прогонялся как cascade через новые frozen GMP
models; таблица служит только gate arithmetic и не является DPD result.
Release-gate PASS разрешал test evaluation frozen GMP, но не заменяет этот
Gate A→B. Текущий decision: **closed**.

## Этап B1. Frozen-evaluator DPD benchmark

Единственный deployment-like test path:

```text
desired x_test -> frozen DPD -> frozen independent PA/physical PA
               -> compare with frozen g * x_test
```

Measured \(y_\mathrm{test}\) не используется как DPD input. Путь с
\(y_\mathrm{test}/g\) допустим только как отдельно маркированный ILA
inverse/postdistorter diagnostic и не является DPD cascade result.

Сравнить без изменения evaluator:

- no DPD;
- MP/GMP DPD;
- OpenDPD reference;
- memoryless complex spline;
- spline signal delays 0,1;
- spline signal delays 0,1,2;
- SPH: complex spline + short FIR;
- validation-selected sparse branch model.

Выход — Pareto frontier, а не один winner:

- pooled/OpenDPD NMSE;
- EVM variants;
- ACLR left/right/average;
- output PSD и error PSD;
- AM/AM и AM/PM residuals;
- PAPR и peak predistorted amplitude;
- numerical/streaming stability и violations проверенного drive support;
- real MUL/ADD/nonlinear/LUT/state/coefficient memory;
- calibration time и inference timing.

Артефакт: `DPD_BENCHMARK.md`.

## Этап C. Robustness and adaptation

- DPA_200MHz и APA_200MHz анализируются отдельно.
- Same-waveform/different-PA experiments маркируются как PA transfer, не как
  обычный test.
- При наличии captures разных power levels:
  train operating point A → test B → recalibration with N samples.
- Learning curves:
  \(N=\{64,128,256,512,1024,2048,\ldots\}\), quality versus fit wall-clock.
- Для stochastic methods использовать минимум 3 seeds и публиковать разброс.
- При наличии подходящих captures отдельно проверять waveform и bandwidth
  generalization, не смешивая смену waveform, PA и operating point в один
  неидентифицируемый split.
- Отдельно оценить coefficient drift, maximum update rate и stability.

Артефакт: `ROBUSTNESS_AND_ADAPTATION.md`.

## Этап D. Hardware reference

- Bit-accurate simulator для лучших validation-selected PA и DPD models, а не
  numerical “fixed-point-like” approximation.
- Formats: signed 16, 14 и 12 bit coefficients/activations.
- Explicit accumulator/state widths, scaling, rounding and saturation.
- Отдельно проверять input quantization и coefficient/LUT interpolation error.
- Для каждого формата публиковать degradation относительно FP32 как минимум по
  NMSE, EVM, ACLR и peak/stability indicators.
- Separate counts: real MUL, ADD, sqrt/nonlinear, comparisons, LUT, coefficient
  and state memory.
- Full-record versus arbitrary streaming chunks must be equivalent for causal
  models; отдельный no-future-dependence test подтверждает causality.
- Измерять latency/throughput на выбранном target; analytical count помечать
  как lower bound.

Артефакт: `HARDWARE_COST.md`.

## Этап E. Final gap analysis

`FINAL_GAP_ANALYSIS.md` отвечает:

- что доказано measured data;
- что доказано только на surrogate;
- как PA evaluator ограничивает выводы;
- каких Huawei acceptance details и physical experiments не хватает;
- какой следующий эксперимент даёт максимум information gain.

До physical predistorted remeasurement на том же PA формулировка
«лучше OpenDPD для базовой станции Huawei» запрещена.
