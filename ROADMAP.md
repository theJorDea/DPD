# Roadmap: two-loop PA identification and DPD

Дата обновления: 2026-07-29.

Главное изменение относительно первого этапа: дальнейшее улучшение DPD
приостановлено, пока независимый PA evaluator не станет заметно точнее
оцениваемого DPD residual. Legacy memory-polynomial surrogate первого этапа
имел test fidelity около −30…−31 dB, а spline-memory DPD уже достигал до
−32.7 dB на том же evaluator. Новый validation-selected MP forward model
улучшил measured-output fidelity до −35.10 dB на DPA и −36.99 dB на APA,
но provisional 10 dB evaluator margin всё ещё не выполнен.

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

Текущий snapshot после `fc85e44`, `5047146`, `0a7e065` и `a6e4689`:

- MP forward baseline, frozen test и residual analysis для DPA/APA завершены;
- causal factorized GMP kernel, exact cost/state counter, ridge/truncated-SVD
  calibration и unit tests завершены;
- validation-only GMP selector завершён: full-record score является primary,
  common-interior score сохраняется отдельно, strict `<1000 MUL` применяется
  до fit;
- causal GMP sweep configs для DPA/APA pre-registered, но **formal GMP sweep
  ещё не запускался**;
- integrity-gated frozen runner поддерживает MP и GMP, включая проверку полного
  `gmp_config`, hashes, operation count и common cooldown до чтения test;
- versioned frame-safe fractional-alignment transform и его unit tests
  реализованы, но численный A0/A1 sensitivity experiment ещё не запускался.

Следовательно, completed GMP quality/result rows пока отсутствуют. Наличие
kernel, selector, configs и frozen runner означает готовность инфраструктуры,
а не измеренный GMP result.

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

Статус: transform и tests завершены; runner, numerical comparison и protocol
decision не выполнены. Этот gate блокирует formal GMP sweep.

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
4. Сохранить pooled NMSE, OpenDPD-compatible NMSE, common-interior NMSE,
   boundary residual и rank/conditioning; DPA и APA анализировать отдельно.
5. A1 принимается как formal-sweep protocol только при устойчивом улучшении,
   которое не объясняется crop/reset boundary. При малом или нестабильном
   различии остаётся A0.
6. Protocol decision и его hashes фиксируются отдельным commit до formal GMP.
   Уже pre-registered A0 configs нельзя незаметно переопределить; для A1 нужны
   новые versioned configs.

## Этап A2. PA baselines

Статус: MP завершён; GMP implementation/selection infrastructure завершена,
formal sweep не запущен и ожидает Gate A0/A1; локального OpenDPD checkpoint
нет.

Порядок:

1. [x] Memory Polynomial с validation grid order/memory/ridge.
2. GMP:
   - [x] causal factorized kernel, exact operations/state и streaming tests;
   - [x] column-scaled ridge и rank-controlled truncated-SVD calibration;
   - [x] validation-only selector и explicit full-record/common-interior metrics;
   - [x] pre-registered DPA/APA causal configs;
   - [x] integrity-gated GMP frozen runner;
   - [ ] пройти Gate A0/A1 и freeze protocol variant;
   - [ ] выполнить formal validation sweep;
   - [ ] выполнить residual/OOF stability checks;
   - [ ] freeze winner и открыть test один раз.
3. [ ] OpenDPD PA backbone/checkpoint, только если checkpoint доступен или
   воспроизводимо обучен.
4. [ ] Sparse complex spline-memory PA.
5. [ ] Spline/CPWL memoryless nonlinearity + short complex FIR.
6. [ ] State-conditioned variant только при residual evidence slow state.

Каждый baseline получает самостоятельный commit: model + tests, затем config +
result artifact. Нельзя менять evaluator одновременно с моделью.

Текущие MP results:

| Dataset | Selected model | Validation pooled NMSE | Frozen test pooled NMSE | MUL/sample |
|---|---|---:|---:|---:|
| DPA_200MHz | odd orders 1…9, 24 delays | −34.962 dB | −35.099 dB | 792 |
| APA_200MHz | powers 1…5, 30 delays | −37.095 dB | −36.990 dB | 960 |

GMP configs `experiments/configs/pa_gmp_dpa200.json` и
`experiments/configs/pa_gmp_apa200.json` фиксируют bounded causal grid,
OpenDPD-compatible initial truncated-SVD `rcond=1e-4`, ridge/SVD refinement
axes, primary full-record validation score и strict exclusive multiplier
budget. Они имеют variant A0 и пока не запускались.

## Этап A3. Residual analysis

Статус: завершён для выбранных MP models; повторяется после каждого нового
frozen PA baseline.

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

MP residual evidence показывает short electrical memory, особенно для APA
(residual ACF lag 1 около 0.43), и устойчивую корреляцию с GMP cross-memory
terms. High-amplitude bins не являются главным источником ошибки. Длительности
captures недостаточно для вывода о thermal memory, поэтому state-conditioned
ветвь пока заблокирована.

## Ближайшая точная последовательность

Каждый пункт ниже — отдельный small-task commit с тестом/проверкой и push:

1. Реализовать train/validation-only A0/A1 sensitivity runner поверх frozen
   fractional transform; не менять GMP model или общий PA evaluator.
2. Зафиксировать DPA и APA sensitivity configs, delay source, coefficients,
   protocol hashes, matched crop и fixed PA recipe до запуска.
3. Запустить A0/A1 сначала для DPA, затем отдельной задачей для APA; сохранить
   numerical artifacts без test access.
4. Отдельным decision commit выбрать A0 или A1 для каждого PA по pre-registered
   gate и обновить/создать versioned formal GMP configs.
5. Запустить formal GMP validation selection отдельно для DPA и APA. До этого
   момента `experiments/results/pa_gmp_*_selection` не считается результатом.
6. Для validation-selected GMP выполнить train OOF/reset-boundary и residual
   analysis; проверить, что gain не является только boundary/crop эффектом.
7. Если GMP acceptance выполнен, freeze model/manifest и отдельной командой
   открыть sealed test ровно один раз для каждого PA.
8. Только после numerical artifacts обновить `PA_MODEL_BENCHMARK.md` и status
   roadmap; затем повторно оценить Gate A→B.

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

Текущий projected margin — не завершённый cascade experiment:

| Dataset | New MP PA test fidelity | Existing spline-DPD residual | Projection margin |
|---|---:|---:|---:|
| DPA_200MHz | −35.099 dB | −29.864 dB | 5.235 dB |
| APA_200MHz | −36.990 dB | −32.741 dB | 4.250 dB |

Existing spline DPD ещё не прогонялся как cascade через новые frozen MP
models; таблица служит только gate arithmetic и не является DPD result.

## Этап B1. Frozen-evaluator DPD benchmark

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
- PAPR и peak predistorted amplitude;
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
- Отдельно оценить coefficient drift, maximum update rate и stability.

Артефакт: `ROBUSTNESS_AND_ADAPTATION.md`.

## Этап D. Hardware reference

- Bit-accurate simulator, а не numerical “fixed-point-like” approximation.
- Formats: signed 16, 14 и 12 bit coefficients/activations.
- Explicit accumulator/state widths, scaling, rounding and saturation.
- Separate counts: real MUL, ADD, sqrt/nonlinear, comparisons, LUT, coefficient
  and state memory.
- Full-record versus arbitrary streaming chunks must be equivalent for causal
  models.
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
