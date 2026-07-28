# Roadmap: two-loop PA identification and DPD

Дата обновления: 2026-07-29.

Главное изменение относительно первого этапа: дальнейшее улучшение DPD
приостановлено, пока независимый PA evaluator не станет заметно точнее
оцениваемого DPD residual. Текущий memory-polynomial PA surrogate имеет
test fidelity около −30…−31 dB, а spline-memory DPD уже достигает до −32.7 dB
на том же evaluator. Это делает optimizer/evaluator coupling главным риском.

## Правила выполнения

- Каждая строка roadmap реализуется отдельным небольшим commit и сразу
  отправляется в `git@github.com:theJorDea/DPD.git`.
- Architecture/hyperparameter selection использует только train/validation.
- Test не используется для tuning и открывается после freeze.
- DPA_200MHz и APA_200MHz не объединяются: это разные PA/captures.
- Surrogate, cross-surrogate и physical-PA результаты маркируются раздельно.
- Parameter count никогда не подменяет operation, state-memory или latency.

## Этап 0. Requirements contract

Статус: завершён документально.

- [x] Отделить явные требования слайдов от интерпретации.
- [x] Зафиксировать фактический OpenDPD contract.
- [x] Составить список неизвестных критериев Huawei.
- [ ] Получить ответы владельца требований на вопросы из `REQUIREMENTS.md`.

Артефакт: `REQUIREMENTS.md`.

## Этап A1. Единый PA-model evaluator

Статус: следующий implementation task.

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

Первый commit этого этапа: evaluator contract, result schema и unit tests без
добавления новых моделей.

## Этап A2. PA baselines

Порядок:

1. Memory Polynomial с validation grid order/memory/ridge.
2. GMP с column normalization, SVD/rank control.
3. OpenDPD PA backbone/checkpoint, только если checkpoint доступен или
   воспроизводимо обучен.
4. Sparse complex spline-memory PA.
5. Spline/CPWL memoryless nonlinearity + short complex FIR.
6. State-conditioned variant только при residual evidence slow state.

Каждый baseline получает самостоятельный commit: model + tests, затем config +
result artifact. Нельзя менять evaluator одновременно с моделью.

## Этап A3. Residual analysis

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
