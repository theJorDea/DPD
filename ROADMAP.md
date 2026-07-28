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
- common-warmup score;
- отдельный integrity-gated test command, который проверяет hashes до первого
  чтения test;
- unit tests на sealed test, frame reset и отсутствие post-hoc gain/delay fit.

Основные файлы: `baseline/pa_benchmark.py`,
`experiments/evaluate_frozen_pa.py`, `tests/test_pa_benchmark.py`,
`tests/test_evaluate_frozen_pa.py`.

## Этап A2. PA baselines

Статус: MP завершён; GMP в работе; локального OpenDPD checkpoint нет.

Порядок:

1. [x] Memory Polynomial с validation grid order/memory/ridge.
2. [ ] GMP с column normalization, SVD/rank control.
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

GMP kernel уже имеет causal leading policy, factorized streaming inference,
exact operation/state count и column-scaled ridge/truncated-SVD fit. Следующая
задача — validation-only topology selection и только затем frozen test.

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
