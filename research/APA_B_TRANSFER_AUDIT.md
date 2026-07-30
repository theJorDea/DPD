# APA_200MHz → APA_200MHz_b transfer audit

Дата preregistration: 2026-07-30.

Этот документ фиксирует следующий эксперимент до запуска любого fit на
`APA_200MHz_b`. Его задача — проверить перенос уже замороженных forward PA
models между двумя capture одного declared APA waveform family. Это не
утверждение о power/temperature adaptation: repository не сообщает, что
именно означает `measurement B`.

## 1. Что проверено до preregistration

Проверены только metadata, `train` и `val` пары. Held-out target split
остаётся запечатанным: его содержимое и hash не входят в этот protocol и не
используются для выбора модели, delay, gain или числа calibration samples.

| Поле | `APA_200MHz` | `APA_200MHz_b` |
|---|---|---|
| spec description | APA device | APA device, measurement B |
| input sample rate | 983.04e6 Hz | 983.04e6 Hz |
| declared bandwidth | 200e6 Hz | 200e6 Hz |
| `nperseg` | 19662 | 19662 |
| waveform | 5-carrier LTE TM3.1a, 256QAM | та же declared family |
| train/val lengths | 58980 / 19662 | 58980 / 19662 |

`spec.json` paths:

- `vendor/OpenDPD/datasets/APA_200MHz/spec.json`
- `vendor/OpenDPD/datasets/APA_200MHz_b/spec.json`

The three source/target input files are byte-identical:

```text
train_input.csv  2ca703c9e7eb39839db1fb01f91081a86e18535e53751c587cc71d2d71e9c625
val_input.csv    39bb15c9bd92549d1653498c140caff5cb2f20edffd433eafa46b4a81c491981
```

The measured output files are different:

```text
source train_output.csv  e760adf3908ed1be1e610c46f056e88bad6107a81cc8b01d91306727316b5930
target train_output.csv  22a4be8a899ab04ae5b845d0a6839a569c76ee0c8499d722b15aea3b486c13fc
source val_output.csv    02f67574444c7a8ba321cde1ea919c07fef1c99fb9d25678befc019c6b6645e2
target val_output.csv    eada577549804d366285a7df46bd28c6153e34f512e285cd8477b77682056615
```

Следствие: это контролируемый **same-excitation capture-transfer** тест, но не
waveform-generalization и не доказанный operating-point/thermal drift.
Unknown axes (время, мощность, bias, температура, feedback calibration,
physical DUT identity) должны быть явно указаны в итоговом отчёте как
неизвестные.

## 2. Замороженные source models

До target fit разрешены только следующие уже опубликованные source models:

| Model | Source artifact | Config / manifest | Role |
|---|---|---|---|
| causal GMP | `experiments/results/pa_gmp_apa200_selection/selected_gmp_pa.npz` | `experiments/configs/pa_gmp_apa200.json`, `experiments/results/pa_gmp_apa200_selection/selection_manifest.json` | accurate high-cost control |
| lag-9 sparse spline-memory | `experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/selected_sparse_pa.npz` | `experiments/configs/pa_sparse_spline_memory_lag9_apa200.json`, `experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/selection_manifest.json` | low-cost candidate |

Frozen hashes:

```text
GMP model       cac7982dc2ff0df0ede56817b207807b87c3bf32fd9115bd2b4098eae3941add
GMP config      0e8dc916eab1fa4d8588fca48e18b44f3aa1d167600b0c786476458f46aa8293
GMP manifest    ef48c9afdfc24ab6066e51f14e3b5abe6b306aa4332313bbd8e5fc83620018dd
sparse model    2e5eb3405c8633d5d766e1b78b84c633d0ede77c4c97c1a415fd27410f1b7475
sparse config   93807ab6f3906b3898d805430e3ffe419b85a90ee72c28bfb7c1b2f486f32770
sparse manifest b6188ba8e5cb29e99b1db745b6a9b7280631361fc6c99f39df0af1e7e6ad36ff
```

Architecture, knots, branch topology, solver family and source coefficients
are immutable during transfer. Target calibration may update coefficients
only; it may not select branches, knots, polynomial order, memory length or
the evaluator.

## 3. Frozen scoring protocol

1. Load and verify config, source artifacts, source dataset train/val hashes,
   target dataset spec/train/val hashes before loading waveform arrays.
2. Use the source frozen integer alignment (`delay=0`, no fractional
   transform) as the primary strict protocol. No post-prediction gain fit is
   allowed.
3. Evaluate each source model on source validation as a reproduction control,
   then on target validation zero-shot.
4. Treat each `nperseg` frame as an independent zero-state record. For the
   target train prefixes, preserve frame boundaries; never concatenate a prefix
   across frames before constructing causal features.
5. Report full-record pooled NMSE, common-interior NMSE, OpenDPD-compatible
   NMSE where a complete frame exists, RMS EVM, per-frame values, finite/
   support violations, and exact operation/memory metadata.
6. A separately labelled nuisance diagnostic may estimate one integer delay
   (`max_abs_delay=32`) and one complex LS gain from target **train only**.
   The estimate is frozen before target validation and is never refit per
   frame/split. It does not replace the strict result and cannot be used to
   claim physical de-embedding. Fractional delay remains disabled.

The primary zero-shot transfer score is therefore:

```text
target input x -> frozen source model -> y_hat
compare y_hat with target measured y
```

## 4. Limited-sample coefficient adaptation

The following calibration prefixes are fixed before seeing target validation:

```text
N = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384} samples per
complete target-train frame
```

For each `N`, use samples `[0:N)` in every complete train frame, with a reset
at every frame boundary. The three prefixes are the only target samples
allowed for that fit. `N` values that leave an underdetermined or rank-deficient
design are recorded as **infeasible**, not silently removed (GMP has 444
complex coefficients; lag-9 sparse has 108).

Two adaptation modes are preregistered:

- `none`: source coefficients unchanged;
- `coefficient_only_ridge`: fixed source topology/knots and the source solver
  family, refit complex coefficients on the selected prefixes. GMP uses
  `ridge=1e-7`, `ridge_lstsq`; sparse uses `ridge=1e-8` and its fixed complex
  augmented ridge solve.

No target validation output participates in fitting. Validation is used only
after all prefix fits exist, to select a deployment sample count by this
frozen rule:

> choose the smallest feasible `N` whose validation full-record NMSE is within
> 0.25 dB of the best feasible `N`, subject to finite output and zero support
> violations; break ties by common-interior NMSE, then wall-clock time.

Target held-out evaluation is a separate post-freeze action. It is not part of
the first transfer run and cannot choose `N`, a nuisance mode, or a model.

## 5. Evidence boundaries

The transfer result can establish:

- whether a source PA model transfers to a second measured capture under the
  same declared excitation;
- how much quality is recovered by coefficient-only calibration versus sample
  count and wall-clock time;
- whether the 72-MUL lag-9 candidate retains a useful cost/quality advantage
  over the 954-MUL GMP control.

It cannot establish without metadata or physical remeasurement:

- thermal/bias state modelling;
- power-level adaptation;
- generalization to a new waveform or bandwidth;
- DPD linearization;
- superiority over OpenDPD on a physical PA;
- compliance with an unstated Huawei acceptance threshold.

