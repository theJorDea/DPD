# Robustness and adaptation protocol

Дата среза: 2026-07-29.

## 1. Статус

Полноценный robustness/adaptation benchmark **ещё не выполнен**. Текущие
formal PA results относятся отдельно к `DPA_200MHz` и `APA_200MHz`; они не
доказывают transfer между PA, waveform, bandwidth или operating point.

Этот документ фиксирует protocol до запуска transfer experiments. Он запрещает
объединять разные physical captures как случайные строки одного train/test
split и задаёт learning curves для быстрой recalibration.

## 2. Доступные captures и границы metadata

| Dataset | Declared device/capture | Waveform | Fs / BW | Split complex samples | Что неизвестно |
|---|---|---|---|---:|---|
| `DPA_200MHz` | DPA device | 10×20 MHz LTE, 64-QAM | 800 / 200 MHz | 23,040 / 7,680 / 7,680 | exact DUT identity, carrier frequency, output power/backoff, temperature, capture time |
| `DPA_160MHz` | DPA device | 4×40 MHz, 1024-QAM | 640 / 160 MHz | 294,912 / 98,304 / 98,304 | является ли DUT/operating point тем же, что DPA_200MHz |
| `APA_200MHz` | APA device | 5×20 MHz LTE TM3.1a, 256-QAM | 983.04 / 200 MHz | 58,980 / 19,662 / 19,662 | exact capture conditions beyond repository description |
| `APA_200MHz_b` | APA device, “measurement B” | тот же declared waveform family | 983.04 / 200 MHz | 58,980 / 19,662 / 19,662 | что именно менялось: time, power, bias, temperature, hardware или calibration |

Sources: `vendor/OpenDPD/datasets/*/spec.json` и
`vendor/OpenDPD/datasets/README.md`.

Надписи “DPA device”, “APA device” и “measurement B” недостаточны, чтобы
утверждать same physical PA/same operating point. До получения metadata каждая
пара трактуется как **capture transfer**, а не как контролируемый power-drift
experiment.

## 3. Запрещённые смешивания

Нельзя:

- конкатенировать DPA и APA samples и делать random row split;
- выдавать `DPA_160MHz -> DPA_200MHz` за чистый bandwidth test: одновременно
  меняются waveform, modulation, sample rate и, возможно, operating point;
- выдавать `APA_200MHz -> APA_200MHz_b` за power adaptation без metadata;
- подбирать architecture или calibration sample count по target test;
- нормировать каждый target split независимо так, чтобы скрыть gain/power
  shift;
- использовать target measured output для deployment DPD input;
- выбирать checkpoint по target test metric.

Каждый transfer report должен перечислять все одновременно изменившиеся axes.

## 4. PA-model robustness matrix

### 4.1 Within-capture control

Для каждого capture сначала воспроизводится ordinary protocol:

```text
source train -> fit PA model
source validation -> architecture/regularization selection
source test -> one frozen final report
```

Alignment, complex gain diagnostics, framing, common boundary mask и
normalization frozen по source train. Это контроль, а не transfer evidence.

### 4.2 Zero-shot capture transfer

После source freeze:

```text
source-trained frozen model
    -> target capture input
    -> y_hat_target
    -> compare with measured y_target
```

Минимальные направления, каждое отдельным result:

- `DPA_200MHz -> DPA_160MHz`;
- `DPA_160MHz -> DPA_200MHz`;
- `APA_200MHz -> APA_200MHz_b`;
- `APA_200MHz_b -> APA_200MHz`.

Разные sample rates требуют explicit resampling/model-timebase decision.
Нельзя молча переиспользовать sample-delay taps: один sample соответствует
разному physical time. Должны публиковаться две трактовки:

1. sample-index topology unchanged;
2. memory horizon mapped in seconds с заранее заданным resampling/delay rule.

Cross-family `DPA <-> APA` допускается только как stress test и всегда
маркируется different-PA transfer.

### 4.3 Limited-sample recalibration

Target capture заранее делится по complete chronological frames:

```text
calibration prefix -> adaptation only
target validation  -> choose N/regularization/update stopping
target test        -> one final report
```

Learning curve:

\[
N\in\{64,128,256,512,1024,2048,4096,8192,\ldots\},
\]

но (N) округляется/маскируется так, чтобы causal history и frame boundaries
были определены. Для very short (N) отдельно указывается effective scored
sample count после warm-up.

Сравниваются adaptation modes:

- no update;
- readout/coefficient-only ridge refit;
- recursive least squares/block-RLS, если реализован;
- limited iteration warm-start optimization;
- full target-train refit как upper reference, не low-sample candidate.

Architecture frozen на source; target validation может выбрать только заранее
preregistered update hyperparameters. Target test не участвует.

## 5. DPD robustness matrix

DPD transfer запускается только после Gate A→B PASS и использует правильный
path:

```text
desired target x -> source/calibrated DPD -> frozen target PA evaluator
                 -> compare with frozen target g*x
```

Для каждой модели:

- source operating point performance;
- zero-shot target performance;
- target performance after (N) calibration samples;
- peak-drive/support violations до и после adaptation;
- coefficient drift and update stability;
- catastrophic degradation/non-finite/saturation count.

Если target evaluator fitted на тех же target samples, что DPD adaptation,
нужен второй independently fitted evaluator либо physical target output, иначе
optimizer/evaluator coupling остаётся.

## 6. Metrics and curves

### 6.1 Quality

- pooled and OpenDPD-compatible complex NMSE;
- ACLR/ACPR left, right, average;
- sample-domain and demodulated EVM, если definitions доступны;
- output/error PSD;
- AM/AM and AM/PM residuals;
- PAPR, average and maximum predistorted drive;
- input-support violation fraction;
- full-record/common-interior gap.

### 6.2 Adaptation efficiency

- calibration samples (N);
- fit/update wall-clock and CPU/device;
- peak memory, если измерена;
- coefficient bytes transmitted/stored;
- updates per second and maximum stable update rate;
- NMSE/ACLR versus (N) with validation-frozen stopping;
- time to recover 50%, 90% and 99% of full-refit gain.

### 6.3 Statistical reporting

Closed-form deterministic fits report exact data order and solver. Stochastic
methods use at least seeds `{0,1,2}` and report mean, standard deviation and
individual runs. Capture variability cannot be replaced random neural seeds:
нужны independent captures для confidence interval по hardware drift.

## 7. Slow-state evidence gate

Current residual manifests declare `independent_capture_count=0`; short
records с reset boundaries не различают thermal/bias/trapping state и software
frame transient. Поэтому state-conditioned spline запрещён до появления:

- multiple chronological captures с recorded elapsed time;
- junction/baseplate temperature или хотя бы controlled warm/cold labels;
- bias/current/output-power metadata;
- sufficiently long envelope evolution relative to proposed \(\beta_l\);
- reproducible residual correlation with slow envelope state on a held-out
  capture.

Только после этого разрешён ablation

\[
q_l[n]=\beta_lq_l[n-1]+(1-\beta_l)|x[n]|^2.
\]

## 8. Required machine-readable manifest

Каждый robustness run должен сохранять:

```text
source_capture + hashes
target_capture + hashes
declared changed axes
source/target split frame indices
alignment/gain/resampling policy
model/checkpoint/config hashes
calibration N and exact indices
solver/update rule/seed
metrics before/after adaptation
wall time and hardware
test access count
evidence scope
```

Canonical outputs должны быть immutable; rerun получает новый versioned output
directory, а не перезаписывает предыдущий result.

## 9. Следующий эксперимент максимальной ценности

До DPD наиболее информативен PA-model zero-shot + limited-calibration test
`APA_200MHz -> APA_200MHz_b`, потому что repository объявляет одинаковую
waveform/spec и явно называет второй capture “measurement B”. Но перед claim
нужно выяснить, что означает B. Без ответа результат будет честно называться
capture transfer, а не power/thermal adaptation.

Для DPA изменение 160↔200 MHz смешивает больше факторов; его следует выполнять
вторым как stress test с explicit resampling/time-memory policy.
