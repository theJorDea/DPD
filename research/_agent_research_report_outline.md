# RESEARCH_REPORT — интеграционный outline

Статус: подготовлено для объединения в итоговый отчёт, 2026-07-28. Это не
отчёт о завершённых RF-экспериментах. Результаты, полученные только статическим
аудитом, smoke-тестом или опубликованной статьёй, явно отделены от ещё не
запущенных измерений.

Связанные документы:

- [`research/opendpd_audit.md`](opendpd_audit.md) — аудит OpenDPD `main`;
- [`research/egor_pipeline_audit.md`](egor_pipeline_audit.md) — аудит
  `DPD_for_PA`/`chaotic_library`;
- [`research/literature_review.md`](literature_review.md) — первичные статьи и
  несопоставимые experimental groups;
- [`research/proposed_methods.md`](proposed_methods.md) — архитектурный
  shortlist;
- [`research/comparison_table.csv`](comparison_table.csv) — текущая comparison
  matrix.

## 1. Физическая задача и правильное направление проверки

Для комплексного desired baseband \(x[n]=I[n]+jQ[n]\) требуется:

\[
x[n]\xrightarrow{D}z[n]\xrightarrow{P}y[n]\simeq g\,x[n],
\]

где \(P\) — физический PA, \(D\) — predistorter, а \(g\) — заранее
зафиксированный complex или real target gain. Основной deployment score
оценивает:

\[
\hat y[n]=P(D(x_\mathrm{test}[n])),\qquad
\hat y[n]\approx g x_\mathrm{test}[n].
\]

ILA имеет другой, допустимый только для calibration, mapping:

\[
u[n]=y[n]/g,\qquad D_\mathrm{post}(u[n])\approx x[n].
\]

Диагностический круг
\(y_\mathrm{test}/g\to D_\mathrm{post}\to\hat P\to y_\mathrm{test}\)
проверяет согласованность inverse и forward моделей на уже известных PA
outputs. Он не доказывает линеаризацию нового \(x_\mathrm{test}\). Формально
нужен отдельный `x_desired -> D -> PA` path; `y_test` не может формировать его
input.

### 1.1 Точная трактовка требования \(10^{-5}\)

До tuning зафиксировать основной gate:

\[
r_\mathrm{NMSE}=
\frac{\sum_n|\hat y[n]-g x[n]|^2}
     {\sum_n|g x[n]|^2}\le 10^{-5},
\qquad
\mathrm{NMSE}_{dB}=10\log_{10}r_\mathrm{NMSE}\le-50\ \mathrm{dB}.
\]

Одновременно публиковать:

1. обычный complex MSE \(N^{-1}\sum|e[n]|^2\) (масштабно-зависим);
2. pooled normalized error power и его dB;
3. OpenDPD-compatible segment-wise mean-of-dB NMSE (для обратной
   совместимости, но не как единственный gate);
4. RMS EVM \(=\sqrt{\sum|e|^2/\sum|g x|^2}\) и EVM dB \(=20\log_{10}\) RMS
   EVM;
5. EVM после demodulation/equalization, если доступен, отдельно от spectral
   EVM OpenDPD;
6. ACLR/ACPR left, right и arithmetic average с frozen PSD configuration.

`loss < 1e-5` в PyTorch OpenDPD — это среднее по двум real channels и не равно
normalized complex error без деления на target power. OpenDPD также считает
segment-wise dB mean, а не pooled ratio; эти величины нельзя молча смешивать.

## 2. Apples-to-apples gate

Новый метод можно назвать лучше OpenDPD только при одновременном выполнении:

- один dataset/capture и один PA checkpoint (или одна физическая PA session);
- зафиксированные commit, CSV SHA-256, split manifest и normalization;
- одинаковые `g`, integer/fractional alignment, feedback correction и
  warm-up/boundary policy;
- одинаковые FFT/Welch: `fs`, `nperseg`, window, overlap, detrend, scaling,
  occupied/adjacent masks;
- tuning/checkpoint/feature/knot/threshold selection только на train/validation;
  test открывается один раз после freeze;
- один deployment direction `x_desired -> D -> PA`; circular inverse score
  публикуется только как diagnostic;
- stateful модели сравниваются в continuous streaming и reset режимах с
  одинаковым контекстом; noncausal look-ahead входит в latency;
- минимум три seeds для stochastic methods; confidence intervals по
  независимым segments;
- отдельно считаются real MUL, real ADD, nonlinear ops, comparisons, LUT,
  reads/writes, stored coefficients и peak memory;
- fixed-point — bit-true reference с форматом, rounding, accumulator width,
  saturation и quantization degradation;
- physical score не заменяется surrogate score. Surrogate-only и
  post-layout/energy-model results имеют отдельные labels.

Сравнительные strata:

| Stratum | Что допустимо сравнивать | Что нельзя объединять |
|---|---|---|
| `APA_200MHz` | OpenDPDv2 TRes/GRU/MP/GMP и новый метод при одном checkpoint | `DPA_200MHz` TCN, 20 MHz SparseDPD, spline E1/E2/E3 |
| `DPA_200MHz` | TCN-DPD и re-run baselines на frozen DGRU surrogate | physical APA OpenDPDv2 |
| `SparseDPD_20MHz` | PNTDNN и её hardware table только внутри paper | physical closed-loop claim |
| `Spline_E1/E2/E3` | SPH/SMP/MP внутри каждого собственного PA set | NMSE/ACLR ranking OpenDPD |
| `FR3_DUT1/2` | feature-selected models внутри capture/DUT | surrogate NMSE как physical OpenDPD result |

## 3. Что подтверждено тремя направлениями аудита

### 3.1 OpenDPD `main` — подтверждено кодом/артефактами

Источники: [`opendpd_audit.md`](opendpd_audit.md), commit
`7426bbf8a47624b59bd7f045a86641b403023f3c`,
[репозиторий OpenDPD](https://github.com/lab-emi/OpenDPD).

- Neural DLA действительно получает `x_desired`, пропускает его через frozen PA
  surrogate и оптимизирует target `g*x`; OpenDPD neural path не является
  Egor-style circular test (`project.py:201-215`, `models.py:172-185`,
  `steps/run_dpd.py:26-28,79-99`).
- MP/GMP benchmark обучает ILA postdistorter, но на val/test переносит
  коэффициенты и подаёт `X_val/X_test` в predistorter
  (`benchmark/benchmark_volterra.py:1-19,899-940`).
- Current repository benchmark DPD metrics получены через learned PA surrogate;
  physical result OpenDPDv2 находится в статье и не воспроизводится из
  checkout без captures/checkpoints.
- Built-in splits фактически 60/20/20; README также содержит старую/неясную
  формулировку 8:2:2. `APA_200MHz` и `APA_200MHz_b` имеют идентичные input
  waveforms, поэтому A→B не является waveform generalization.
- В loader нет alignment, fractional-delay, complex-gain, DC/IQ или feedback
  frequency-response correction; утверждение о предварительном time alignment
  есть только в документации.
- TRes-GRU/DeltaGRU имеют ±16 sample residual context и `roll(-1)` future
  sample/wrap; TCN также noncausal. Current offline reset-per-segment semantics
  не доказывают streaming equivalence.
- Temporal sparsity зануляет weights/deltas, но included eager и Triton paths
  всё равно выполняют dense matrix-vector products; `HW_PARAM` — proxy, не
  runtime count.
- 999 stored parameters не означают `<1000` real multiplications/sample:
  TRes-DeltaGRU-H15 оценивается примерно в 1058 MUL после gates/feature path.
  Аналитический GRU-H16 может быть около 944 MUL по принятой convention, но
  streaming/latency/fixed-point ещё не измерены.
- Quantization — частичный fake quant; floating Conv/Hardswish/path и silent
  fallback в исходный float model возможны.
- OpenDPD NMSE/EVM/ACLR evaluator нестандартен: segment-wise dB NMSE,
  spectral-bin EVM и adjacent subchannel относительно strongest in-band
  subchannel. Репортировать его только как compatibility metric.
- В git отсутствуют referenced checkpoints, raw logs, polynomial JSON/source
  snapshots, inference timing и calibration wall-clock.

Опубликованный physical APA reference из
[OpenDPDv2](https://arxiv.org/abs/2507.06849): TRes-DeltaGRU-999 — NMSE
−39.6 dB, EVM −42.1 dB, average ACPR −59.9 dBc на 3.5 GHz GaN Doherty,
200 MHz TM3.1a; это published measurement, не локально повторённый run.

### 3.2 Egor `DPD_for_PA` + `chaotic_library` — подтверждено кодом/данными

Источники: [`egor_pipeline_audit.md`](egor_pipeline_audit.md),
[DPD_for_PA](https://github.com/EgorMa1tsev/DPD_for_PA),
[chaotic_library](https://github.com/CapitalistGeorge/chaotic_library).

- CSV Егора побайтно совпадают с OpenDPD `DPA_200MHz` train/test; val split
  отсутствует в Egor checkout.
- Notebook cell 10 и cell 14 выполняют круговой
  `y_test/g -> inverse -> PA surrogate -> y_test`; cell 11 использует
  правильный `x_test -> DPD -> PA surrogate -> g*x`, но не считает NMSE/EVM/ACLR
  и не проверяет физический PA.
- Notebook train fit — ILA `y/g -> x`, что само по себе допустимо; ошибочен
  статус circular reconstruction как доказательства deployment linearization.
- `R=800` PA и `R=600` DPD; `W` — dense NumPy array, даже при `sparsity=0.1`.
  Развёрнутый DPD: около 728 622 real MUL и 726 152 ADD/sample; идеальный CSR
  lower bound всё ещё около 80 410 MUL/sample. Ограничение 1000 нарушено.
- `predict()` создаёт zero reservoir state; `last_state_` не используется
  ordinary prediction path. Chunked/streaming output не эквивалентен full-array.
- Независимые I/Q reservoirs/readouts не имеют phase-equivariant tying; численная
  величина нарушения не измерялась.
- Fourier features — pointwise `sin/cos` от standardized I/Q, а не FFT или
  временная frequency representation; Cartesian inductive bias не доказан как
  полезный для PA.
- Notebook использует `fs=200`, `nperseg=256`, хотя DPA_200MHz spec — 800 MS/s,
  `nperseg=2560`; PSD plots поэтому невалидны как dataset ACPR.
- Cached PA/circular values примерно −31.67/−32.09 dB complex NMSE, не
  normalized \(10^{-5}\)/−50 dB; correct-direction scalar score отсутствует.
- README/`DPD_3.pdf` claims около 100×/10× faster training и меньшую память
  противоречат друг другу и не сопровождаются timing logs, baseline config или
  checkpoints. Они пока неподтверждены.

### 3.3 Литературный аудит — подтверждено первичными источниками

Полная matrix находится в [`literature_review.md`](literature_review.md).
Ключевые anchors:

- [OpenDPDv2](https://arxiv.org/abs/2507.06849): physical APA_200MHz,
  TRes-DeltaGRU, temporal sparsity/quantization;
- [TCN-DPD](https://arxiv.org/abs/2506.12165): DPA_200MHz frozen-surrogate,
  noncausal depthwise TCN, no physical verification;
- [SparseDPD](https://arxiv.org/abs/2506.16591) и
  [code](https://github.com/MannoVersluis/SparseDPD): 20 MHz PNTDNN,
  64 parameters/72 ops, FPGA post-implementation simulation;
- [spline-interpolated LUT](https://arxiv.org/abs/1907.02350): physical E1/E2/E3,
  SPH/SMP, 36–75 reported real MUL/sample in selected configurations;
- [piecewise closed-loop](https://arxiv.org/abs/2003.06348): 28 GHz OTA array,
  pruning/orthogonalization trade-off;
- [feature selection](https://arxiv.org/abs/2607.15441): v1 submitted
  2026-07-16, 321 200-feature offline dictionary, physical DUT1 table;
- [MP](https://doi.org/10.1109/TCOMM.2003.822188),
  [GMP](https://doi.org/10.1109/TSP.2006.879264),
  [ILA](https://doi.org/10.1109/78.552219),
  [DLA](https://doi.org/10.1109/TSP.2006.882058),
  [DOMP/LASSO](https://doi.org/10.3390/s21175772),
  [PN-RNN](https://doi.org/10.1109/LMWT.2024.3393859),
  [DeltaDPD](https://doi.org/10.1109/LMWT.2025.3565004).

Литература подтверждает, что complex local-support spline — разумный
low-cost baseline, а SPH/SMP/selected memory branches — минимальные следующие
расширения. Ни одна внешняя работа из проверенного корпуса не доказывает
apples-to-apples превосходство над APA_200MHz OpenDPDv2.

## 4. Что пока не подтверждено

Нельзя утверждать до запуска отдельного протокола:

- корректный-direction NMSE/EVM/ACPR Егора;
- физическую PA linearization любого нового spline/MP кандидата;
- превосходство над OpenDPDv2 по quality, calibration time или robustness;
- реализацию temporal sparsity в runtime;
- continuous-state quality/latency OpenDPD;
- real-time throughput на CPU/FPGA/ASIC;
- bit-true W16/W12/12-bit degradation;
- operation counts, использующие paper FLOPs без перевода в выбранную
  real-MUL convention;
- generalization к новой waveform, power, temperature, carrier или PA drift;
- claims о 10×/100× speed-up reservoir pipeline.

Текущий локальный smoke status: `python -m unittest discover -s tests -v`
прошёл **26 тестов за 0.030 s**; это проверка alignment/spline/complex
regression/complexity/fixed-point primitives, не RF benchmark. CLI
`python -m baseline.train_spline --help` доступен. Один train+validation fit
на DPA_200MHz занял 0.12 s, на APA_200MHz — 0.54 s на текущем CPU; эти числа
являются только planning smoke timings, без test/physical claims.

## 5. Кандидаты для Pareto frontier

| Кандидат | Формат/ожидаемая роль | Nominal real MUL/sample | Calibration | Fixed-point fit | Статус |
|---|---|---:|---|---|---|
| No DPD | нижняя quality reference | 0 | none | trivial | protocol control |
| MP/GMP ILA | classical quality baseline | complex coeff × 4 + basis | LS/ILA | MAC-friendly | OpenDPD benchmark, exact count pending |
| GRU-H16 | neural cost baseline | ~944 по audit convention | gradient, 300 epochs | not yet bit-true | OpenDPD candidate |
| TRes-GRU/DeltaGRU-H15 | OpenDPDv2 quality reference | ~1058+ / generic paper FLOPs 1324 | gradient, 240–300 epochs | partial quant only | physical paper reference |
| Egor EnhancedESN-FAN | reservoir hypothesis | ~728 622 dense DPD | state rollout + eigensolve/ridge | poor without sparse engine | gate failure confirmed |
| Complex memoryless linear spline | \(z=xC(|x|)\), \(K=8..64\) | 9 (4M2A convention) | one complex ridge solve | excellent LUT candidate | implementation smoke-tested |
| Spline branches / SPH | local spline + selected delays/FIR | ~9B or \(9+4L\) | group ridge/alternating | excellent | next experiment |
| Sparse CPWL/GMP dictionary | selected memory terms | term-dependent | OMP/group-LASSO | good if fixed topology | after memoryless |
| State-conditioned spline | slow thermal/bias state | low, state-dependent | fixed \(\beta\)+ridge | good | only with drift evidence |
| Spline + tiny residual TCN | residual quality rescue | measure end-to-end | stochastic E2E | medium | last resort |

Theoretical spline count includes radius, interval index/coordinate, interpolation
and complex multiply; comparisons, LUT reads, square root/division and memory
traffic are reported separately. It is not a measured latency.

## 6. Experiment stages

### Stage 0 — provenance and evaluator

Freeze:

```text
OpenDPD commit: 7426bbf8a47624b59bd7f045a86641b403023f3c
Dataset: vendor/OpenDPD/datasets/APA_200MHz (and DPA_200MHz for transfer study)
Split: train/val/test files, SHA-256 manifest
Gain: compare complex-LS and OpenDPD peak protocols, select one before tuning
Metrics: pooled NMSE + OpenDPD-compatible NMSE + RMS/demod EVM + ACPR L/R/avg
```

First commands (safe smoke/provenance):

```bash
python -m unittest discover -s tests -v
python -m baseline.train_spline --help
sha256sum vendor/OpenDPD/datasets/APA_200MHz/*.csv \
          vendor/OpenDPD/datasets/APA_200MHz/spec.json
git -C vendor/OpenDPD show -s --format='%H%n%aI%n%s' HEAD
```

Before model comparison, add the missing `test_dpd_direction.py` and an explicit
test-only evaluator. `baseline/train_spline.py` deliberately never loads test;
there is currently no checked-in `baseline/evaluate_spline.py`, so a final test
command must not be invented until that entry point is implemented.

### Stage 1 — reproduce controls

OpenDPD environment (planned; not run on current CPU-only host):

```bash
cd vendor/OpenDPD
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -v -m "not extended"
bash benchmark/reproduce_benchmark_report.sh --dry-run \
  --output-dir /tmp/opendpd-benchmark-dry-run
```

For a GPU reproduction, pin the recorded benchmark environment (Python 3.13.14,
PyTorch 2.13.0+cu132, CUDA 13.2, RTX PRO 6000 Blackwell) or record any
deviation. The repository's published matrix took 16 369 s (about 4 h 33 min)
for one recorded run; three neural seeds require a fresh measured estimate,
not an assumed linear extrapolation.

Reference PA/DPD commands after evaluator freeze:

```bash
cd vendor/OpenDPD
python main.py --dataset_name APA_200MHz --step train_pa \
  --PA_backbone tres_gru --PA_hidden_size 27 --frame_length 200 \
  --frame_stride 1 --n_epochs 300 --seed 0 --re_level hard \
  --eval_val 1 --eval_test 0 --accelerator cuda
# Then select the PA checkpoint by validation NMSE and run a separate
# x_desired -> DPD -> frozen PA evaluator; open test only once.
```

Do not treat bundled benchmark JSON as a local completed reproduction: its
checkpoints/raw logs are absent from checkout.

### Stage 2 — spline validation sweep

First smoke command (correctly directed PA surrogate only, validation selection):

```bash
python -m baseline.train_spline \
  --dataset vendor/OpenDPD/datasets/DPA_200MHz \
  --output-dir experiments/results/spline_dpa200_validation \
  --knot-counts 8,12,16,24,32,48,64 \
  --knot-strategies uniform_amplitude,uniform_power,quantile,compression_aware \
  --ridges 0,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2 \
  --smoothnesses 0 \
  --gain-strategy complex_ls \
  --fit-pa-surrogate --pa-orders 1,3,5,7 --pa-delays 0,1,2 \
  --selection-metric surrogate_cascade_nmse
```

This is 280 configurations before smoothness ablations. Add a second
validation-only sweep for nonzero second-difference penalties; never select
knots/ridge/smoothness on test. The fitted memory-polynomial surrogate is a
planning surrogate, not the OpenDPD frozen checkpoint and not physical PA
evidence.

For the primary low-cost baseline use the same command with
`--selection-metric inverse_nmse` as a calibration diagnostic, then evaluate
the selected model through the common evaluator in the correct direction.
Record `training_report.json`, model NPZ, config, software versions, hashes,
fit time, validation metrics, extrapolation fraction, PAPR and maximum
predistorted amplitude.

### Stage 3 — memory ablations

Only if memoryless residual/lags justify it:

```text
branch set: {(m,d)} = {(0,0)}, {(0,0),(1,0)}, {(0,0),(1,0),(2,0)}
selection: greedy forward, group OMP/group LASSO, validation NMSE + ACLR delta
models: sparse spline memory, SPH short FIR, radial CPWL
```

Each branch must add a row with incremental real MUL/ADD, coefficient storage,
validation gain, test result after freeze, peak drive and fixed-point loss.
No branch is retained solely because parameter count is larger/smaller.

### Stage 4 — fixed point and streaming

Run bit-true reference for FP32, FP16-like storage, signed 16-bit and signed
12-bit coefficients/input; sweep accumulator widths and saturation. Check:

```text
full-array == chunked-stream output (within declared tolerance)
reset and warm-up behavior
no future sample/wrap
coefficient interpolation/address quantization
PAPR, saturation count and NMSE/ACPR degradation
```

OpenDPD fake-quant path must be hard-failed if any module silently falls back to
float. Sparse counts are valid only for an implementation that actually skips
zeros.

### Stage 5 — physical PA

Only after surrogate gate passes: export identical coefficients/quantized
formats, drive the same physical PA session and use a separate measured
feedback capture. Report output power, gain/alignment, occupied/adjacent masks,
left/right ACPR, demod EVM, stability and thermal/bias operating point. A
surrogate-only win is labelled surrogate-only.

## 7. Resources, time and reproducibility

Current audit host: Python 3.14.6, NumPy 2.5.1, Intel i5-12450H (8 physical /
12 logical CPUs), 15 GiB RAM, no NVIDIA runtime. Local primitive tests took
0.030 s (26 tests). A single NumPy train+validation spline fit (one \(K=16\)
configuration, no test access) took 0.12 s on DPA_200MHz and 0.54 s on
APA_200MHz; these are smoke planning measurements, not quality benchmarks.

The largest dense APA design matrix at \(N=58\,980,K=64\) is about 60.4 MB in
complex128 before solver workspaces; the planned spline sweep fits comfortably
within 15 GiB if trials are processed sequentially. Exact full-sweep wall-clock
must be measured with a fixed BLAS thread count. Neural OpenDPD reproduction
requires CUDA GPU; current CPU-only machine is suitable for evaluator/tests and
spline calibration, not the full 300-epoch neural matrix.

Every result directory should contain:

```text
command.txt
config.json
environment.txt
dataset_manifest.sha256
pa_checkpoint_hash (or "not available")
training_report.json / metrics.json
raw predictions or explicit checksum
operation_count.json
fixed_point_config.json (if applicable)
```

## 8. First implementation changes

1. Add explicit `evaluate_spline.py`/common evaluator with three named paths:
   `PA_ONLY`, `CIRCULAR_DIAGNOSTIC`, `PREDISTORTION`.
2. Add `test_dpd_direction.py`, pooled-vs-segment NMSE tests, and no-test-access
   assertion.
3. Freeze alignment/gain/PSD protocol and expose left/right/average ACPR.
4. Add a streaming state API only where a model has state; prohibit TRes/TCN
   future wrap or declare look-ahead.
5. Add physical/surrogate provenance fields to every JSON result.
6. Add exact operation/latency/memory counters and bit-true quantization checks.
7. Then run memoryless spline sweep, followed by nested memory extensions.

No quality or superiority claim belongs in `BENCHMARK_REPORT.md` until these
changes and the frozen apples-to-apples gate are complete.
