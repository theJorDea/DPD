# Hardware cost and fixed-point contract

Дата среза: 2026-07-30.

Уточнение научного руководителя: 1000-real-MUL-equivalent time budget
относится к модулю DPD, а не к behavioral PA model. Все операции DPD вместе
сравниваются по времени с одним reference kernel; поэтому таблицы PA ниже
являются audit стоимости evaluator, но не Huawei complexity gate. Пока target
и timing kernel не определены, финальный DPD pass/fail по latency невозможен.

## 1. Статус и область доказательства

Для selected MP, causal GMP, legacy spline DPD, APA SPH, первого
non-factorized sparse spline-memory PA и residual-guided lag-9 sparse PA имеются
analytical operation/state counts. Для causal GMP и SPH также доказана
equivalence NumPy full-record, reset-per-frame и arbitrary streaming chunks в
floating-point. Lag-9 sparse PA также прошёл exact streaming/reset checks и
является internal cheap-Pareto point, но не независимым evaluator.
На независимом `APA_200MHz_b` capture эти же frozen costs были проверены в
zero-shot и coefficient-only transfer режимах; transfer не меняет inference
topology, только коэффициенты.

Теперь выполнены:

- bit-accurate 16/14/12-bit PA evaluation для frozen causal GMP и lag-9 sparse
  PA на APA train/validation;
- explicit input/coefficient/power/output formats, 56-bit accumulators,
  nearest-even rounding, saturation counters и integer sqrt/interpolation;
- exact reset-per-frame и arbitrary-chunk streaming equivalence;
- train-only scale freeze с machine-readable report;
- pinned-single-core DPD-only Python/NumPy timing diagnostic с попарным
  1000-real-MUL scalar reference, exact streaming equivalence и hash-bound
  provenance. Это instrumentation evidence, не target hardware pass.

Пока **не выполнены**:

- bit-accurate evaluation selected spline-memory DPD;
- fixed-point PA→DPD cascade;
- synthesis/place-and-route на FPGA/ASIC;
- measured latency, throughput, DSP/LUT/BRAM use, power или timing closure.

Следовательно, completed PA fixed-point work доказывает только численную
реализуемость evaluator arithmetic. Он не обязан укладываться в DPD timing
budget и не заменяет bit-accurate/timed implementation самого predistorter.

Source of the completed PA arithmetic evidence:
`experiments/configs/pa_fixed_point_apa200.json`,
`experiments/evaluate_fixed_point_pa.py` and
`experiments/results/pa_fixed_point_apa200/fixed_point_report.json`.

## 2. Counting convention

Primary convention:

```text
1 complex multiply = 4 real MUL + 2 real ADD
1 real FMA = 1 real MUL + 1 real ADD
```

Отдельные columns:

- real multiplication;
- real addition/subtraction;
- real division;
- nonlinear operation (`sqrt`, activation, trigonometry);
- comparison;
- LUT access;
- real memory read/write;
- stored coefficient, constant and persistent state.

Optional Gauss complex multiply `3 MUL + 5 ADD` может быть отдельной hardware
schedule. Parameter count никогда не заменяет operations/sample. Для DPD ни
эта convention, ни одна колонка `MUL` не заменяет measured equivalent latency:
`div/sqrt/LUT/compare/memory` входят в суммарное время.

## 3. Current analytical Pareto points

Counts относятся к одному complex sample и используют persistent streaming
state. FP32 bytes ниже включают real coefficients + constants + state, но не
code, buffers, allocator, address logic или output queues.

| Model/dataset | MUL | ADD | Nonlinear | Compare | LUT | Reads/writes | Real coeff. | Constants | State reals | FP32 stored bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MP DPA | 792 | 502 | 0 | 24 | 0 | 288/2 | 240 | 29 | 46 | 1,260 |
| GMP DPA | 766 | 759 | 1 | 0 | 0 | 1,092/8 | 712 | 9 | 188 | 3,636 |
| MP APA | 960 | 628 | 30 | 30 | 0 | 360/2 | 300 | 35 | 58 | 1,572 |
| GMP APA | 954 | 947 | 1 | 0 | 0 | 1,362/8 | 888 | 9 | 236 | 4,532 |
| Legacy spline-memory DPD DPA | 21 | 24 | 1 | 5 | 6 | 18/2 | 144 | 47 | 4 | 780 |
| Legacy spline-memory DPD APA | 21 | 24 | 1 | 3 | 6 | 18/2 | 48 | 15 | 4 | 268 |
| APA standalone SPH (`K=32,L=8`) | **37** | **36** | 1 (`sqrt`) | 5 | 4 | 36/2 | **78** | **63** | **14** | **620** |
| APA sparse non-factorized (`K=12`, 6 branches) | **54** | **58** | 6 (`sqrt`) | 24 | 12 | 36/2 | **144** | **23** | **48** | **860** |
| APA lag-9 sparse non-factorized (`K=12`, 9 branches) | **72** | **82** | 6 (`sqrt`) | 24 | 18 | 54/2 | **216** | **23** | **48** | **1,148** |

Sources:

- MP/GMP: `baseline/complexity.py`, frozen selection/test manifests under
  `experiments/results/pa_{mp,gmp}_*`;
- spline-memory DPD:
  `experiments/results/spline_memory_{dpa200,apa200}/memory_ablation_report.json`.
- SPH: `experiments/results/pa_sph_apa200_selection/selection_manifest.json`
  and `selected_sph_pa.npz`; the exact counter is recomputed by
  `SplineHammersteinPA.operation_count()`.
- Sparse PA: `experiments/results/pa_sparse_spline_memory_apa200_selection/selection_manifest.json`
  and `selected_sparse_pa.npz`; the exact counter is recomputed by
  `SparseSplineMemoryPA.operation_count()`.
- Lag-9 sparse PA:
  `experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/selection_manifest.json`
  and `selected_sparse_pa.npz`; the exact counter is recomputed by the same
  `SparseSplineMemoryPA.operation_count()` implementation.

MP artifact warning: historical MP manifests were written before delay-line
state bookkeeping correction and contain stale zero state fields. The table
uses the corrected current counter: 46 DPA and 58 APA state reals. GMP and
spline-memory manifests already include state values shown above.

### 3.1 SPH cost audit and quality caveat

The selected SPH schedule uses one shared amplitude `sqrt`, binary interval
selection, two local control-point reads and a causal seven-tap FIR tail. The
counter convention is explicit:

```text
37 real MUL = (2 power + 1 coordinate/weight + 2 spline-interpolation
              + 4 current x*C) + 7*4 causal complex FIR products
36 real ADD = (1 power + 1 coordinate + 4 spline/interpolation + 2 current
              product) + 7*(2 FIR product + 2 accumulation)
1 sqrt, 5 comparisons, 4 LUT accesses
```

`h[0]=1+0j` is fixed and is neither stored nor multiplied. Storage is
`78 + 63 + 14 = 155` real values, i.e. `620` FP32 bytes before code/buffer
overhead. Streaming state is 7 complex nonlinear-output samples (14 real
values); reset and arbitrary-chunk equivalence are exact in complex128.

This is not a claim of hardware latency: the four LUT accesses and magnitude
primitive still need an RTL mapping, and memory traffic/address generation is
reported separately. More importantly, APA SPH train-OOF NMSE is −30.4024 dB,
6.652 dB worse than matched MP and 7.943 dB worse than GMP. The low cost must
therefore not be used to justify moving the DPD contour to SPH.

### 3.2 Sparse non-factorized PA cost audit and quality caveat

The selected sparse model has six branches
`(0,0),(1,1),(2,2),(22,22),(23,23),(24,24)` and twelve shared amplitude knots.
Envelope magnitude, interval address and interpolation weight are shared only
within equal envelope-delay groups. The exact schedule is:

```text
54 real MUL, 58 real ADD, 0 divisions,
6 sqrt nonlinear operations, 24 comparisons, 12 LUT accesses,
36 real reads, 2 writes, 144 real coefficient values,
23 constants, 48 persistent state reals.
```

The 48-state value includes the causal delay line through sample 24. FP32
coefficient/constant/state storage is `(144+23+48)*4 = 860 bytes`, excluding
code, buffers and interface queues. The model is phase-equivariant in floating
point and exact under arbitrary chunks/reset according to the published
bundle. It is still an analytical schedule, not a synthesized datapath.

The low count does not imply a useful evaluator by itself: train-OOF NMSE is
`−32.030011 dB`, versus `−38.345410 dB` for matched GMP and `−37.054329 dB`
for matched MP. The candidate therefore fails the internal cheap-Pareto gate
and must not be used to claim DPD quality or Huawei real-time readiness.

### 3.3 Residual-guided lag-9 sparse PA cost audit

The selected lag-9 topology adds three delayed-signal branches with the current
envelope to the six-branch parent:

```text
(m,d) = (0,0),(1,1),(2,2),(22,22),(23,23),(24,24),(8,0),(9,0),(10,0)
```

The exact schedule is:

```text
72 real MUL, 82 real ADD, 0 divisions,
6 sqrt nonlinear operations, 24 comparisons, 18 LUT accesses,
54 real reads, 2 writes, 216 real coefficient values,
23 constants, 48 persistent state reals.
```

There are six unique envelope delays, so the six magnitude/address primitives
are shared; there are nine unique signal delays, which explains the larger
read count and 72-MUL branch total. Storage is
`(216 + 23 + 48) * 4 = 1,148` FP32 bytes before code, buffers and interface
logic. The maximum delay remains 24 samples, so state memory does not grow
relative to the parent.

The model is a valid low-complexity point on the APA within-capture
identification frontier (`−37.792478 dB` train OOF, `−37.860728 dB` reused
validation), but it remains `0.552932 dB` behind matched GMP on full OOF and
has no fixed-point or synthesized hardware measurement yet. The 72-MUL count
must not be presented as FPGA DSP usage or as a DPD inference result.

### 3.4 Capture-transfer cost/quality evidence

The frozen `APA_200MHz -> APA_200MHz_b` release reuses exactly these inference
costs and FP32 storage values:

| Model / mode | Target test full NMSE | Common NMSE | Fit time | Host batch inference | MUL / ADD | FP32 model+state |
|---|---:|---:|---:|---:|---:|---:|
| GMP zero-shot | −23.795441 dB | −23.800907 dB | 0 | 0.03616 s | 954 / 947 | 4,532 B |
| GMP coefficient-only, N=16384 | **−37.895152 dB** | **−38.003839 dB** | 6.860 s | 0.03637 s | 954 / 947 | 4,532 B |
| lag-9 sparse zero-shot | −23.695838 dB | −23.700933 dB | 0 | 0.00620 s | 72 / 82 | 1,148 B |
| lag-9 sparse coefficient-only, N=16384 | −34.801474 dB | −35.437986 dB | 1.513 s | 0.00547 s | 72 / 82 | 1,148 B |

At `N=16384`, sparse is `13.25x` cheaper in real MUL, about `4.53x` faster
to fit and about `6.65x` faster in this host batch call, but is `3.094 dB`
worse than GMP on primary held-out full-record NMSE. These are analytical and
host-Python diagnostics, not FPGA/DSP latency. The release used two accesses
because the first failed before inference/metric; the second used unchanged
frozen choices. Incident and access count are part of the published bundle.

### 3.5 APA fixed-point PA arithmetic result

The sealed train→freeze→validation runner evaluates the same two frozen PA
models at signed 16, 14 and 12 bits.  Input/output full scales are derived
from train peaks (including the frozen model's train prediction peak for the
output guard); coefficient scales are derived from the frozen coefficient
peak.  Validation is descriptive only and cannot alter a format.

| Model / bits | Train PA NMSE | Validation PA NMSE | Fixed-vs-float NMSE (val) | MUL / ADD / DIV | Coeff bytes | State bytes |
|---|---:|---:|---:|---:|---:|---:|
| GMP / 16 | −38.7838 dB | −38.6459 dB | −64.20 dB | 954 / 947 / 0 | 1,776 | 472 |
| GMP / 14 | −38.5712 dB | −38.4320 dB | −51.62 dB | 954 / 947 / 0 | 1,554 | 413 |
| GMP / 12 | −34.3376 dB | −34.4282 dB | −36.41 dB | 954 / 947 / 0 | 1,332 | 354 |
| lag-9 sparse / 16 | −37.8661 dB | −37.8604 dB | −77.29 dB | 66 / 88 / 6 | 432 | 96 |
| lag-9 sparse / 14 | −37.8585 dB | −37.8523 dB | −65.29 dB | 66 / 88 / 6 | 378 | 84 |
| lag-9 sparse / 12 | −37.7570 dB | −37.7464 dB | −53.59 dB | 66 / 88 / 6 | 324 | 72 |

The sparse fixed schedule reports `66 MUL + 6 integer divisions`; if a target
implements each division as a reciprocal multiplication, the equivalent
multiplier budget is approximately `72 MUL`, matching the floating-point
operation convention's 72-MUL envelope/branch count.  The division is therefore
not hidden by the lower raw-MUL number.  All six rows reported zero input,
coefficient, power, interpolation, accumulator and output saturation, zero
knot collisions, and bit-identical chunk/reset checks.

The report's host timings (`~0.21 s` train GMP, `~0.05 s` train sparse for the
58,980-sample capture) are NumPy reference timings only.  They do not estimate
an FPGA initiation interval, DSP count or power.  The 12-bit GMP quality loss
is a negative result that must remain visible; sparse arithmetic is more
quantization-stable here, but its float PA fidelity is still below GMP.

### 3.6 DPA fixed-point GMP result

The same contract was rerun on the separate DPA source capture with its own
frozen hashes and 2,560-sample frame protocol.  No APA sparse topology was
transferred to DPA because it was not DPA-selected.

| Model / bits | Train PA NMSE | Validation PA NMSE | Fixed-vs-float NMSE (val) | Coeff bytes | State bytes |
|---|---:|---:|---:|---:|---:|
| DPA GMP / 16 | −35.4897 dB | −35.3633 dB | −75.83 dB | 1,424 | 376 |
| DPA GMP / 14 | −35.4832 dB | −35.3490 dB | −64.25 dB | 1,246 | 329 |
| DPA GMP / 12 | −35.4071 dB | −35.2975 dB | −52.53 dB | 1,068 | 282 |

The float validation reference is `−35.3659 dB` for all rows.  All formats
reported zero input/coefficient/power/accumulator/output saturation and exact
chunk equivalence.  This is a separate DPA forward-identification result; it
does not imply that an APA sparse model transfers to DPA or that either model
is a physical-PA DPD evaluator.

### 3.7 Target-calibrated `APA_200MHz_b` PA arithmetic

The frozen `N=16384` coefficient payloads selected in the immutable pre-test
capture-transfer bundle were evaluated on target train/validation. Provenance
binds the source topology, coefficient archive, selected model/N and exact
coefficient array. No target test file was named, hashed or opened.

| Model / bits | Float validation NMSE | Fixed validation NMSE | Fixed-vs-float NMSE | Coeff bytes | State bytes |
|---|---:|---:|---:|---:|---:|
| target GMP / 16 | −37.8908 dB | −37.8708 dB | −63.04 dB | 1,776 | 472 |
| target GMP / 14 | −37.8908 dB | −37.6427 dB | −50.00 dB | 1,554 | 413 |
| target GMP / 12 | −37.8908 dB | −35.1728 dB | −38.59 dB | 1,332 | 354 |
| target sparse / 16 | −35.3585 dB | −35.3576 dB | −76.22 dB | 432 | 96 |
| target sparse / 14 | −35.3585 dB | −35.3515 dB | −64.07 dB | 378 | 84 |
| target sparse / 12 | −35.3585 dB | −35.2650 dB | −52.15 dB | 324 | 72 |

All saturation/collision counters are zero and chunk equivalence is exact.
The 12-bit target GMP degradation (`2.718 dB`) is material even without
overflow; target sparse degradation is only `0.0935 dB`, but sparse starts
`2.532 dB` behind GMP in floating-point validation fidelity. This result
closes the planned PA-payload quantization cleanup. It is not subject to the
deployment DPD latency gate.

## 4. Interpreting the cost

### 4.1 GMP versus MP

DPA GMP is simultaneously 0.286 dB better on frozen test and 26 MUL/sample
cheaper than MP under current factorized schedule, but it uses:

- 712 versus 240 real coefficients;
- 188 versus 46 state reals;
- 1,092 versus 288 real reads/sample;
- 257 more additions/sample.

APA GMP improves test NMSE by 1.618 dB and saves only 6 MUL/sample; it grows
FP32 coefficient+state storage from 1,572 to 4,532 bytes. Поэтому GMP не
доминирует MP по memory traffic/power/area без hardware synthesis.

### 4.2 Spline local support

На sample активны только два соседних complex control points per branch.
Selected three-branch spline reuses the same envelope coordinate and needs
21 MUL/sample. Binary interval selection costs 5 comparisons for (K=24) DPA
и 3 для (K=8) APA; uniform-power knots could reduce addressing, but это
другая trained model/ablation, не бесплатная optimization существующего
result.

### 4.3 Sample-rate implication and why one count is insufficient

Даже около 1000 MUL/sample означает высокую aggregate rate. Следующие строки
относятся к PA evaluators и не являются DPD acceptance:

| Evaluator | Fs | Counted real MUL rate |
|---|---:|---:|
| DPA GMP | 800 MHz | 612.8 GMUL/s |
| APA GMP | 983.04 MHz | 937.82 GMUL/s |

Это не требует буквально столько отдельных multipliers: pipelining,
time/interleaving, SIMD and DSP packing меняют mapping. Но один serial MAC не
обеспечит target sample rate. Для DPD нужны target clock, allowed latency,
parallelism и эталонный kernel, после чего проверяется
`T_DPD/sample <= T_reference(1000 real MUL)`.

### 4.4 DPD-only host-Python timing diagnostic

Для frozen `signal_delay_012` выполнен один и тот же диагностический protocol:
первые 512 desired-input samples, CPU affinity `[0]`, а thread-control
environment variables NumPy/BLAS/OpenMP заданы равными 1. Два warm-up pair и
девять measured pair использованы для каждого chunk. В каждой паре DPD и scalar
Python reference на 1000 real products/sample запускались рядом, а порядок
чередовался. PA evaluator в измерение не входил.

Median measured DPD time:

| Dataset | chunk 1 | chunk 8 | chunk 64 | chunk 512 | Paired ratio DPD/reference, chunk 1 / 512 |
|---|---:|---:|---:|---:|---:|
| DPA | 177.496 µs/sample | 21.700 µs | 3.076 µs | 0.625 µs | 0.6778 / 0.002345 |
| APA | 186.089 µs/sample | 22.354 µs | 2.949 µs | 0.534 µs | 0.6633 / 0.002047 |

All eight chunk checks have `streaming_equivalent=true`: state resets once at
the stream start and is carried across calls. DPA/APA values are now
consistent for the same three-branch topology, unlike the rejected contended
exploratory run.

The timed implementation is Python control plus NumPy complex128 reference
arithmetic. It executes validation, allocation, division and memory work that
is not the optimized deployment schedule. Conversely, the scalar reference
also executes Python additions/conversions and is not the unknown customer
kernel. Therefore the published ratio is **not** a multiplication-equivalent
latency or Huawei pass/fail. The separate analytical DPD vector remains
`21 MUL, 24 ADD, 1 nonlinear, 5/3 comparisons, 6 LUT, 18 reads, 2 writes,
4 state reals`.

Evidence:

- `experiments/results/dpd_timing_dpa200_validation.json`;
- `experiments/results/dpd_timing_apa200_validation.json`;
- `experiments/benchmark_dpd_timing.py`.

## 5. Fixed-point implementations: exact scope and limitation

`baseline/fixed_point.py` реализует deterministic signed-integer reference для
memoryless complex spline:

- quantized I/Q, knots and complex control points;
- round-to-nearest-even;
- finite signed accumulator;
- explicit saturation counters;
- integer interpolation weights and local complex product.

`experiments/evaluate_fixed_point.py` применял его только к first-stage
memoryless spline DPD и старому surrogate. Он покрывает FP16-like storage,
signed 16-bit and signed 12-bit paths, но:

- не покрывает 14 bit;
- не покрывает spline-memory branches;
- не покрывает MP/GMP PA;
- magnitude использует rounded NumPy `sqrt`, а не выбранный RTL primitive;
- cascade остаётся `surrogate_only`;
- artifact explicitly declares `bit_true_rtl=false` и
  `hardware_latency_or_resources=false`.

Новый `baseline/fixed_point_pa.py` и
`baseline/fixed_point_sparse_spline_pa.py` покрывают selected causal GMP и
lag-9 sparse PA, а `experiments/evaluate_fixed_point_pa.py` sealed train/val
runner публикует их degradation.  Они всё ещё не являются RTL bit-true
доказательством: integer-sqrt implementation, division/reciprocal schedule,
accumulator tree ordering и memory interface должны быть зафиксированы в
target HLS/RTL. Архивные memoryless-DPD numbers поэтому не смешиваются с
новым PA report.

## 6. Required bit-accurate simulator

### 6.1 Numeric contract

Для каждого model family config обязан задавать:

```text
two's-complement signed representation
input I/Q word and fractional bits
coefficient word and fractional bits
state word and fractional bits
intermediate power/envelope formats
interpolation-weight format
product widths
accumulator width
rounding after every declared stage
saturation versus wrap policy
output word/scale
```

Обязательные formats:

- signed 16-bit coefficients and activations;
- signed 14-bit coefficients and activations;
- signed 12-bit coefficients and activations;
- FP32 reference;
- optional FP16-like storage только как software reference.

Accumulator width не обязан совпадать с activation width; он выбирается до
test по analytical worst case + train calibration and then frozen.

### 6.2 PA GMP integer datapath

Simulator должен воспроизвести stage by stage:

1. input/state quantization;
2. (q=I^2+Q^2);
3. shared amplitude/power basis, включая explicit `sqrt`/LUT/CORDIC choice;
4. aligned/lagging/causal-leading envelope delays;
5. real-scalar × complex input products;
6. complex coefficient products;
7. accumulation order exactly matching intended hardware tree;
8. output rounding/saturation;
9. persistent state across arbitrary chunks and reset at declared frames.

Changing from amplitude to power-coordinate basis to remove `sqrt` requires
retraining and отдельный model label; это не arithmetic-only quantization.

### 6.3 Spline-memory DPD datapath

Simulator должен включать:

1. shared envelope magnitude/power primitive;
2. deterministic knot interval address;
3. quantized interpolation fraction;
4. two local complex coefficient reads per branch;
5. complex interpolation;
6. delayed complex signal fetch;
7. branch complex products and fixed-order accumulation;
8. saturation counters and peak-drive guard.

Uniform, power-domain and nonuniform/quantile knots получают разные addressing
costs; каждый variant считает comparisons/LUT/constants отдельно.

## 7. Verification tests before numerical sweep

Required unit/property tests:

- known scalar quantization vectors and tie-to-even cases;
- positive/negative saturation boundaries;
- accumulator overflow detection;
- fixed reset state;
- full-record equals arbitrary chunked streaming bit-for-bit;
- no future-sample dependence for causal models;
- coefficient/model serialization round trip;
- operation/state counter matches selected topology;
- no non-finite float conversion;
- phase-equivariance degradation reported, not assumed zero after Cartesian
  quantization;
- repeatability independent of NumPy/BLAS threading.

## 8. Fixed-point evaluation matrix

Architecture, scaling policy and accumulator rules are frozen from train and
the preregistered format matrix; validation is descriptive and cannot choose or
retune a format.  The test split is not part of the current PA arithmetic
runner and remains sealed.

For each model/format publish:

- pooled/OpenDPD NMSE degradation versus FP32;
- ACLR left/right/average degradation;
- EVM degradation;
- error/output PSD delta;
- PAPR and peak-drive delta;
- input/coefficient/state/accumulator/output saturation counts;
- maximum accumulator magnitude and headroom;
- coefficient/state bytes at actual bit widths;
- LUT/address storage;
- bit-identical full/chunk result;
- host reference runtime, separately from target hardware measurements.

Provisional internal engineering gate, not Huawei requirement (not yet applied
to a physical PA):

```text
NMSE degradation <= 0.25 dB
average ACLR degradation <= 0.5 dB
no accumulator/output saturation on validation/test
no new support violation or instability
```

Если 12 bit fails, negative result публикуется; нельзя менять scale по test.

## 9. Hardware implementation report contract

Analytical result становится hardware result только после указания:

- FPGA/ASIC/DSP part and tool versions;
- clock frequency and sample-interface width;
- initiation interval and end-to-end latency;
- sustained complex samples/s;
- DSP, LUT/ALM, FF, BRAM/URAM, routing utilization;
- on-chip/off-chip coefficient and state bandwidth;
- timing slack;
- measured/simulated power and measurement method;
- RTL/HLS commit and bit-exact vector hashes.

Coefficient update/calibration hardware считается отдельно от inference
datapath. Closed-form ridge solve на CPU и coefficient download может быть
приемлемым online calibration path, но не является “real-time coefficient
solving” без update-time requirement Huawei.

## 10. Следующая hardware задача

PA arithmetic coverage достигнута на DPA/APA source captures и на frozen
target-calibrated `APA_200MHz_b` coefficient payloads. Target cleanup выполнен
на train/validation с hash-bound provenance без target-test access. На этом
расширение PA quantization без нового evidence need останавливается.

Следующий deployment-relevant hardware step — selected spline-memory DPD
fixed-point path и correct desired-x→DPD→frozen-PA cascade. Host timing
instrumentation уже проверена; после получения target/timing-reference
definition она должна быть заменена измерением latency и throughput всех
операций DPD на целевой реализации относительно customer reference kernel.
Synthesis/throughput следует запускать лишь после выбора конкретного
word-length/reciprocal/sqrt implementation; до этого analytical counts
помечаются lower bounds.
