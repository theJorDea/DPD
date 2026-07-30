# Hardware cost and fixed-point contract

Дата среза: 2026-07-30.

## 1. Статус и область доказательства

Для selected MP, causal GMP, legacy spline DPD, APA SPH, первого
non-factorized sparse spline-memory PA и residual-guided lag-9 sparse PA имеются
analytical operation/state counts. Для causal GMP и SPH также доказана
equivalence NumPy full-record, reset-per-frame и arbitrary streaming chunks в
floating-point. Lag-9 sparse PA также прошёл exact streaming/reset checks и
является internal cheap-Pareto point, но не независимым evaluator.

Пока **не выполнены**:

- bit-accurate 16/14/12-bit evaluation selected PA GMP;
- bit-accurate evaluation selected spline-memory DPD;
- fixed-point PA→DPD cascade;
- synthesis/place-and-route на FPGA/ASIC;
- measured latency, throughput, DSP/LUT/BRAM use, power или timing closure.

Следовательно, `<1000 real multiplications/sample` сейчас является analytical
software-schedule constraint, а не количеством физических multipliers в
конкретном Huawei device.

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
schedule, но не используется для текущих `<1000` decisions. Parameter count
никогда не заменяет operations/sample.

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

### 4.3 Sample-rate implication

Даже `<1000 MUL/sample` означает высокую aggregate rate:

| Evaluator | Fs | Counted real MUL rate |
|---|---:|---:|
| DPA GMP | 800 MHz | 612.8 GMUL/s |
| APA GMP | 983.04 MHz | 937.82 GMUL/s |

Это не требует буквально столько отдельных multipliers: pipelining,
time/interleaving, SIMD and DSP packing меняют mapping. Но один serial MAC не
обеспечит target sample rate. Нужны target clock, allowed latency, parallelism
и physical DSP-block definition Huawei.

## 5. Existing fixed-point code: exact limitation

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

Поэтому архивные small degradation numbers нельзя переносить на новый PA/DPD
pipeline или выдавать за hardware result.

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

Architecture, scaling policy and accumulator rules выбираются по train;
validation выбирает allowed format; test используется один раз only after
freeze.

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

Provisional internal engineering gate, not Huawei requirement:

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

Сначала расширить integer reference на selected causal GMP, parent sparse PA и
lag-9 sparse PA, доказать bit-identical streaming at 16 bit, а затем добавить
14/12 bit.
Только после раздельной PA quantization перейти к spline-memory DPD/cascade.
Такой порядок изолирует degradation PA evaluator от degradation DPD и не
смешивает две новые реализации в одном experiment.
