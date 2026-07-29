# Hardware cost and fixed-point contract

Дата среза: 2026-07-29.

## 1. Статус и область доказательства

Для selected MP, causal GMP и legacy spline DPD имеются analytical
operation/state counts. Для causal GMP также доказана equivalence NumPy
full-record, reset-per-frame и arbitrary streaming chunks в floating-point.

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

Sources:

- MP/GMP: `baseline/complexity.py`, frozen selection/test manifests under
  `experiments/results/pa_{mp,gmp}_*`;
- spline-memory DPD:
  `experiments/results/spline_memory_{dpa200,apa200}/memory_ablation_report.json`.

MP artifact warning: historical MP manifests were written before delay-line
state bookkeeping correction and contain stale zero state fields. The table
uses the corrected current counter: 46 DPA and 58 APA state reals. GMP and
spline-memory manifests already include state values shown above.

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

Сначала расширить integer reference на selected causal GMP PA и доказать
bit-identical streaming at 16 bit. Затем добавить 14/12 bit и только после
этого spline-memory DPD/cascade. Такой порядок изолирует degradation PA
evaluator от degradation DPD и не смешивает две новые реализации в одном
experiment.
