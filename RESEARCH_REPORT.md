# Research report: low-complexity PA identification and DPD

Дата среза: 2026-07-30.

## 1. Outcome

Проект переформулирован из одной surrogate-DPD optimization задачи в два
независимых контура:

```text
A. measured x -> PA identifier -> y_hat; compare with measured y
B. desired x -> DPD -> independently frozen PA/physical PA -> compare with g*x
```

Контур A воспроизводимо выполнен для MP и causal factorized GMP на
`DPA_200MHz`/`APA_200MHz`. Контур B имеет legacy surrogate-only spline
evidence, но новая optimization приостановлена: Gate A→B закрыт.

Лучший текущий forward result:

| Dataset | Causal GMP test NMSE | Real MUL/ADD per sample | Complex coeff. |
|---|---:|---:|---:|
| DPA_200MHz | −35.385 dB | 766 / 759 | 356 |
| APA_200MHz | −38.608 dB | 954 / 947 | 444 |
| APA_200MHz lag-9 sparse (reused validation) | −37.861 dB | 72 / 82 | 216 |
| APA_200MHz_b calibrated GMP (held-out test) | −37.895 dB | 954 / 947 | 444 |
| APA_200MHz_b calibrated lag-9 sparse (held-out test) | −34.801 dB | 72 / 82 | 216 |

Если Huawei `10^-5` означает normalized error power, target равен −50 dB и
не достигнут. Никакого claim “лучше OpenDPD” или “готово для Huawei base
station” нет.

## 2. Physical and mathematical formulation

Desired complex baseband:

\[
x[n]=I[n]+jQ[n].
\]

Для PA (P), DPD (D) и train-frozen complex gain (g):

\[
z[n]=D(x[n]),\qquad y[n]=P(z[n]),\qquad y[n]\approx g x[n].
\]

Deployment metric:

\[
\operatorname{NMSE}_{pool}=10\log_{10}
\frac{\sum_n|P(D(x[n]))-g x[n]|^2}
     {\sum_n|g x[n]|^2}.
\]

ILA fit допустим как postdistorter calibration:

\[
u[n]=y[n]/g,\qquad D_{post}(u[n])\approx x[n].
\]

Но diagnostic

\[
y_{test}/g\to D_{post}\to \hat P\to y_{test}
\]

оценивает (\lVert\hat P(D_{post}(y/g))-y\rVert^2), а не требуемое
(\lVert P(D(x))-gx\rVert^2). Он может показать inverse/forward consistency,
но не linearization нового desired input.

## 3. Requirements audit

Предоставленные slides явно требуют strong expression, low complexity,
real-time-friendly coefficients, verification error below `10^-5` и fewer than
1000 real multipliers. Они также называют Volterra/MP, neural and CPWL models
и memory sources: signal bandwidth, thermal and trapping effects.

Slides не определяют:

- error normalization/aggregation;
- относится ли error к forward PA, inverse или cascade;
- operations/sample versus physical DSP blocks;
- carrier/output power/backoff/bias/temperature;
- feedback-path correction and spectral masks;
- fixed-point words/accumulator/rounding;
- update-time and target hardware.

Поэтому проект публикует MSE, relative error power, pooled NMSE and
OpenDPD-compatible NMSE separately. Только если `10^-5` — normalized error
power:

\[
10\log_{10}(10^{-5})=-50\ \mathrm{dB}.
\]

Detailed contract: `REQUIREMENTS.md`.

## 4. Repository audit

### 4.1 OpenDPD main

Code audit at vendored commit
`7426bbf8a47624b59bd7f045a86641b403023f3c` established:

- neural DLA direction is correct: desired `x -> DPD -> frozen PA`, target
  `g*x`;
- MP/GMP benchmark uses ILA fit but transfers coefficients and evaluates on
  desired `X_val/X_test`;
- PA behavioral modeling and DPD learning are distinct stages;
- built-in results rely on a learned PA surrogate unless physical results are
  taken from the paper;
- loader does not independently perform feedback-path de-embedding/alignment;
- state/context includes noncausal/offline semantics in several TRes/TCN paths;
- temporal zeros do not automatically reduce eager dense matrix runtime;
- stored parameter count is not operations/sample;
- quantization is partial fake-quant/software evidence, not bit-true hardware;
- archived neural checkpoint binaries referenced by bundled JSON are absent.

The bundled APA PA reference reports approximately −38.70 dB GMP and up to
−39.18 dB TRes-DeltaGRU test NMSE, but local neural rerun is unavailable.
Our causal APA GMP reaches −38.608 dB at 954 counted MUL/sample and zero
lookahead; the small quality difference versus bundled GMP is not an
apples-to-apples superiority claim because solver/boundary/cost semantics
differ.

Detailed file/line audit: `research/opendpd_audit.md`.

### 4.2 Egor DPD_for_PA and chaotic_library

Independent code/notebook audit confirmed:

- data are OpenDPD `DPA_200MHz` train/test; validation is absent;
- postdistorter training mapping `y/g -> x` is valid ILA in principle;
- notebook cells 10 and 14 use circular
  `y_test/g -> inverse -> PA surrogate -> y_test`;
- cell 11 uses desired `x_test`, but lacks a complete correct-direction scalar
  RF benchmark;
- cached PA/circular NMSE is around −31.67/−32.09 dB, far from −50 dB;
- PA reservoirs use (R=800), DPD (R=600), with dense NumPy `W @ state`;
- DPD pair costs about 728,622 MUL/sample and full learned cascade about
  2,020,044, despite ~10% nonzero initialization;
- `predict` begins with zero state; ordinary chunks are not streaming
  equivalent;
- independent random I/Q models do not enforce phase equivariance;
- notebook PSD uses `fs=200`, `nperseg=256`, whereas dataset requires
  800 MHz and 2560.

Thus high (R^2), coefficient sparsity and small serialized readout do not
establish NMSE/ACLR quality or real-time cost.

Detailed hypotheses, paths and line references:
`research/egor_pipeline_audit.md`.

## 5. Literature synthesis

The source review uses original papers/arXiv/DOI repositories and separates
incomparable PA groups. Full table: `research/literature_review.md` and
`research/comparison_table.csv`.

### 5.1 Main evidence groups

| Work/group | Evidence | Main lesson | Directly comparable here? |
|---|---|---|---|
| OpenDPDv2, APA_200MHz | physical 3.5 GHz GaN Doherty, 200 MHz; TRes-DeltaGRU about −39.6 NMSE, −42.1 EVM, −59.9 dBc avg ACPR | strong physical neural reference; 999 params ≠ <1000 real MUL | only after same evaluator/checkpoint/measurement |
| TCN-DPD, DPA_200MHz | frozen neural surrogate, multiple seeds | convolution can improve wideband memory modeling; causality/physical test gap remains | same CSV group, different surrogate protocol |
| SparseDPD, 20 MHz | surrogate RF + FPGA post-implementation simulation | structured sparsity needs zero-skipping datapath | no, different data/PA/BW |
| Spline-interpolated LUT E1/E2/E3 | physical PA, spline Hammerstein/memory branches | local support + short memory can approach MP quality at tens of MUL | architecture prior only |
| Piecewise closed-loop DPD | physical 28 GHz active array | low-complexity online adaptation matters under load/beam drift | no, different task |
| Feature-selected GMP/PNN, FR3 | physical DUT-specific feature selection | selection can reduce inference but offline search cost/generalization matter | no |
| PN-RNN | physical 200 MHz | phase-normalization/state improve inductive bias | no common operations table |
| DPD-NeuralEngine | 22 nm post-layout, 250 MS/s | compact GRU can map to hardware, but post-layout ≠ fabricated base-station proof | no |

No external row is mixed into a single numerical leaderboard with our DPA/APA
results. Direct OpenDPD comparison requires the same capture/evaluator and
metric conventions.

### 5.2 Architectural conclusion

Literature supports this ladder:

1. phase-equivariant complex local spline as cost floor;
2. short spline memory branches or spline→FIR Hammerstein;
3. sparse GMP/CPWL dictionary with group selection;
4. state-conditioned spline only with long-capture evidence;
5. tiny neural residual only if structured models leave reproducible residual.

## 6. Corrected PA experimental pipeline

The implemented forward protocol freezes from train:

- integer delay and fractional-delay diagnostic;
- complex-LS/peak-gain diagnostics;
- frame length and reset policy;
- AM/AM/AM/PM bins and input support;
- PSD/Welch settings;
- common warm-up/cooldown.

A0/A1 sensitivity rejected fractional transform for both datasets; A0 integer
delay zero/no transform was frozen before GMP selection. Fractional deltas were
far below the preregistered −0.25 dB gate and APA OOF full-record reversed
sign. This is sensitivity evidence, not measurement-path calibration.

Formal GMP workflow:

1. 154 validation fits per dataset under strict `<1000 MUL` filter;
2. frozen train refit and hash-bound model;
3. coefficient-OOF residual audit against matched MP;
4. rank/conditioning/support/boundary/streaming release gate;
5. one frozen test call per dataset, no refit/gain/delay tuning.

Complete commands and hashes: `EXPERIMENT_PLAN.md`.

## 7. PA results and evaluator limit

| Dataset | MP val/test | GMP val/test | GMP OOF full/common | GMP cost |
|---|---:|---:|---:|---:|
| DPA | −34.962/−35.099 | −35.366/−35.385 | −35.316/−35.422 | 766 MUL, 3,636 FP32 stored bytes |
| APA | −37.095/−36.990 | −38.665/−38.608 | −38.345/−38.751 | 954 MUL, 4,532 FP32 stored bytes |

GMP improves MP but does not reach −50 dB. Existing spline DPD versus GMP
fidelity gives only arithmetic—not cascade—margin:

| Dataset | DPD test residual | GMP test error | Margin | Internal target |
|---|---:|---:|---:|---:|
| DPA | −29.864 | −35.385 | 5.521 dB | 10 dB |
| APA | −32.741 | −38.608 | 5.867 dB | 10 dB |

Evaluator error is about 28.0%/25.9% of DPD residual power, rather than the
internal ceiling 10%. A DPD optimizer could exploit structured surrogate error.
Therefore Gate A→B is closed.

Detailed analysis: `PA_MODEL_BENCHMARK.md` and `FINAL_GAP_ANALYSIS.md`.

Отдельный lag-9 sparse PA достиг `−37.792478 dB` train-OOF full NMSE и
`−37.860728 dB` reused validation при 72 MUL/sample. Это улучшение MP по
quality/cost внутри APA capture, но оно уступает GMP на `0.552932 dB` full OOF
и не является evaluator для DPD. На независимом `APA_200MHz_b` capture
zero-shot обе families дают около −23.8 dB; после `N=16384` coefficient-only
calibration held-out test даёт GMP `−37.895 dB`, sparse `−34.801 dB`. Это
capture-transfer evidence, но не controlled power/thermal result.

## 8. DPD baseline and proposed method

### 8.1 Legacy result

The selected three-branch complex spline

\[
z[n]=\sum_{m=0}^{2}x[n-m]C_m(|x[n]|)
\]

uses 21 real MUL, 24 ADD and one nonlinear magnitude operation/sample. It
achieved −29.864 dB DPA and −32.741 dB APA only through the old MP surrogate.
This is a valuable cost baseline, not physical/cross-evaluator proof.

### 8.2 Candidate 0 formulation

Memoryless phase-equivariant spline:

\[
z[n]=x[n]C(|x[n]|),
\]

with two locally active complex control points. Complex ILA ridge:

\[
\Phi_{n,k}=u[n]B_k(|u[n]|),\qquad
\hat c=\arg\min_c\lVert\Phi c-x\rVert^2+
\lambda\lVert c\rVert^2+\mu\lVert D_2c\rVert^2.
\]

Candidate knot counts `{8,12,16,24,32,48,64}` and uniform-amplitude,
uniform-power, quantile and compression-aware strategies remain valid. One
complex solve is preferred over unrelated I/Q models.

### 8.3 Memory extensions

| Candidate | Nominal cost | Calibration | Main risk |
|---|---:|---|---|
| Memoryless spline | 9 MUL + 8 ADD + sqrt | small complex ridge | no memory |
| 2–3 spline branches | roughly 15–27 MUL with shared envelope | group ridge | correlated branches |
| Spline→short FIR | `9+4L` MUL | alternating/ridge | scale ambiguity |
| Sparse spline/GMP/CPWL | selected branches/terms | OMP/group LASSO | selection leakage/instability |
| State-conditioned spline | local 2-D LUT + one-pole states | fixed-beta ridge | unidentifiable on short captures |
| Tiny residual TCN | tens/hundreds MAC + activations | E2E | surrogate exploitation/latency |

New complexity is allowed only after ablation. The bounded residual-guided
lag-9 sparse spline-memory **forward PA model** experiment is complete and is
the current low-complexity Pareto point. Independent `APA_200MHz_b`
transfer/adaptation is also complete through held-out test: GMP is the
quality point, sparse is the cheaper point. Further local delay expansion and
DPD optimization remain paused until controlled operating-point/physical-PA
evidence closes the evaluator gap.

Detailed shortlist: `research/proposed_methods.md`.

## 9. Hardware and fixed point

Current operation accounting separates MUL, ADD, nonlinear, comparisons,
lookups, reads/writes, coefficients and state. At native Fs, APA GMP represents
937.82 GMUL/s aggregate analytical work, so `<1000/sample` alone does not prove
real time.

Existing integer reference covers only first-stage memoryless spline 16/12-bit
software arithmetic and declares `bit_true_rtl=false`. Selected GMP and
lag-9 sparse PA fixed-point evaluation is still pending. Required order:

1. selected causal GMP integer datapath at 16 bit;
2. 14/12 bit ablation with frozen scale/accumulator;
3. spline-memory DPD;
4. cascade;
5. FPGA/ASIC synthesis and measured throughput/latency/resources/power.

Detailed contract: `HARDWARE_COST.md`.

## 10. Robustness and adaptation

`DPA_160MHz` and `APA_200MHz_b` are available, but capture identity/power/bias/
temperature metadata are insufficient. They must be treated as capture
transfer, never pooled random splits.

The source-frozen `APA_200MHz -> APA_200MHz_b` zero-shot PA modeling and
limited coefficient recalibration is complete, including held-out release:

| Model / mode | Target test full NMSE | Common NMSE | Fit time | MUL/sample |
|---|---:|---:|---:|---:|
| GMP zero-shot | −23.795441 dB | −23.800907 dB | 0 | 954 |
| GMP coefficient-only, N=16384 | **−37.895152 dB** | **−38.003839 dB** | 6.860 s | 954 |
| lag-9 sparse zero-shot | −23.695838 dB | −23.700933 dB | 0 | 72 |
| lag-9 sparse coefficient-only, N=16384 | −34.801474 dB | −35.437986 dB | 1.513 s | 72 |

Target nuisance alignment was estimated from target train only. The release
audit records two accesses because the first failed before inference/metric;
the retry used unchanged frozen choices. This is capture-transfer evidence,
not known power/thermal adaptation.

Detailed protocol: `ROBUSTNESS_AND_ADAPTATION.md`.

## 11. Claims and limitations

Supported:

- low-complexity causal GMP forward PA identification on held-out measured
  captures under the project operation convention;
- fixed-topology coefficient-only transfer calibration on `APA_200MHz_b`
  held-out test, with explicit two-access incident audit;
- lag-9 sparse PA is a reproducible 72-MUL cheap-Pareto point relative to MP
  within the APA capture, with exact floating-point streaming/reset behavior;
- correct pipeline guards, OOF/release/test provenance and streaming software
  equivalence;
- legacy spline establishes an extremely cheap surrogate-only DPD point;
- Egor circular evaluation and dense reservoir cost problems are confirmed.

Not supported:

- Huawei `10^-5` acceptance;
- physical PA DPD linearization;
- better-than-OpenDPD claim;
- fixed-point GMP/DPD or FPGA readiness;
- generalization/adaptation under controlled drift or known physical axes;
- thermal/state model evidence.

The next high-information experiment is a bit-accurate selected-PA audit,
followed by controlled operating-point metadata/capture. The decisive final
experiment remains calibrated physical PA remeasurement of
no-DPD, OpenDPD and the selected low-cost candidate on the same waveform,
operating point and spectral evaluator. Until then all DPD conclusions must
retain their surrogate/forward-model evidence label.
