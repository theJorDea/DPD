# Benchmark report

Дата среза: 2026-07-29.

## 1. Scope

Этот report содержит только реально выполненные local runs и отдельно
маркирует upstream/legacy evidence. Primary completed task — forward PA
identification:

```text
measured x -> frozen PA model -> y_hat -> compare with measured y
```

Новый DPD-through-GMP cascade, physical PA experiment, 14-bit GMP и hardware
synthesis не выполнялись. Поэтому report не утверждает better-than-OpenDPD.

## 2. Environment and provenance

| Item | Value |
|---|---|
| CPU | Intel Core i5-12450H, 12 logical CPUs |
| RAM | 15 GiB |
| Accelerator | no detected NVIDIA GPU |
| Python | 3.14.6 |
| NumPy | 2.5.1 |
| Platform | Linux 7.0.5-2-cachyos, glibc 2.44 |
| OpenDPD vendored commit | `7426bbf8a47624b59bd7f045a86641b403023f3c` |
| Primary protocol | A0, integer delay 0, no fractional transform |
| Common GMP support | warm-up 49, cooldown 0 samples/frame |
| Complex multiply | 4 real MUL + 2 real ADD |
| Operation gate | strict `<1000` real MUL/complex sample |

Exact commands/configs/hashes: `EXPERIMENT_PLAN.md` and result execution
records. Selection and residual stages read train/validation only. GMP test
was opened once per dataset after release-gate PASS.

## 3. Datasets

| Dataset | Train / validation / test | Fs | Frame | Declared waveform |
|---|---:|---:|---:|---|
| DPA_200MHz | 23,040 / 7,680 / 7,680 | 800 MHz | 2,560 | 10×20 MHz LTE, 64-QAM |
| APA_200MHz | 58,980 / 19,662 / 19,662 | 983.04 MHz | 19,662 | 5-carrier TM3.1a, 256-QAM |

DPA and APA are different PA/captures and are never pooled.

## 4. Alignment sensitivity result

A1 applied a train-frozen frame-safe 65-tap windowed-sinc fractional transform
with equal 32-sample guard crop. Negative delta favors A1.

| Dataset | GMP OOF common Δ A1−A0 | GMP validation common Δ | Full-record check | Decision |
|---|---:|---:|---|---|
| DPA | −0.00114 dB | −0.00017 dB | same sign | A0 |
| APA | −0.00841 dB | −0.00654 dB | OOF sign reversed: +0.01070 dB | A0 |

The preregistered A1 threshold was −0.25 dB on OOF and validation common
support. A1 failed. Result artifact:
`experiments/results/pa_alignment_protocol_decision.json`.

This is a negative experiment: fractional correlation diagnostic did not
justify changing the evaluator. It does not prove measurement-path
de-embedding.

## 5. Forward PA quality

### 5.1 Validation-selected MP and GMP

Full-record pooled NMSE is primary. OpenDPD-compatible NMSE averages
per-complete-frame dB; common score excludes the preregistered boundary.

| Dataset/model | Validation full | Test full | Test OpenDPD | Test common | Relative test error power |
|---|---:|---:|---:|---:|---:|
| DPA MP | −34.9617 | −35.0990 | −35.1018 | historical 29-sample steady: −35.1607 | 3.0910e−4 |
| DPA causal GMP | −35.3659 | −35.3850 | −35.3983 | −35.4192 | 2.8940e−4 |
| APA MP | −37.0952 | −36.9905 | −36.9905 | historical 29-sample steady: −37.0745 | 1.9996e−4 |
| APA causal GMP | −38.6653 | −38.6081 | −38.6081 | −38.7075 | 1.3778e−4 |

GMP test improvement over MP:

- DPA: 0.2860 dB;
- APA: 1.6176 dB.

If `10^-5` is normalized error power, all rows fail the −50 dB target. GMP is
28.94×/13.78× above the threshold for DPA/APA.

### 5.2 Selected GMP topology

| Dataset | Topology | Ridge/solver | Complex coeff. | Required history |
|---|---|---|---:|---:|
| DPA | `ka7/la24`, `kb4/lb24/mb1`, causal `kc4/lc24/mc1` | 1e−5 / column-scaled complex ridge `lstsq` | 356 | 24 samples |
| APA | `ka7/la30`, `kb2/lb30/mb2`, causal `kc2/lc30/mc2` | 1e−7 / column-scaled complex ridge `lstsq` | 444 | 31 samples |

Both models are causal, zero-lookahead and full rank in the final fit.

## 6. OOF and residual release evidence

| Dataset | GMP train OOF full/common | GMP validation full/common | OOF gain over matched MP full/common | OOF→validation full/common |
|---|---:|---:|---:|---:|
| DPA | −35.3157/−35.4224 | −35.3659/−35.4684 | 0.2952/0.3009 | 0.0503/0.0460 dB |
| APA | −38.3454/−38.7505 | −38.6653/−38.7346 | 1.2911/1.6506 | 0.3198/−0.0159 dB |

Fold diagnostics:

| Dataset | Fold count | Condition/full ratio range | Coeff-norm/full ratio range | Max held/fit amplitude |
|---|---:|---:|---:|---:|
| DPA | 9 | 0.9952–1.0104 | 0.9829–1.0332 | 1.000000001 |
| APA | 3 | 0.9984–1.0043 | 0.9941–1.0639 | 1.0003002 |

All folds were full rank. Streaming and reset-frame maximum error was exactly
0 at `rtol=atol=1e-12`. Slow-state gate remained false because there are no
independent long captures.

These frame folds are not independent physical captures, so no confidence
interval across hardware sessions is reported. The closed-form models are
deterministic (`seed=null`); neural three-seed variance is not applicable to
these runs. Reporting an artificial statistical CI would overstate evidence.

## 7. Complexity and memory

| Model | MUL | ADD | Nonlinear | Reads/writes | Real coeff. | State reals | FP32 coeff+const+state |
|---|---:|---:|---:|---:|---:|---:|---:|
| MP DPA | 792 | 502 | 0 | 288/2 | 240 | 46 | 1,260 B |
| GMP DPA | 766 | 759 | 1 | 1,092/8 | 712 | 188 | 3,636 B |
| MP APA | 960 | 628 | 30 | 360/2 | 300 | 58 | 1,572 B |
| GMP APA | 954 | 947 | 1 | 1,362/8 | 888 | 236 | 4,532 B |

GMP is the quality winner but does not dominate memory traffic/storage. These
are analytical factorized schedules, not FPGA resource measurements.

## 8. Timing

| Task | DPA | APA | Meaning |
|---|---:|---:|---|
| Formal GMP selection, 154 fits | 70.988 s | 212.762 s | full process wall |
| Final selected fit | 1.234 s | 5.555 s | train coefficient solve |
| OOF/residual process | 10.259 s | 24.872 s | train OOF + validation diagnostics |
| Frozen-test process | 0.066 s | 0.31 s | no fit; process wall measurement differs by method |
| Test predictor single batch | 8.673 ms | 30.286 ms | NumPy batch diagnostic |
| Test batch throughput | 0.885 Msample/s | 0.649 Msample/s | host software, not real-time target |

Peak RSS was not measured. The host throughput is far below 800/983.04
MSample/s and does not represent a factorized FPGA implementation.

## 9. Legacy DPD benchmark: surrogate only

Best old spline branch was `signal_delay_012`, evaluated through an old MP PA
surrogate:

| Metric | DPA | APA |
|---|---:|---:|
| No DPD pooled NMSE | −20.1886 | −19.9477 |
| With spline DPD pooled NMSE | −29.8645 | −32.7408 |
| Legacy surrogate fidelity | −30.1305 | −31.0908 |
| Spectral EVM | −32.8672 | −33.8163 |
| ACLR left/right/average | −37.101/−38.864/−37.983 | −44.375/−41.663/−43.019 |
| Peak predistorted amplitude | 1.1865 | 0.9927 |
| PAPR | 10.008 dB | 9.993 dB |
| Spline cost | 21 MUL, 24 ADD | 21 MUL, 24 ADD |

This table is not rerun through causal GMP and is not physical PA evidence.

Arithmetic GMP fidelity margin is 5.521 dB DPA and 5.867 dB APA, below the
internal 10 dB Gate A→B. Therefore Gate A→B remains closed.

## 10. OpenDPD reference layer

Bundled APA report on matching CSV hashes records:

| PA model | Validation/test NMSE | Provenance |
|---|---:|---|
| OpenDPD GMP | −38.702/−38.661 dB | bundled closed-form report; future-envelope/offline semantics |
| GRU-H28 | −38.850/−38.937 dB | checkpoint referenced but binary absent |
| TRes-GRU-H27 | −39.045/−39.129 dB | checkpoint referenced but binary absent |
| TRes-DeltaGRU-H27 | −39.093/−39.178 dB | checkpoint referenced but binary absent |

These are not local reruns. The local causal GMP differs from bundled GMP by
only 0.037/0.053 dB, but boundary/solver/operation conventions differ; no
superiority/equivalence claim is made.

## 11. Fixed-point, robustness and hardware

Current status:

| Item | Result |
|---|---|
| First-stage memoryless spline FP16/16/12 numerical emulation | completed, surrogate-only, not selected spline-memory DPD |
| Selected GMP 16/14/12 bit | not run |
| Selected spline-memory DPD 16/14/12 bit | not run |
| GMP→DPD integer cascade | not run |
| APA_200MHz_b capture transfer/adaptation | not run |
| FPGA/ASIC synthesis and latency/throughput | not run |
| Physical predistorted PA measurement | not available |

Details: `HARDWARE_COST.md` and `ROBUSTNESS_AND_ADAPTATION.md`.

## 12. Negative and failed experiments

- A1 fractional alignment failed preregistered improvement gate; A0 retained.
- DPA GMP gives only 0.286 dB test gain despite larger coefficient/state
  storage.
- Neither GMP reaches possible −50 dB requirement.
- PA evaluator margin remains below 10 dB; DPD stage remains blocked.
- Slow-state candidate lacks independent-capture evidence.
- Local OpenDPD neural reproduction is blocked by missing checkpoint binaries
  and no GPU.
- Existing Egor circular score does not establish deployment DPD and dense
  reservoir cost exceeds the multiplier gate by orders of magnitude.
- Peak RSS, physical hardware latency and capture-level confidence intervals
  are unavailable; they are not estimated as measurements.

## 13. Plots and raw artifacts

GMP `test_evaluation.json` contains numerical error PSD, AM/AM and AM/PM
residual arrays; `test_prediction.npz` contains frozen predictions. New
rendered plots were intentionally not generated inside sweeps. At this
snapshot no canonical GMP plot bundle is claimed complete; plot generation is
a separate deterministic post-processing task and must not alter metrics.

Raw locations:

- `experiments/results/pa_gmp_dpa200_{selection,residuals,test}/`;
- `experiments/results/pa_gmp_apa200_{selection,residuals,test}/`;
- `experiments/results/spline_memory_{dpa200,apa200}/`.

## 14. Benchmark conclusion

Causal GMP is the current forward PA quality point under 1000 counted real
MUL/sample. It materially improves APA and slightly improves DPA, but fails
the possible −50 dB target and does not sufficiently isolate DPD residual from
evaluator error. The next justified work remains in PA identification/external
capture validation, not further surrogate-specific DPD optimization.
