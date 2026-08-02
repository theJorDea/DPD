# Benchmark report

Дата среза: 2026-08-03.

Requirement correction: the 1000-real-MUL-equivalent time budget applies only
to deployment DPD, not to PA models in this report. The completed strict
`<1000 MUL` PA searches remain valid historical experiments, but their cost is
not a Huawei PA-model acceptance gate. Final DPD quality is expected to use
harmonic/spur attenuation; its exact bands/reference/threshold are pending.

## 1. Scope

Этот report содержит только реально выполненные local runs и отдельно
маркирует upstream/legacy evidence. Большинство завершённых local runs
относятся к вспомогательному forward PA evaluator; конечная цель проекта —
low-latency DPD, проверенный через независимый frozen evaluator или physical PA:

```text
measured x -> frozen PA model -> y_hat -> compare with measured y
```

Воспроизводимый validation-only spline-memory DPD demo и его
16/14/12-bit integer replay выполнены в правильном направлении
`desired x -> frozen DPD -> frozen legacy PA surrogate`. Новый cascade
через независимый GMP/OpenDPD evaluator, physical PA experiment и timed
DPD hardware implementation не выполнялись. DPA/APA GMP и APA sparse PA
arithmetic также проверены при 16/14/12 bit, но это
evaluator-numerics evidence, не DPD timing.
APA standalone SPH и первый non-factorized sparse
forward run были недостаточно точными для замены GMP; bounded lag-9 sparse run
впервые прошёл cheap-Pareto gate относительно MP, но всё ещё уступил GMP.
Дополнительно выполнен preregistered capture transfer на `APA_200MHz_b`,
включая frozen held-out release и независимое воспроизведение четырёх test
metrics. Поэтому report содержит external-capture evidence, но не утверждает
better-than-OpenDPD или результат на физическом PA после DPD.

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
| Historical PA search bound | strict `<1000` real MUL/complex sample; not the DPD timing gate |

Exact commands/configs/hashes: `EXPERIMENT_PLAN.md` and result execution
records. Selection and residual stages read train/validation only. Source GMP
tests were opened once per dataset after release-gate PASS. B-capture release
has an explicitly recorded two-access audit: the first process failed after
loading test but before inference/metric; the unchanged frozen release then
completed on the second access.

## 3. Datasets

| Dataset | Train / validation / test | Fs | Frame | Declared waveform |
|---|---:|---:|---:|---|
| DPA_200MHz | 23,040 / 7,680 / 7,680 | 800 MHz | 2,560 | 10×20 MHz LTE, 64-QAM |
| APA_200MHz | 58,980 / 19,662 / 19,662 | 983.04 MHz | 19,662 | 5-carrier TM3.1a, 256-QAM |
| APA_200MHz_b | 58,980 / 19,662 / 19,662 | 983.04 MHz | 19,662 | same declared family, “measurement B”; change axes unknown |

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

Relative-error values are retained only as optional numerical diagnostics. If
`10^-5` is reported as normalized error power, it corresponds mathematically
to `−50 dB`, but this is not a Huawei acceptance target and does not select the
PA evaluator or DPD architecture.

### 5.2 Selected GMP topology

| Dataset | Topology | Ridge/solver | Complex coeff. | Required history |
|---|---|---|---:|---:|
| DPA | `ka7/la24`, `kb4/lb24/mb1`, causal `kc4/lc24/mc1` | 1e−5 / column-scaled complex ridge `lstsq` | 356 | 24 samples |
| APA | `ka7/la30`, `kb2/lb30/mb2`, causal `kc2/lc30/mc2` | 1e−7 / column-scaled complex ridge `lstsq` | 444 | 31 samples |

Both models are causal, zero-lookahead and full rank in the final fit.

### 5.3 APA standalone SPH candidate

The next isolated PA family was a phase-equivariant spline-Hammerstein model:

```text
v[n] = x[n] * C(|x[n]|)
y_hat[n] = v[n] + sum(l=1..7) h[l] * v[n-l], h[0] = 1
```

The staged search was frozen before validation load and never accessed or
hashed APA test. The selected hard-valid recipe was
`amplitude_uniform_K32_L8_cr1e-08_sm1e-08_fr0e+00`.

| Candidate | Full OOF NMSE | Common OOF NMSE | MUL / ADD | Coefficients / state | Decision |
|---|---:|---:|---:|---:|---|
| Matched MP | −37.054329 dB | −37.099951 dB | 960 / 628 | 300 / 58 | reference |
| Matched GMP | −38.345410 dB | −38.750526 dB | 954 / 947 | 888 / 236 | reference |
| SPH `K=32,L=8` | −30.402374 dB | −30.437014 dB | **37 / 36** | **78 / 14** | rejected |

SPH is `6.651955/6.662937 dB` worse than MP (full/common) and
`7.943037/8.313512 dB` worse than GMP. It fails the preregistered ≤3 dB
cheap-Pareto loss gate despite satisfying the historical PA-search arithmetic
bound. The
immutable bundle is
`experiments/results/pa_sph_apa200_selection/`; its execution record reports
620.531 s before publication, 60 unique recipes and 180 completed OOF fits.

The residual is structurally useful: proper causal correlation peaks at lags
22–24 (`0.684–0.723`) on train OOF and repeats on reused validation, while
instantaneous envelope correlation is small. This points to delay-dependent
nonlinear branches, not more knots or an unproven slow state.

### 5.4 APA non-factorized sparse spline-memory candidate

The next preregistered forward family used
`y_hat[n] = sum_b x[n-m_b] C_b(|x[n-d_b]|)` with joint complex coefficients,
local two-point spline support, causal delays and explicit frame resets. The
runner executed S0 topology screening, S1 knot sweep and S2 ridge sweep using
train leave-one-frame-out OOF only. It performed 16 stage associations, 14
unique recipe evaluations and 42 OOF fits; validation was loaded after the
full-train model freeze and the test split was never opened.

| Candidate | Full OOF NMSE | Common OOF NMSE | Full loss vs GMP | MUL / ADD | Coefficients / state | Decision |
|---|---:|---:|---:|---:|---:|---|
| Matched MP | −37.054329 dB | −37.099951 dB | +1.291081 dB | 960 / 628 | 300 / 58 | reference |
| Matched GMP | −38.345410 dB | −38.750526 dB | 0 dB | 954 / 947 | 888 / 236 | reference |
| Sparse `K=12`, 6 branches | −32.030011 dB | −32.088250 dB | **+6.315399 dB** | **54 / 58** | **144 / 48** | rejected |

Full-train refit scored `−32.049190 dB`; reused validation scored
`−32.048219 dB` (OpenDPD-compatible `−32.048219 dB`). The selected family was
`mixed_diagonal_long` with branches `(0,0),(1,1),(2,2),(22,22),(23,23),(24,24)`.
Hard identifiability gates passed (rank `72/72`, condition `78.57`, minimum
feature support `8`), but the ≤3 dB cheap-Pareto gate versus MP failed by a
wide margin. The decision is `neither_evaluator_nor_cheap_pareto`.

Residual analysis selected the next bounded hypothesis: the strongest causal
proper correlation was lag 9 (`0.69064` train OOF, `0.69131` validation), while
radial-envelope correlation peaked only around `0.140`. The lag-9 family was
preregistered before its fit; validation remained descriptive and was not a
selection input. Parent bundle:
`experiments/results/pa_sparse_spline_memory_apa200_selection/`.

### 5.5 APA residual-guided lag-9 sparse candidate

The follow-up config
`experiments/configs/pa_sparse_spline_memory_lag9_apa200.json` was committed
before fitting. It froze nine topology families around the observed lag-9
residual, `K={8,12,16}`, four ridge values and a maximum of 66 OOF fit calls.
The selected recipe is
`parent_plus_signal_lag8_10_current_envelope` with branches
`(8,0),(9,0),(10,0)` added to the six-branch parent, `K=12`, ridge `1e-8`.

| Candidate / split | Full NMSE | Common NMSE | Gain vs parent | Loss vs MP | Loss vs GMP | MUL / ADD | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Lag-9 sparse, train OOF | −37.792478 dB | −37.852832 dB | **+5.762467 / +5.764583 dB** | **−0.738150 / −0.752881 dB** | +0.552932 / +0.897694 dB | **72 / 82** | cheap-Pareto only |
| Lag-9 sparse, full-train refit | −37.866643 dB | −37.927296 dB | — | — | — | 72 / 82 | frozen model |
| Lag-9 sparse, reused validation | −37.860728 dB | −37.898605 dB | — | — | — | 72 / 82 | descriptive |

The incremental gate passed in all three OOF folds; minimum gains were
`+5.717845 dB` full and `+5.731338 dB` common. The selected model has rank
`108/108`, augmented condition `2427.39`, maximum coefficient `1.15093`,
216 real coefficient values, 48 state values, exact streaming/reset
equivalence and a `62.5693 s` pre-publication runtime. Its normalized full-OOF
error power is about `1.66e-4`; this is a secondary diagnostic, not a Huawei
pass/fail result.

Two envelope-only topology families with repeated signal delay `(0,d)` were
rank-deficient because spline partition of unity duplicates the same `x[n]`
linear component; a `K=16` candidate also failed rank in OOF folds. These are
recorded as hard-invalid trials, not silently regularized into the ranking.
The immutable result is
`experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/`.

### 5.6 APA capture transfer and limited coefficient calibration

The preregistered pre-test config
`experiments/configs/pa_transfer_apa200_to_b.json` verified byte-identical
source/target inputs for train and validation and different measured outputs.
Source topology and coefficients were frozen for zero-shot evaluation; target
adaptation changed coefficients only. Validation selected `N=16384` for both
families before the separate test release was authorized.

| Model / mode | N per frame | Target validation full NMSE | Common NMSE | Fit s | MUL / ADD | FP32 model+state |
|---|---:|---:|---:|---:|---:|---:|
| causal GMP zero-shot | 0 | −23.794841 | −23.793859 | 0 | 954 / 947 | 4532 B |
| lag-9 sparse zero-shot | 0 | −23.701383 | −23.703027 | 0 | 72 / 82 | 1148 B |
| causal GMP coefficient-only | 1024 | −36.884942 | −36.927799 | 0.148 | 954 / 947 | 4532 B |
| causal GMP coefficient-only | 16384 | **−37.890764** | **−37.961563** | 6.860 | 954 / 947 | 4532 B |
| lag-9 sparse coefficient-only | 1024 | −29.470637 | −29.478619 | 0.060 | 72 / 82 | 1148 B |
| lag-9 sparse coefficient-only | 16384 | **−35.358475** | **−35.446027** | 1.513 | 72 / 82 | 1148 B |

The full preregistered validation curves, including infeasible GMP `N=64/128` and
negative sparse short-prefix points, are in
`experiments/results/pa_transfer_apa200_to_b_pretest/transfer_manifest.json`.
The source/target nuisance diagnostic estimated delay 0 and
`|complex-LS gain|=1.152146` from target train only; it did not alter strict
scores or select a model. The independent verifier reports 20 reproduced
metric records and verifies that test was still sealed at the pre-test stage.

This is capture-transfer evidence, not known operating-point adaptation.
Zero-shot source coefficients fail on B, while coefficient-only calibration
recovers much of GMP fidelity. Sparse calibration is about 4.5× faster and
13.25× cheaper in real MUL than GMP at `N=16384`, but remains 2.53 dB worse.

The frozen held-out release then used target `x_test` only as the PA-model
input and compared predictions with measured target `y_test`:

| Model / mode | Target test full NMSE | Common NMSE | Relative error power | Fit s | Host batch inference s |
|---|---:|---:|---:|---:|---:|
| causal GMP zero-shot | −23.795441 | −23.800907 | 4.1731e−3 | 0 | 0.03616 |
| causal GMP coefficient-only, N=16384 | **−37.895152** | **−38.003839** | **1.6236e−4** | 6.860 | 0.03637 |
| lag-9 sparse zero-shot | −23.695838 | −23.700933 | 4.2699e−3 | 0 | 0.00620 |
| lag-9 sparse coefficient-only, N=16384 | −34.801474 | −35.437986 | 3.3102e−4 | 1.513 | 0.00547 |

GMP validation→test changed by only `−0.0044 dB`. Sparse changed by
`+0.5570 dB` on full-record score but only `+0.0080 dB` on common support;
its visible full-score loss is therefore mainly a 24-sample causal reset
boundary effect. Primary full-record values are retained.

The release required two accesses, not one. On the first access an incorrect
train-length guard failed after loading target test and before any model
inference or metric. Only that guard was corrected; topology, coefficients,
selected `N` and metric protocol were unchanged. The incident and final
verification explicitly record `strict_single_open_execution=false`,
`test_used_for_selection=false` and `test_used_for_coefficient_fit=false`.

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

### 6.1 APA short widely-linear residual ablation

Four causal corrections \(\sum_d b_d x^*[n-d]\) were fitted two-stage over
the GMP coefficient-OOF folds. Validation was an already-viewed descriptive
split; test was not accessed.

| Support | MUL/ADD | OOF full/common gain | Minimum fold full/common | Eligible |
|---|---:|---:|---:|---|
| `{0}` | 958/951 | 0.02677/0.02978 dB | 0.02408/0.02673 dB | no |
| `{0,1}` | 962/955 | **0.02735/0.03055 dB** | 0.02391/0.02690 dB | no |
| `{0,1,2}` | 966/959 | 0.02684/0.02990 dB | 0.02450/0.02702 dB | no |
| `{0,1,2,3,4}` | 974/967 | 0.02479/0.02956 dB | 0.01840/0.02688 dB | no |

All fits were full rank and passed exact reset/streaming checks. No support
met the preregistered 0.1 dB full/common threshold, so `no_correction`
remained selected at 954 MUL / 947 ADD. This is a negative post-discovery
internal-resampling result, not physical IQ attribution.

### 6.2 APA proper-complex long-memory FIR ablation

The stable proper residual-correlation peak around causal lag 45 was tested
with nested sparse supports. Validation remained descriptive/reused and test
was not accessed.

| Support | MUL/ADD/state | OOF full/common gain | Minimum fold full/common | Eligible |
|---|---:|---:|---:|---|
| `{45}` | 958/951/268 | 0.01332/0.01469 dB | 0.00834/0.00935 dB | no |
| `{44,45,46}` | 966/959/270 | **0.01818/0.02007 dB** | 0.01142/0.01265 dB | no |
| `{43,…,48}` | 978/971/274 | 0.01775/0.01995 dB | 0.01155/0.01441 dB | no |
| `{42,…,49}` | 986/979/276 | 0.01769/0.02013 dB | 0.01143/0.01435 dB | no |

Every fold improved and all fits were full rank with exact reset/streaming
checks, but no support reached the preregistered 0.1 dB full/common gate.
Therefore `no_correction` remained selected at 954 MUL / 947 ADD / 236 state
reals. A visible normalized correlation did not imply useful explained error
power.

## 7. Complexity and memory

| Model | MUL | ADD | Nonlinear | Reads/writes | Real coeff. | State reals | FP32 coeff+const+state |
|---|---:|---:|---:|---:|---:|---:|---:|
| MP DPA | 792 | 502 | 0 | 288/2 | 240 | 46 | 1,260 B |
| GMP DPA | 766 | 759 | 1 | 1,092/8 | 712 | 188 | 3,636 B |
| MP APA | 960 | 628 | 30 | 360/2 | 300 | 58 | 1,572 B |
| GMP APA | 954 | 947 | 1 | 1,362/8 | 888 | 236 | 4,532 B |
| APA SPH `K=32,L=8` | **37** | **36** | 1 sqrt | 36/2 | **78 + 63 constants** | **14** | **620 B** |
| APA sparse non-factorized `K=12` | **54** | **58** | 6 sqrt | 36/2 | **144 + 23 constants** | **48** | **860 B** |
| APA lag-9 sparse non-factorized `K=12` | **72** | **82** | 6 sqrt | 54/2 | **216 + 23 constants** | **48** | **1,148 B** |

GMP is the quality winner but does not dominate memory traffic/storage. These
are analytical factorized schedules, not FPGA resource measurements.

## 8. Timing

| Task | DPA | APA | Meaning |
|---|---:|---:|---|
| Formal GMP selection, 154 fits | 70.988 s | 212.762 s | full process wall |
| Final selected fit | 1.234 s | 5.555 s | train coefficient solve |
| OOF/residual process | 10.259 s | 24.872 s | train OOF + validation diagnostics |
| Widely-linear residual audit | — | 14.805 s | selected `no_correction`; OOF fit 13.224 s |
| Proper long-FIR residual audit | — | 25.473 s | selected `no_correction`; OOF fit 23.395 s |
| APA SPH four-stage selection | — | 620.531 s | train OOF search + atomic publication |
| APA sparse staged selection | — | 33.589 s | train OOF search + frozen refit + reused validation |
| APA lag-9 sparse staged selection | — | **62.569 s** | train OOF search + frozen refit + reused validation |
| Frozen-test process | 0.066 s | 0.31 s | no fit; process wall measurement differs by method |
| Test predictor single batch | 8.673 ms | 30.286 ms | NumPy batch diagnostic |
| Test batch throughput | 0.885 Msample/s | 0.649 Msample/s | host software, not real-time target |

Peak RSS was not measured. The host throughput is far below 800/983.04
MSample/s and does not represent a factorized FPGA implementation.

## 9. Reproducible DPD surrogate demo

### 9.1 One-command frozen validation replay

Коммиты `a385635` и `b982266` добавили и затем усилили одну
sealed точку запуска. `DPD_DEMO_OUTPUT` должен указывать на ещё не
существующий каталог; runner намеренно не перезаписывает артефакты.

```bash
DPD_DEMO_OUTPUT=/tmp/dpd-surrogate-demo-manual
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.run_surrogate_demo --output-root "$DPD_DEMO_OUTPUT"
```

Команда не обучает и не выбирает модель. Она проверяет хеши
конфигов, кода, waveform inputs и frozen artifacts, запускает float и
16/14/12-bit paths для DPA и APA, а затем публикует `summary.json` и
`completion_manifest.json`. Путь оценки жёстко зафиксирован как
`desired validation x -> frozen DPD -> frozen PA surrogate`; measured PA output
и test split не открываются.

| Dataset | Float NMSE no DPD -> DPD | Configured adjacent relative improvement L/R | Main-power change | Peak / PAPR |
|---|---:|---:|---:|---:|
| DPA_200MHz validation | -20.3381 -> -30.5324 dB | +4.7494 / +7.7372 dB | -0.0515 dB | 1.1926 / 10.4690 dB |
| APA_200MHz validation | -19.9688 -> -32.3800 dB | +16.4797 / +13.8639 dB | -0.0414 dB | 1.0615 / 10.5835 dB |

Пример 12-bit preservation: DPA cascade NMSE `-30.5148 dB`,
fixed-vs-float drive NMSE `-54.8714 dB` и configured absolute suppression
left/right `4.7819/7.6921 dB`; APA соответственно `-32.3790 dB`,
`-53.5984 dB` и `16.3691/13.9821 dB`. Все шесть integer rows имеют
нулевые saturation/knot-collision counters, exact configured
chunked-streaming equivalence и bit-exact 90-degree rotation на проверенных
signals. Precision не выбиралась.

Аналитический float operation vector на complex sample: `21 MUL, 24 ADD,
0 DIV, 1 magnitude nonlinear, 6 LUT, 18 reads, 2 writes, 4 state reals`;
comparisons `5/3` и stored coefficient reals `144/48` для DPA/APA.
Integer reference: `20 MUL, 25 ADD, 1 DIV, 1 integer sqrt, 8 LUT, 28 reads,
2 writes, 4 state reals`; comparisons также `5/3`. Эти vectors не
заменяют target timing.

Целевой sealing suite:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest tests.test_run_surrogate_demo -v
```

На текущем environment все **10/10 tests passed**, включая end-to-end
run, hash/tolerance contracts, symlink/no-follow defense, exact набор из 12 child
completion manifests, output-identity/no-overwrite guards и отсутствие final
manifest у незавершённого run.

### 9.2 Evidence boundary

Validation уже участвовала в историческом выборе floating model. Поэтом demo
— это воспроизводимая `validation_replay_surrogate_only` демонстрация, а
не untouched final evidence. Она не доказывает physical-PA linearization,
customer-defined RF harmonic/spur attenuation, превосходство над
OpenDPD, Huawei acceptance, target FPGA/ASIC latency/resources/power или
достаточность 12 bit. Gate A->B остаётся закрыт.

### 9.3 Historical legacy-test replay: surrogate only

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
| Selected source PA 16/14/12 bit | completed for DPA/APA GMP and APA lag-9 sparse; train/validation only |
| Target-calibrated APA-B PA 16/14/12 bit | completed for hash-bound GMP/sparse payloads; train/validation only, no target test access |
| Selected spline-memory DPD 16/14/12 bit | completed on reused validation; frozen legacy surrogate only, no precision selected |
| Integer DPD → frozen legacy PA-surrogate cascade | completed for DPA/APA; zero saturation/collision and exact streaming on evaluated signals |
| Integer DPD → independent GMP/OpenDPD evaluator or physical PA | not run |
| APA_200MHz_b capture transfer/adaptation | held-out release completed; access count 2 with first metric-free failure; metadata axes unknown |
| FPGA/ASIC synthesis and latency/throughput | not run |
| Physical predistorted PA measurement | not available |

Details: `HARDWARE_COST.md` and `ROBUSTNESS_AND_ADAPTATION.md`.

## 12. Negative and failed experiments

- A1 fractional alignment failed preregistered improvement gate; A0 retained.
- DPA GMP gives only 0.286 dB test gain despite larger coefficient/state
  storage.
- Neither GMP reaches the optional −50 dB normalized-error reference; this is
  not a Huawei requirement and does not decide the spectral DPD task.
- PA evaluator margin remains below 10 dB; DPD stage remains blocked.
- Slow-state candidate lacks independent-capture evidence.
- APA short conjugate residual family failed its 0.1 dB OOF gain threshold;
  the best observed support improved only 0.027/0.031 dB full/common.
- APA proper long-FIR family also failed that threshold; the best support
  improved only 0.018/0.020 dB despite positive gains in every fold.
- APA SPH met the arithmetic budget but failed quality: 37 MUL/sample and
  −30.402 dB OOF, 6.652 dB worse than matched MP. K48/K64 raw-score variants
  were rejected for rank deficiency, so the factorized family is closed.
- APA non-factorized sparse spline-memory met the arithmetic budget at 54
  MUL/sample, improved SPH by 1.628 dB, but remained 5.024 dB worse than MP
  and 6.315 dB worse than GMP; its evaluator gate also failed.
- APA lag-9 sparse met the incremental and cheap-Pareto gates at 72 MUL/sample:
  it improved the parent by 5.762/5.765 dB and beat MP by 0.738/0.753 dB,
  but remained 0.553/0.898 dB behind GMP, so its evaluator gate and Gate A→B
  both remained closed.
- APA capture transfer exposed a large zero-shot gap: GMP −23.795 dB and
  lag-9 sparse −23.701 dB on target validation. Coefficient-only calibration
  recovered GMP to −37.891 dB and sparse to −35.358 dB at 16,384 samples/frame;
  this is not a power/thermal claim because “measurement B” metadata is absent.
- Held-out target confirmed calibrated GMP at −37.895 dB, but sparse reached
  only −34.801 dB full-record. Both remain above the optional −50 dB
  normalized-error reference; the actual Huawei decision must use the
  configured spectral/spur protocol, and sparse full-record sensitivity to
  startup/reset is materially larger.
- Target fixed-point PA arithmetic has no saturation, but 12-bit GMP loses
  2.718 dB versus float validation; sparse loses only 0.0935 dB while retaining
  lower absolute fidelity. Absence of overflow is therefore not sufficient
  for selecting a word length.
- Target release was not a strict single-open execution: the first access
  failed before inference/metric due to a frame-length guard bug. The second
  access used unchanged frozen decisions; the incident is retained rather
  than hidden.
- Repeated `(0,d)` envelope-only branches were hard-invalid for rank deficiency;
  this identifiability failure is a documented model constraint, not a
  post-fit numerical workaround.
- Bundled OpenDPD checkpoints/GPU are unavailable, but a sealed CPU
  train/validation runner has completed bounded GRU/TRes-GRU/TRes-DeltaGRU
  preflight. Full validation-quality training remains pending.
- Existing Egor circular score does not establish deployment DPD. The DPD
  reservoir pair alone requires about 728,622 dense real MUL/sample before
  additions, activations and memory traffic, so it is implausible under the
  reference-time budget; definitive pass/fail still requires same-target
  total streaming-time measurement.
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
- `experiments/results/pa_widely_linear_residual_apa200/`;
- `experiments/results/pa_long_fir_residual_apa200/`;
- `experiments/results/pa_sph_apa200_selection/`;
- `experiments/results/pa_sparse_spline_memory_apa200_selection/`;
- `experiments/results/pa_sparse_spline_memory_lag9_apa200_selection/`;
- `experiments/results/pa_transfer_apa200_to_b_pretest/`;
- `experiments/results/pa_transfer_apa200_to_b_test_release/`;
- `experiments/results/pa_transfer_apa200_to_b_test_release_verification.json`;
- `experiments/results/pa_fixed_point_{dpa200,apa200,apa200_b}/`;
- `experiments/results/spline_memory_{dpa200,apa200}/`.

## 14. Benchmark conclusion

Causal GMP remains the current forward PA fidelity point among the completed
historically `<1000 MUL/sample` searches. The lag-9 sparse PA establishes a
reproducible 72-MUL
cheap-Pareto point and removes the discovered residual peak on the source
capture, but it still trails GMP and cannot move the DPD contour. The
`APA_200MHz_b` held-out release confirms that zero-shot transfer is poor and
that coefficient-only calibration restores most GMP quality. Sparse remains
a much cheaper but lower-fidelity point. Physical PA remeasurement and
controlled power/bias/temperature metadata are still pending. GMP and lag-9
sparse both remain above the optional −50 dB normalized-error reference, and neither supplies an independent
DPD evaluator with sufficient margin. Source/target PA fixed-point coverage is
complete, but it is not the deployment DPD timing result.
