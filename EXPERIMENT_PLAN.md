# Experiment plan and execution protocol

Дата фиксации: 2026-07-28. Цель — получить воспроизводимое сравнение
complex-spline/structured DPD с OpenDPD на общей measurement/surrogate
protocol, не приписывая surrogate-only результат физическому PA.

## 1. Frozen provenance

| Item | Frozen value |
|---|---|
| OpenDPD source | `7426bbf8a47624b59bd7f045a86641b403023f3c` |
| Egor source | `8e8127cfbea4b2d67cc3d944514b4835e4c7e947` |
| chaotic source | `f4ebc3e7c302e83d2eb1c44244f5ecd6e2d884ce` |
| host | Linux, Intel i5-12450H, 12 logical CPUs, 15 GiB RAM, no NVIDIA runtime |
| baseline Python | 3.14.6 |
| baseline NumPy | 2.5.1 |
| optional audit env | `.venv`, NumPy 2.5.1, SciPy 1.18.0, pandas 3.0.5, scikit-learn 1.8.0 |

The built-in split files are used verbatim: train/validation/test = 60/20/20.
No row-wise normalization is applied. The gain is computed from train only and
reported in every result:

1. `complex_ls`: \(g=\sum x^*y/\sum|x|^2\);
2. `opendpd_peak`: \(\max|y|/\max|x|\), retained only as a compatibility
   protocol.

Alignment is estimated on calibration data only. Integer delay is frozen before
validation/test; fractional-delay correlation is diagnostic until a validated
resampling filter is selected. Test is opened once after model/configuration
freeze.

## 2. Metric contract

Primary quality metric:

\[
\mathrm{NMSE}_{\rm pooled,dB}
 =10\log_{10}\frac{\sum_n|\hat y_n-y_n|^2}
                       {\sum_n|y_n|^2}.
\]

The project requirement “error < \(10^{-5}\)” is recorded in all reports as an
unresolved definition until the owner chooses one of:

- ordinary MSE (scale dependent);
- normalized error power (equivalent to pooled NMSE < −50 dB);
- relative RMS amplitude error (a different threshold).

Compatibility metrics are kept separately:

- OpenDPD segment-wise dB NMSE;
- OpenDPD spectral-bin “EVM”;
- OpenDPD strongest-inband-subchannel ACLR left/right/average;
- time-domain RMS sample EVM;
- conventional integrated-main-band ACLR;
- PAPR, maximum predistorted amplitude, clipping/saturation.

All PSD parameters are explicit. For DPA_200MHz:
`fs=800e6`, `nperseg=2560`, 10 × 20 MHz subchannels. For APA_200MHz:
`fs=983.04e6`, `nperseg=19662`, 5 × 40 MHz subchannels.

## 3. Exact commands

### 3.1 Unit/evaluator tests

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest discover -s tests -v
```

Expected current result: 28 tests, under 0.1 s on the audit host.

### 3.2 NumPy spline sweep

The committed configs are
[`experiments/configs/spline_dpa200.json`](experiments/configs/spline_dpa200.json)
and
[`experiments/configs/spline_apa200.json`](experiments/configs/spline_apa200.json).
They enumerate \(7\times4\times10\times1=280\) candidates:

```bash
.venv/bin/python -m baseline.train_spline \
  --config experiments/configs/spline_dpa200.json --overwrite
.venv/bin/python -m baseline.train_spline \
  --config experiments/configs/spline_apa200.json --overwrite
```

Training reads only train/validation. It writes `validation_trials.json` after
each completed fit, then writes the selected NPZ/report. `selection_metric=auto`
uses the explicitly fitted train-only memory-polynomial surrogate for
validation cascade selection; without that flag it falls back to inverse
postdistorter diagnostic selection.

### 3.3 Frozen test evaluation

```bash
.venv/bin/python -m baseline.evaluate_spline \
  --dataset vendor/OpenDPD/datasets/DPA_200MHz \
  --training-report experiments/results/spline_dpa200_surrogate/training_report.json \
  --pa-surrogate experiments/results/spline_dpa200_surrogate/pa_surrogate.npz \
  --mode both \
  --output-json experiments/results/spline_dpa200_surrogate/test_evaluation.json \
  --output-npz experiments/results/spline_dpa200_surrogate/test_waveforms.npz
```

Use the analogous APA paths for `APA_200MHz`. The evaluator asserts the
deployment direction in its JSON: `test_input -> spline -> supplied PA
surrogate`; measured `test_output` is only used for the separate inverse
diagnostic and surrogate-fidelity comparison.

### 3.4 OpenDPD controls

The repository contains no checkpoints in this checkout, and the current host
has no GPU. Therefore the full neural matrix is a planned, not locally
reproduced, command:

```bash
cd vendor/OpenDPD
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -v -m "not extended"
bash benchmark/reproduce_benchmark_report.sh --device 0
```

The bundled evidence records 16,369 s for one RTX PRO 6000 run. This is not
extrapolated to three seeds; each seed must have its own timing/log.

### 3.5 Egor reproduction

The exact static source can be imported without changing upstream:

```bash
cd vendor/DPD_for_PA/data1
PYTHONPATH="$PROJECT_ROOT/vendor/chaotic_library/src:$PROJECT_ROOT" \
  "$PROJECT_ROOT/.venv/bin/python" reproduce_egor.py
```

`reproduce_egor.py` is a future convenience wrapper; the executed audit command
is recorded in `BENCHMARK_REPORT.md` and used the notebook parameters
(`R_PA=800`, `R_DPD=600`, seeds 42/43/100/101, FAN=8). It reports PA-only,
circular, and correct-direction surrogate scores separately.

## 4. Ablations

### Memoryless spline (completed first)

- \(K\in\{8,12,16,24,32,48,64\}\);
- knot placement: uniform amplitude, uniform power placement, quantile,
  compression-aware;
- ridge \(\lambda\in\{0,10^{-10},\ldots,10^{-2}\}\);
- second-difference smoothness \(\mu=0\), then a separate validation-only
  \(\mu\) sweep around the selected ridge;
- complex64 export versus complex128 solve.

### Memory additions (only after residual evidence)

1. `(m,d)=(0,0)`;
2. `(0,0),(1,0)`;
3. `(0,0),(1,0),(2,0)`;
4. sparse delays selected by greedy forward validation, group OMP or group
   LASSO;
5. spline → FIR (SPH) with \(L=2,4,8\);
6. state-conditioned spline only if a slow residual correlation is observed.

Every branch has an incremental operation/storage row and must beat a frozen
validation gate before being retained.

### Fixed point

- FP32 reference;
- FP16-like component storage (explicitly not a vendor-specific FP16 claim);
- signed 16-bit and signed 12-bit input/control points;
- accumulator widths and saturation;
- quantized knot/address interpolation;
- full-vs-chunked streaming equivalence.

## 5. Acceptance gates

A claim of “better than OpenDPD” requires all of:

1. same dataset and PA checkpoint/physical session;
2. same train/validation/test boundaries and normalization;
3. same delay/gain/alignment and PSD/ACLR procedure;
4. no test use for tuning, feature/knot/checkpoint selection;
5. at least three seeds for stochastic methods;
6. pooled complex NMSE, EVM, ACLR L/R/average, AM/AM, AM/PM, PAPR and peak
   predistorted amplitude;
7. real MUL/ADD/nonlinear/control/memory counts, not parameter count;
8. measured calibration wall-clock and inference latency/throughput;
9. fixed-point degradation with declared formats/accumulator/saturation;
10. physical PA verification, or an explicit `surrogate-only` label.

The project target of `<1000 real multiplications/sample` is counted with
`complex multiply = 4M+2A`. A model that meets quality but misses this gate is
not a qualifying method; a cheaper but lower-quality model remains a valid
Pareto point.

## 6. Expected resources

The NumPy sweep is sequential and fits within the current 15 GiB host; measured
280-fit runs were seconds to tens of seconds. The dense OpenDPD neural matrix
requires CUDA and several GPU-hours per recorded matrix. Physical PA work also
requires RF instrumentation, a calibrated feedback path and a fresh session;
no physical measurement is claimed in this workspace.
