# Независимый аудит OpenDPD `main`

Дата аудита: 2026-07-28. Репозиторий: <https://github.com/lab-emi/OpenDPD>.  
Проверенный commit: `7426bbf8a47624b59bd7f045a86641b403023f3c` (`2026-07-26T23:31:13+02:00`, `Rebuild the benchmark with reproducible PA and DPD results (#19)`).  
Локальный путь: `vendor/OpenDPD`. Worktree после аудита чистый.

Этот документ — аудит только OpenDPD. Числа, помеченные как «опубликованный benchmark», взяты из bundled machine evidence и **не были повторно получены на текущей машине**. Числа, помеченные как «измерение статьи», относятся к физическому PA из статьи и не смешиваются с surrogate-результатами репозитория.

## 1. Краткий итог

Подтверждено по коду:

1. Neural DPD в OpenDPD обучается в корректном направлении DLA: на вход DPD подаётся желаемый `x`, затем выполняется `DPD(x) -> frozen PA surrogate`, а target равен `g x`. Экспорт также применяет DPD к `X_test`, а не к известному `y_test`. Круговой тест вида `y_test/g -> inverse -> PA -> y_test` для neural DLA **не используется** (`project.py:201-215`, `models.py:172-185`, `steps/run_dpd.py:26-28,79-99`).
2. MP/GMP benchmark действительно обучает postdistorter по ILA `Phi(y_train/g)c ≈ x_train`, но при validation/test переносит коэффициенты в predistorter и подаёт ему `X_val/X_test`. Направление test корректно (`benchmark/benchmark_volterra.py:1-19,899-940`).
3. Все опубликованные DPD-числа текущего repository benchmark вычислены через learned PA surrogate, а не через новое физическое измерение. Репозиторий сам это оговаривает (`benchmark/benchmark_report.md:15-19,114-123`).
4. Stateful backbones фактически сбрасывают recurrent state на каждом train frame, validation/test segment и chunked plot call. Нет warm-up mask и streaming-state API (`models.py:156-169`, `modules/train_funcs.py:60-91,100-135`, `project.py:317-328`).
5. TRes-GRU и TRes-DeltaGRU не являются causal zero-look-ahead моделями в текущей форме: residual Conv1d использует ±16 samples, а `torch.roll(...,-1)` использует следующий sample и замыкает последний sample на первый (`backbones/tres_gru.py:30-47,71-85`, `backbones/tres_deltagru.py:39-56,102-120`).
6. Temporal sparsity не уменьшает фактическое число умножений в included eager или Triton kernels: обе реализации выполняют dense matrix-vector products после зануления delta. `HW_PARAM` — аналитический proxy, не runtime counter (`backbones/tres_deltagru.py:122-143,230-260`; `backbones/triton_deltagru.py:102-127,153-158`).
7. 999 stored parameters TRes-DeltaGRU-H15 не означают `<1000 real multiplications/sample`. Уже dense weight products дают 999 real multiplications/sample, а GRU gates и feature extraction поднимают строгую нижнюю границу до примерно 1048, ещё до Hardswish.
8. Quantization — partial fake-quantization, а не bit-accurate fixed-point inference. Conv1d/Hardswish и часть feature path остаются floating point; catch-all fallback может молча вернуть float model (`quant/__init__.py:21-38`, `quant/quant_envs.py:145-171,195-213,286-305`).
9. Метрики OpenDPD нестандартны: NMSE — среднее segment-wise dB, EVM — spectral-bin MAE, ACLR — adjacent subchannel относительно самого мощного in-band subchannel. Их нельзя без оговорок сравнивать с pooled NMSE, demodulated RMS EVM и стандартным total-channel ACLR (`utils/metrics.py:42-187`).
10. Текущий benchmark существенно улучшил provenance, однако в git нет ни одного checkpoint/raw log/raw polynomial result/source snapshot, на которые ссылается JSON. Для повторения требуется полное переобучение.

Главные блокеры для заявления «лучше OpenDPD»:

- смешение измеренного no-DPD output и surrogate DPD output в стандартном comparison plot;
- отсутствие общего physical-PA evaluator для кандидатов;
- state/boundary/look-ahead mismatch между train, logged test, plot и exported streaming use;
- неполная fixed-point модель;
- отсутствие operation/latency/throughput/calibration-time benchmark;
- один seed в опубликованном repository benchmark;
- nonstandard metric definitions и отсутствие alignment/gain calibration в evaluator.

## 2. Что просмотрено

Полностью/адресно просмотрены:

- `README.md`, `datasets/README.md`, `pyproject.toml`, CI/weekly workflows;
- `main.py`, `project.py`, `models.py`, `arguments.py`;
- `steps/train_pa.py`, `steps/train_dpd.py`, `steps/run_dpd.py`, `steps/plot.py`;
- `modules/data_collector.py`, `train_funcs.py`, `loggers.py`, `paths.py`;
- `utils/metrics.py`, `util.py`, `plotting.py`;
- все `backbones/*.py`, особо GRU/LSTM/DGRU/TCN/TRes-GRU/TRes-DeltaGRU/GMP;
- `quant/**`;
- `benchmark/benchmark_volterra.py`, `collect_benchmark_report.py`, `reproduce_benchmark_report.sh`, `benchmark_report.md`, `results/benchmark_report_results.json`;
- built-in datasets, `spec.json`, CSV hashes, split lengths и numerical diagnostics;
- `tests/**`, особенно backbone, E2E, benchmark и quantization smoke tests;
- bundled papers `papers/OpenDPDv2.pdf`, `OpenDPDv1.pdf`, `DeltaDPD_IMS2025.pdf`, `TCN-DPD_IMS2025.pdf`, `MP-DPD_IMS2024.pdf`.

В repository нет `.ipynb`; README ссылается на внешний Colab (`README.md:70-71`). В repository также нет `.pt/.pth/.ckpt/.onnx/.npz`, RTL, C implementation или Gem5 configuration/source.

## 3. Фактически выполненные проверки и среда

Ключевые команды:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch main --single-branch --depth 1 \
  --filter=blob:none https://github.com/lab-emi/OpenDPD.git vendor/OpenDPD

git -C vendor/OpenDPD status --short --branch
git -C vendor/OpenDPD show -s --format='%H%n%aI%n%s' HEAD
find vendor/OpenDPD -type f
pdftotext -layout vendor/OpenDPD/papers/OpenDPDv2.pdf /tmp/OpenDPDv2.txt
```

Дополнительно выполнены read-only NumPy/Python checks:

- AST parse всех 92 Python files: 0 syntax errors;
- row counts всех split CSV;
- SHA-256 verification executable sources и benchmark datasets;
- complex least-squares gain и peak-gain для train;
- normalized complex cross-correlation для integer delay `[-64,64]`;
- parabolic diagnostic вокруг correlation peak;
- exact equality APA A/B input arrays;
- exact hashes complete non-overlapping `nperseg` frames между train/val/test;
- DC/IQ descriptive diagnostics;
- inventory checkpoint/notebook/hardware files.

Среда текущей машины:

- Python `3.14.6`;
- NumPy `2.5.1`;
- Intel i5-12450H, 8 cores / 12 logical CPUs;
- GPU/NVIDIA runtime отсутствует;
- `torch`, `scipy`, `pandas`, `pytest` dependencies OpenDPD не установлены.

Поэтому training, pytest, metric execution и inference timing **не запускались**. Установка зависимостей или изменение upstream-кода в рамках этого этапа не выполнялись.

## 4. Dataset, split, normalization, alignment

### 4.1 Фактические splits

Loader split-CSV просто читает шесть готовых файлов без normalization/alignment (`modules/data_collector.py:68-77`). Для single-CSV используется последовательный split 60/20/20 (`modules/data_collector.py:100-140`).

| Dataset | Train | Validation | Test | `nperseg` | Metadata |
|---|---:|---:|---:|---:|---|
| APA_200MHz | 58,980 | 19,662 | 19,662 | 19,662 | 983.04 Msps, 200 MHz, 5 subchannels, 256QAM |
| APA_200MHz_b | 58,980 | 19,662 | 19,662 | 19,662 | те же signal parameters |
| DPA_160MHz | 294,912 | 98,304 | 98,304 | 16,384 | 640 Msps, 160 MHz, 4 subchannels, 1024QAM |
| DPA_200MHz | 23,040 | 7,680 | 7,680 | 2,560 | 800 Msps, 200 MHz, 10 subchannels, 64QAM |

Это 60/20/20, что совпадает со статьёй (`papers/OpenDPDv2.pdf`, p. 5, Table I) и `spec.json`, но не с README-фразой «8:2:2» (`README.md:158-162`).

Split files уже сохранены отдельно, поэтому `split_ratios` для built-in split-CSV — metadata, не исполняемая логика. Для custom single-CSV границы последовательные; нет guard interval, grouping by waveform/capture или leakage detector.

APA metadata нельзя считать однозначным описанием physical waveform:

- paper и top-level FAQ называют сигнал `5×40 MHz` (`papers/OpenDPDv2.pdf`, p. 5; `README.md:308-315`);
- `datasets/APA_200MHz/spec.json:2` и `datasets/README.md:187-204` называют carriers `5×20 MHz` с 40 MHz spacing;
- `spec.json:10-14` задаёт composite band 200 MHz, `bw_sub_ch=40 MHz`, `n_sub_ch=5`;
- FAQ утверждает, что `spec.json` настроен как «single 200 MHz channel» (`README.md:317-321`), хотя current file задаёт пять subchannels.

Следовательно `spec.json` следует воспринимать как current evaluator configuration, а не бесспорную RF waveform truth. Exact adjacent-band masks должны быть заданы отдельно от этой metadata.

### 4.2 Leakage checks

- Exact hash complete non-overlapping `nperseg` blocks не выявил повторяющихся blocks между train/val/test ни в одном из четырёх datasets.
- Это не доказывает отсутствие повторяющихся OFDM symbols/subsequences или generator-level leakage; такого validator в repository нет.
- Все три input splits `APA_200MHz` и `APA_200MHz_b` побитно идентичны друг другу. Поэтому A→B проверяет другой PA measurement/operating condition на **том же excitation waveform**, а не waveform generalization. Это согласуется с описанием `datasets/README.md:206-209`, но должно быть явно отражено в robustness claims.
- Train framing создаёт сильно перекрывающиеся samples внутри train split при stride 1 (`modules/data_collector.py:233-268`). Это не train/test leakage, но effective number of independent training examples намного меньше числа frames.

### 4.3 Normalization и target gain

Код не нормализует CSV во время загрузки. Масштаб уже зашит в файлах.

DPD target gain:

```text
g_peak = max_n |y_train[n]| / max_n |x_train[n]|
```

реализован в `utils/util.py:26-33`. Это real positive peak scaling, а не complex LS alignment

```text
g_LS = sum x*[n] y[n] / sum |x[n]|^2.
```

Диагностика:

| Dataset | `g_peak` | `|g_LS|` | `angle(g_LS)`, rad |
|---|---:|---:|---:|
| APA_200MHz | 1.000000 | 1.162571 | -0.003038 |
| APA_200MHz_b | 1.000000 | 1.152146 | -0.001226 |
| DPA_160MHz | 2.295543 | 2.902335 | ≈0 |
| DPA_200MHz | 2.520809 | 3.165638 | ≈0 |

`g_peak` может быть осознанной safe-output/backoff convention, но это **не** gain/phase alignment. На этих captures phase mismatch мал, однако amplitude target заметно отличается от LS gain. Для apples-to-apples нужно сохранить OpenDPD `g_peak` как один protocol и отдельно публиковать physically motivated target gain; нельзя незаметно заменить его и приписать улучшение модели.

### 4.4 Time/fractional alignment и feedback path

В source нет integer delay search, fractional-delay filter, complex gain alignment, DC correction, IQ imbalance correction или feedback frequency-response equalization. Документация лишь утверждает, что CSV «time-aligned sample-by-sample» (`datasets/README.md:43-45`).

Read-only complex-correlation diagnostic дал integer delay `0` на train/val/test всех datasets, с `|rho|≈0.993–0.995`. Parabolic peak indication в samples:

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| APA_200MHz | +0.080 | +0.081 | +0.081 |
| APA_200MHz_b | -0.005 | -0.004 | -0.004 |
| DPA_160MHz | -0.031 | -0.033 | -0.030 |
| DPA_200MHz | -0.008 | -0.011 | ≈0.000 |

Это только diagnostic correlation-lobe interpolation, не независимая оценка fractional delay: PA memory/frequency response и нелинейность могут смещать peak. Integer alignment выглядит правдоподобно; fractional alignment остаётся неподтверждённым.

APA train DC magnitude составляет примерно `0.5%` RMS и для input, и для output; код его не удаляет. Это не обязательно measurement defect, но evaluator не способен отделить intentional waveform DC от observation-path DC.

## 5. Framing, state и real-time semantics

### 5.1 Train/eval mismatch

- Train всегда использует `IQFrameDataset` с sliding frames (`project.py:220-230`).
- Validation/test всегда используют independent `IQSegmentDataset` (`project.py:221-237`).
- CLI flag `--use_segments` объявлен (`arguments.py:25-29`), но `Project.build_dataloaders()` его не использует.
- `IQSegmentDataset` zero-pads последний segment без validity mask (`modules/data_collector.py:203-230`). Основные built-in val/test lengths кратны `nperseg`, поэтому там padding не срабатывает. Для arbitrary/custom data padded zeros попадут в loss/NMSE/EVM.
- В APA train длина на 6 samples меньше трёх `nperseg`; это не влияет на текущий frame training, но повлияло бы при segmented training.

### 5.2 Hidden-state reset

`CoreModel.forward()` создаёт zero state, когда state не передан, и возвращает только output (`models.py:156-169`). Train frames shuffled и вызываются независимо (`modules/train_funcs.py:60-91`), validation/test segments также независимо (`modules/train_funcs.py:100-135`). Warm-up samples не исключаются из loss/metrics.

Для TRes-DeltaGRU все пять states (`x_p, h, h_p, dm_nh, dm`) обнуляются, если не передан хотя бы один (`backbones/tres_deltagru.py:214-229,306-350`), а top-level TRes forward вообще не возвращает их.

Следствия:

- reported metrics относятся к reset-per-frame/segment semantics;
- нет честно проверенной continuous streaming semantics;
- начало каждого frame обучается с artificial zero history;
- `run_dpd.py` подаёт весь `X_test` одним sequence (`steps/run_dpd.py:79-99`), поэтому exported waveform имеет другие state/boundary semantics, чем logged test;
- `steps/plot.py:80-85` также выполняет full-test forward, затем лишь reshapes output для metrics;
- full-sequence epoch plots режут данные arbitrary chunks по 16,384 и сбрасывают state (`project.py:317-360`).

### 5.3 Future context

TRes residual Conv1d имеет kernel 3, dilation 16, symmetric padding 16: output `n` зависит от `n-16,n,n+16`. Feature `last_step = roll(x,-1)` добавляет `x[n+1]`, причём последний sample получает первый sample того же supplied sequence (`backbones/tres_gru.py:30-47,71-85`; аналогично TRes-DeltaGRU).

Итого нужен минимум 16-sample look-ahead/buffering, а boundary policy не streaming-safe. При 983.04 Msps это около 16.3 ns sample look-ahead, но latency/throughput на hardware не измерены.

TCN ещё более noncausal: четыре symmetric depthwise layers с dilation 1,2,4,8 и kernel 5 дают ±30 samples receptive context (`backbones/tcn.py:14-31,82-96`). Статья TCN-DPD прямо называет architecture noncausal (`papers/TCN-DPD_IMS2025.pdf`, pp. 1–2).

## 6. PA model и DPD learning

### 6.1 PA surrogate

PA behavioral model обучается supervised `x -> y` с raw `nn.MSELoss` (`steps/train_pa.py:17-49,83-99`; `project.py:262-270`). Best checkpoint выбирается по validation NMSE (`steps/train_pa.py:86-99`, `modules/loggers.py:165-179`).

PA checkpoint binding в обычном DPD path основан на architecture-derived filename (`steps/train_dpd.py:27-42`). Model ID не включает optimizer, learning rate, batch, epochs и другие recipe fields; source теперь предупреждает о collision, но сохраняет прежнюю схему (`modules/paths.py:111-135`). В `modules/paths.py:93` условие `elif args.step == 'train_dpd' or 'run_dpd'` логически всегда true для любой ветви кроме `train_pa`; текущий CLI случайно не выводит это за допустимый scope.

### 6.2 Neural DLA

Pipeline корректен по направлению:

```text
x_desired -> neural DPD -> frozen differentiable PA surrogate -> y_hat
target = g_peak * x_desired
loss = MSE(y_hat, target)
```

Evidence:

- target replacement: `project.py:210-215`;
- PA checkpoint loading and freezing: `steps/train_dpd.py:27-42,63-70`;
- cascade order: `models.py:172-185`;
- export input is `X_test`: `steps/run_dpd.py:26-28,79-99`.

Поэтому гипотеза о circular inverse reconstruction не относится к OpenDPD neural DLA.

Однако DLA может exploit ошибки frozen surrogate. Особенно показательно, что surrogate test NMSE порядка `-39 dB`, тогда как simulated DPD cascade в некоторых случаях достигает `-48…-54 dB` относительно идеального target. Это не логическое противоречие, но такая точность не переносится автоматически на физический PA.

### 6.3 MP/GMP ILA

Benchmark MP/GMP использует правильную complex formulation:

- PA model: `Phi(x)c ≈ y`;
- postdistorter: `Phi(y/g)c ≈ x`;
- predistorter validation/test: `Phi(X_val/test)c`, затем common PA surrogate.

См. `benchmark/benchmark_volterra.py:50-136,266-282,348-519,899-940`.

Плюсы:

- complex coefficients;
- column L2 scaling;
- explicit zero-fill delay boundaries;
- segmented boundary policy;
- GMP PA truncated-SVD path с reported rank.

Ограничения:

- ILA MP/GMP использует CUDA `gels` без ridge и без numerical-rank estimate;
- coefficient transfer postdistorter→predistorter не гарантирует оптимальность при сильной nonlinearity/memory;
- все final DPD metrics снова через learned PA;
- train/validation/test boundary reset отличается от continuous PA;
- inference operation count отсутствует.

### 6.4 Смешение physical и simulated paths

Стандартный plot/evaluator берёт:

- without-DPD: измеренный `y_test`;
- with-DPD: `net_pa(net_dpd(X_test))`, то есть surrogate.

Это видно в `steps/plot.py:52-54,80-85,95-133` и в training plots `steps/train_dpd.py:96-112`.

Такое сравнение не является общей physical A/B measurement и не является общей surrogate simulation. Оно может быть полезно визуально, но improvement delta методологически нечестна.

`run_dpd.py` экспортирует predistorted CSV для bench test, но repository не содержит end-to-end ingestion, re-alignment, gain/power matching и physical evaluation результата.

## 7. Метрики

### 7.1 NMSE и требование `10^-5`

OpenDPD:

```text
r_s = mean_n |e_s[n]|^2 / mean_n |target_s[n]|^2
NMSE_repo = mean_s 10 log10(r_s)
```

`utils/metrics.py:42-52`.

Это dB геометрического среднего segment ratios, не

```text
10 log10(sum_{s,n}|e|^2 / sum_{s,n}|target|^2).
```

Если `10^-5` означает normalized error power ratio, то `10 log10(10^-5) = -50 dB`. Но необходимо определить:

- pooled ratio;
- per-segment maximum/mean;
- arithmetic mean ratio;
- или repo mean-of-dB.

Они не эквивалентны при разных segment powers.

Training loss — raw PyTorch MSE по двум real channels (`project.py:262-270`):

```text
MSE_torch = (1/(2N)) sum_n (e_I^2 + e_Q^2)
           = 0.5 * complex_MSE.
```

Поэтому `loss < 1e-5` не означает `NMSE < -50 dB` без target power и точного aggregation rule. В OpenDPD нет acceptance check `error < 1e-5`.

### 7.2 EVM

`utils/metrics.py:55-108`:

- FFT каждого segment;
- деление main band на `n_sub_ch`;
- mean absolute complex spectrum error в каждом subchannel;
- normalization на mean absolute reference spectrum;
- average subchannels/segments;
- `20 log10`.

Это не RMS symbol EVM, не comparison с known QAM reference grid, и нет standard channel equalization/common phase error procedure. README прямо признаёт EVM inaccuracy (`README.md:306-325`). Статья, в свою очередь, использует physical «input vs digitized output» EVM (`papers/OpenDPDv2.pdf`, p. 5, Table I), что также не идентично repository spectral EVM.

### 7.3 ACLR/ACPR

`utils/metrics.py:111-151`:

- PSD: SciPy Welch;
- main channel делится на `n_sub_ch`;
- denominator — maximum power одного in-band subchannel;
- left/right integration width — один derived subchannel;
- reported avg — arithmetic mean dB left/right.

Это ближе к per-carrier ACPR, чем к total-main-channel ACLR. Поле `bw_sub_ch` не используется; ширина выводится из `bw_main_ch/n_sub_ch`.

Welch вызывает `scipy.signal.welch` только с `nperseg`, `return_onesided=False`, `scaling='spectrum'` (`utils/metrics.py:154-187`). Значит window=`hann`, overlap=50%, detrend=`constant`, nfft=`nperseg` берутся неявными SciPy defaults и должны быть зафиксированы для межреализационного сравнения.

### 7.4 PSD, AM/AM, AM/PM и отсутствующие metrics

- Model-comparison PSD нормирует каждую curve на её собственный maximum (`utils/plotting.py:221-241`), что скрывает absolute-power shift.
- Model AM/AM нормирует каждую output curve на её собственный max (`utils/plotting.py:252-288`), что может скрыть gain/compression differences.
- AM/PM строится, но нет binned confidence/statistics (`utils/plotting.py:291-322`).
- Input waveform PAPR записан в benchmark evidence (`benchmark/collect_benchmark_report.py:1564-1619`), но predistorted PAPR, max `|z|`, clipping margin, output power, stability sweep, coefficient norm и OOD amplitude sweep не входят в стандартный DPD report.

## 8. Backbones

### 8.1 Реально доступные

Dispatch находится в `models.py:26-147`: GMP, GRU, DGRU, QGRU variants, LSTM, VDLSTM, RVTDCNN, APNRRU, BOJANET, DeltaGRU, DeltaJANET, PGJANET, DVRJANET, TRes-DeltaGRU, TRes-GRU, TCN, MCLDNN.

CLI и dispatch расходятся:

- CLI предлагает `janet`, `fcn`, `mamba`, `pntdnn`, `pdgru`, `pnjanet`, `djanet`, `snn`, но dispatch их не реализует (`arguments.py:57-69`);
- dispatch ожидает `apnrru`, CLI предлагает `apnrnn`;
- `neuraltx` branch импортирует отсутствующий file;
- tests сами документируют missing CLI implementations (`tests/test_backbones.py:15-17`, `tests/test_e2e_pipeline.py:115-134`);
- current weekly train list не включает TRes-GRU/TRes-DeltaGRU, хотя README утверждает training every supported backbone (`README.md:241-250`).

### 8.2 GRU/LSTM

GRU и LSTM — real-valued recurrent layers с final 2-output FC (`backbones/gru.py`, `backbones/lstm.py`). Они:

- causal внутри sequence;
- не phase-equivariant by construction;
- не имеют streaming-state return;
- reset state на каждом outer call.

### 8.3 DGRU/Delta variants

DGRU/DeltaGRU/DeltaJANET/TCN формируют `I,Q,|x|,|x|^3,sin(phi),cos(phi)`. В нескольких implementations `sin=Q/|x|`, `cos=I/|x|` без epsilon (`backbones/dgru.py:63-78`, `backbones/deltagru.py:60-73`, `backbones/tcn.py:82-96`). Exact zero даёт `0/0`; backbone tests специально используют nonzero random input (`tests/test_backbones.py:55-69`).

Это особенно важно после input/fixed-point quantization, когда exact zeros становятся обычными.

### 8.4 TCN

TCN использует noncausal symmetric dilated depthwise convolutions и residual I/Q (`backbones/tcn.py:14-31,82-96`). `count_flops()` (`backbones/tcn.py:33-80`) смешивает multiplications, divisions, sqrt, activations и additions в один scalar, не считает dot-product additions последовательно и не является общим operation counter.

### 8.5 TRes-GRU / TRes-DeltaGRU

При `H=15`:

```text
GRU x->h weights     = (3H)*6  = 270
GRU h->h weights     = (3H)*H  = 675
FC weights           = 2H      = 30
TCN Conv(2,3,3)      = 18
TCN Conv(3,2,1)      = 6
stored weights total           = 999
```

Это совпадает с current benchmark и `papers/OpenDPDv2.pdf`, p. 5, Table II. README сообщает 996 (`README.md:80-85`) — это ошибка относительно кода/статьи.

TRes feature path не mathematically phase-equivariant: raw I/Q и future I/Q входят в arbitrary real GRU, а residual real convolutions свободно смешивают components. Phase rotation equivariance не гарантирована.

### 8.6 GMP и MP

Есть две разные реализации:

1. `backbones/gmp.py:5-51` — production CLI backbone:
   - hardcoded memory 11/degree 5, потому что `models.py:26-28` не передаёт CLI `K`/`gmp_memory_length`;
   - 495 **real** weights, применяемых к complex features;
   - один real coefficient не может выразить arbitrary AM/PM complex coefficient;
   - Python loop по каждому sample;
   - large repeated intermediate tensors;
   - это не benchmark GMP/MP.
2. `benchmark/benchmark_volterra.py` — корректные complex MP/GMP LS baselines:
   - explicit complex coefficients;
   - MP и aligned/lagging/leading GMP;
   - segmented boundary reset;
   - GPU LS/SVD.

Отдельного production `MP` backbone нет.

## 9. Operation count

Принята конвенция:

- real multiply и real add считаются отдельно;
- one complex multiply = 4 real multiplies + 2 real adds;
- также приведён lower-bound вариант 3-multiply;
- sigmoid/tanh/sqrt/Hardswish/comparison/lookup считаются отдельными nonlinear/control operations;
- fused MAC не скрывает два арифметических действия, если сравнение задано в real multiplications/sample.

### 9.1 GRU-H16 benchmark DPD

Stored parameters:

```text
W_ih: 3H*2 = 96
W_hh: 3H*H = 768
b_ih+b_hh = 6H = 96
FC weight+bias = 2H+2 = 34
total = 994
```

Per sample:

```text
matrix/FC real multiplications = 96 + 768 + 32 = 896
GRU elementwise multiplications = 3H = 48
total real multiplications = 944
```

При explicit non-FMA bias/add convention:

```text
real additions ≈ 3H^2 + 13H = 976
nonlinear scalar ops = 2H sigmoid + H tanh = 48
```

Таким образом GRU-H16 формально укладывается в `<1000 real multiplications/sample`, но current API ещё не демонстрирует continuous state/latency/fixed-point behavior.

### 9.2 TRes-GRU/TRes-DeltaGRU-H15

```text
dense weight multiplications = 999
GRU elementwise gate/state multiplications = 45
I^2 and Q^2 feature multiplications = 2
absolute lower bound before |x|^3/Hardswish = 1046
|x|^3 as two ordinary multiplies -> 1048
```

Пять scalar Hardswish outputs/sample добавляют примерно 10 multiplications при прямом `x*clip((x+3)/6)`, то есть типичная оценка около 1058 real multiplications/sample, плюс sqrt, comparisons/clamps и additions.

Для Delta version добавляются:

- `delta_x`, `delta_h`;
- abs;
- threshold comparisons;
- masked updates;
- five state-vector updates;
- accumulator memory traffic.

При nonzero threshold theoretical skip-aware hardware может пропускать соответствующие weight columns. Но included eager path вызывает dense `nn.Linear` (`backbones/tres_deltagru.py:250-260`), а Triton умножает полные `wx*dx` и `wh*dh` (`backbones/triton_deltagru.py:153-158`). Поэтому реальный software multiplication count остаётся dense.

Paper Table II сообщает `1324 FLOPs`, но source не содержит воспроизводимого TRes FLOP counter или однозначной FLOP/MAC convention. Нельзя использовать 999 active/stored parameters как operation count.

### 9.3 Benchmark MP/GMP, 500 complex coefficients

Один только coefficient dot product:

```text
4-multiply complex convention: 500 * 4 = 2000 real multiplications/sample
3-multiply convention:         500 * 3 = 1500 real multiplications/sample
```

К этому добавляются basis powers, envelopes, complex-by-real products и accumulation. Следовательно MP/GMP с 1000 reported «real parameters» заведомо не проходит `<1000 real multiplications/sample`.

Для complex64 500 coefficients занимают 4000 bytes — почти столько же, сколько 999 FP32 neural weights (3996 bytes), но runtime radically различается.

### 9.4 Что repository не считает

Нет общего счётчика:

- real mult/add;
- nonlinear ops;
- comparisons/lookups;
- memory reads/writes;
- state/context storage;
- latency/throughput;
- dense-vs-skip-aware execution.

Только TCN/MCLDNN имеют локальные mixed FLOP estimates, а benchmark сравнивает stored parameters (`benchmark/benchmark_report.md:114-123`).

## 10. Temporal sparsity

TRes-DeltaGRU хранит thresholded deltas, но:

- current published repository benchmark использует `THX=THH=0`, то есть threshold pruning отключён (`benchmark/benchmark_report.md:102-110`);
- `get_temporal_sparsity()` вычисляет `HW_PARAM = fixed + recurrent_weights*(1-zero_fraction)` (`backbones/tres_deltagru.py:122-143`);
- это «active parameter-equivalents» для column-skipping hardware model;
- actual eager/Triton kernels column skipping не реализуют;
- sparse indices, encoding cost, irregular memory access и threshold overhead не включены в `HW_PARAM`.

Статья рассматривает специальный C/Gem5 path и потенциальный ASIC, но C source, binary, Gem5 config и raw traces в repository отсутствуют. Поэтому paper energy results нельзя воспроизвести из main checkout.

## 11. Quantization/fixed point

### 11.1 Что реализовано

`INT_Quantizer` округляет scale к power of two, clamp/round и сразу dequantizes обратно в float (`quant/qmodules/quantizers.py:56-81`). Это QAT/fake quantization.

Replacer поддерживает:

- `nn.Linear`;
- `nn.Conv2d`;
- sigmoid/tanh и custom Add/Mul;
- sqrt/pow зарегистрированы, но по умолчанию используют identity quantizers.

Не поддерживается `nn.Conv1d` (`quant/quant_envs.py:145-156`), поэтому TRes residual convolution остаётся FP. Hardswish тоже не заменяется. Direct PyTorch feature `pow/sqrt/cat/roll` TRes model остаётся FP. `INT_Linear` output quantizer всегда 16-bit независимо от requested activation bits и включается только для last layer/eval (`quant/qmodules/quant_layers.py:48-82`).

Нет:

- integer accumulator widths;
- rounding mode per accumulator;
- saturation/overflow propagation;
- input ADC quantization;
- coefficient export format;
- bit-true streaming kernel;
- fixed-point latency/throughput;
- RTL/HLS/FPGA synthesis.

### 11.2 Silent float fallback

`get_quant_model()` ловит любое exception и возвращает исходный float model (`quant/__init__.py:21-38`). Run по имени «quantized» может поэтому завершиться и сохранить float checkpoint.

CI smoke особенно слаб:

- сначала обучается default GRU с 2-input GRU;
- затем test просит `qgru`, у которого 4-input feature GRU (`tests/test_e2e_pipeline.py:56-77`, `backbones/qgru.py:9-32,59-70`);
- conversion заменяет `nn.GRU` на structurally different `PYGRU` (`quant/quant_envs.py:114-130,215-248`);
- strict load прежнего GRU state dict обязан дать missing/unexpected keys и input-shape mismatch;
- exception подавляется, возвращается fresh float QGRU;
- test проверяет лишь появление quant-labelled checkpoint, а не наличие quantized modules.

Следовательно README claim о W16A16 E2E smoke не является proof bit-accurate или даже fake-quant execution (`README.md:241-250`).

Для default `OpenDPDv2.sh` TRes-DeltaGRU pretrained structure совместимее и recurrent Linear/Add/Mul могут fake-quantize, но Conv1d/Hardswish/features всё равно остаются float (`bash_scripts/OpenDPDv2.sh:21-60,88-160`).

## 12. Reproducibility artifacts

Плюсы current main:

- fixed reproduction matrix и commands;
- source/data/checkpoint hashes в machine evidence;
- exact environment и GPU;
- validation-selected checkpoints;
- explicit limitation section;
- benchmark code separates primary common-surrogate ranking от PA-surrogate sensitivity.

Проверка hashes:

- 61/62 `source_file_sha256` entries совпали current main;
- единственный mismatch — generated `benchmark/benchmark_report.md` itself;
- все executable/config source files в manifest совпали;
- все 6 CSV hashes APA_200MHz и 6 CSV hashes DPA_160MHz совпали JSON.

Это делает table provenance существенно сильнее обычного README claim.

Но:

- report был запущен на dirty `benchmark-fix` commit `3df35e…`, не на current `main`; exact hashes частично компенсируют это (`benchmark/benchmark_report.md:125-138`);
- referenced 27 artifact paths (checkpoints, logs, raw polynomial JSON, source archive) отсутствуют в checkout;
- один seed (`benchmark/benchmark_report.md:116-119`);
- `re_level=soft`, cuDNN benchmark enabled (`project.py:115-129`, `benchmark/reproduce_benchmark_report.sh:525-550`);
- test metrics вычисляются каждый epoch по default (`arguments.py:20-24`, `project.py:393-417`): automatic selection остаётся validation-only, но human tuning получает repeated test exposure;
- recipe fields не входят в standard model ID;
- нет per-job runtime в published JSON; есть только total 16,369 s для всей publication run;
- no inference timing, memory traffic или calibration-time table.

Published run environment:

- Python 3.13.14;
- PyTorch 2.13.0+cu132;
- CUDA 13.2;
- NVIDIA RTX PRO 6000 Blackwell;
- total job duration 16,369 s ≈ 4 h 32 min 49 s;
- 300 epochs, batch 64, frame 200/stride 1, seed 0 (`benchmark/reproduce_benchmark_report.sh:525-550`).

## 13. Результаты: не смешивать две системы оценки

### 13.1 Current repository benchmark — surrogate simulation

Test numbers из `benchmark/benchmark_report.md:21-79`. Все DPD строки ниже оцениваются через learned PA surrogate.

| Dataset | DPD | Method / PA surrogate | Stored real degrees | NMSE dB | EVM-repo dB | ACLR L/R/avg dB |
|---|---|---|---:|---:|---:|---:|
| APA | MP | ILA / TRes-GRU-H27 | 1000 | -42.19 | -48.15 | -46.07 / -44.30 / -45.19 |
| APA | GMP | ILA / TRes-GRU-H27 | 1000 | -38.53 | -46.35 | -44.07 / -43.12 / -43.59 |
| APA | GRU-H16 | DLA / TRes-GRU-H27 | 994 | -45.13 | -47.43 | -51.06 / -50.97 / -51.01 |
| APA | TRes-GRU-H15 | DLA / TRes-GRU-H27 | 999 | -44.29 | -45.10 | -53.47 / -53.51 / -53.49 |
| APA | TRes-DeltaGRU-H15 | sensitivity / TRes-DeltaGRU-H27 | 999 | -47.97 | -49.70 | avg -54.40 |
| DPA160 | MP | ILA / TRes-GRU-H27 | 1000 | -43.19 | -50.50 | -49.44 / -52.43 / -50.94 |
| DPA160 | GMP | ILA / TRes-GRU-H27 | 1000 | -43.97 | -50.19 | -52.62 / -53.61 / -53.11 |
| DPA160 | GRU-H16 | DLA / TRes-GRU-H27 | 994 | -49.76 | -53.98 | -57.68 / -56.85 / -57.26 |
| DPA160 | TRes-GRU-H15 | DLA / TRes-GRU-H27 | 999 | -53.19 | -57.85 | -60.27 / -60.47 / -60.37 |
| DPA160 | TRes-DeltaGRU-H15 | sensitivity / TRes-DeltaGRU-H27 | 999 | -54.37 | -59.40 | avg -61.66 |

TRes-Delta rows не являются частью primary four-model same-surrogate ranking; это отдельные independently trained sensitivity runs (`benchmark/benchmark_report.md:42-49,72-79`).

PA surrogate test NMSE:

- APA best TRes-DeltaGRU-H27: `-39.18 dB`;
- DPA160 best TRes-DeltaGRU-H27: `-39.74 dB`.

Это важный surrogate-fidelity ceiling/risk marker, но не прямой mathematical ceiling для target-tracking внутри surrogate.

### 13.2 OpenDPDv2 paper — physical PA measurement

`papers/OpenDPDv2.pdf` соответствует arXiv v2, 16 Dec 2025: <https://arxiv.org/abs/2507.06849>.

Setup (`p. 5`, Tables I–II):

- R&S SMW200A + FSW43;
- 3.5 GHz Ampleon GaN Doherty;
- 41.2 dBm average output;
- TM3.1a 5×40 MHz, 200 MHz, 256QAM;
- 983.04 Msps;
- 98,304 samples, 60/20/20;
- 240 epochs, AdamW, RTX 4090;
- best ACPR checkpoint.

Physical result:

| Model | Params | Paper FLOPs | NMSE dB | EVM dB | ACPR dBc |
|---|---:|---:|---:|---:|---:|
| No DPD | — | — | -20.5 | -24.7 | -28.3 |
| TRes-GRU | 999 | 1282 | -38.4 | -41.2 | -59.0 |
| TRes-DeltaGRU | 999 | 1324 | -39.6 | -42.1 | -59.9 |

Это не те же metric implementation/recipe/PA path, что repository benchmark, и paper не даёт separate left/right ACPR.

Paper quantization/sparsity Table IV (`p. 6`) содержит measured results, например:

- dense FP32 TRes-Delta-999: EVM -42.1, ACPR -59.9;
- dense W16A16: -41.2 / -58.8;
- dense W12A12: -37.3 / -54.5;
- 56% sparse, 450 active, W16A16: -39.3 / -53.2;
- 56% sparse, 450 active, W12A12: -35.2 / -51.8.

Но main checkout не содержит raw captures, corresponding checkpoints, C/Gem5 implementation или bit-true verifier. Это published measurement evidence, не locally reproducible artifact.

README headline `996 params, -59.4 dBc, -42.1 dB` (`README.md:80-85`) расходится и со статьёй (`999, -59.9, -42.1`), и с repository surrogate benchmark (`999, -54.4 avg` для sensitivity path).

## 14. Что пока не подтверждено

- Physical PA performance current main checkpoints: checkpoints/captures отсутствуют.
- Fractional alignment accuracy исходных captures.
- Feedback-path frequency response, IQ imbalance, LO leakage и calibration.
- Continuous streaming quality recurrent/TRes models.
- Latency/throughput на CPU/GPU/FPGA/ASIC.
- Реальный speedup temporal sparsity included software kernels.
- Bit-true W16/W12 behavior, accumulator widths и saturation.
- Robustness по seeds, power levels, waveform types и PA drift.
- Calibration time per model.
- Whether paper’s exact 1324-FLOP convention includes all arithmetic/nonlinear operations.
- Generalization APA↔APA_b beyond same-waveform operating-condition change.

## 15. Минимальный apples-to-apples protocol перед новой моделью

1. Заморозить commit, CSV hashes, PA checkpoint hash и split manifest.
2. Разделить два protocols:
   - `surrogate_protocol`: все candidates через один frozen PA;
   - `physical_protocol`: все exported `z` измеряются на одном PA session с power/gain control.
3. Не строить improvement между measured no-DPD и simulated with-DPD.
4. Зафиксировать два gain protocols: OpenDPD `g_peak` и complex-LS/desired RF gain; не смешивать результаты.
5. Добавить integer + fractional alignment только по train/feedback calibration; затем apply frozen transform к val/test.
6. Сделать pooled NMSE основным, repo segment-dB NMSE сохранить как compatibility metric.
7. Публиковать repository spectral EVM и true demodulated RMS EVM раздельно.
8. Зафиксировать Welch window/overlap/nfft/detrend/scaling и exact adjacent bands.
9. Stateful evaluation:
   - continuous stream;
   - explicit initial reset;
   - warm-up exclusion;
   - deterministic context overlap;
   - same boundary policy для всех models.
10. Запретить future wrap. Для noncausal моделей явно считать look-ahead и latency.
11. Записывать `PAPR(z)`, max/percentiles `|z|`, output power, clipping/saturation count.
12. Operation counter разделяет mult/add/nonlinear/comparison/lookup/memory access; sparse count принимается только при реально skip-aware kernel.
13. Test выполнять один раз после frozen model/threshold selection; минимум 3 seeds neural methods.
14. Fixed point: bit-true integer reference, coefficient/input formats, accumulator width, rounding, saturation, degradation.

## 16. Предлагаемые команды следующего этапа

Сначала isolated environment и tests, затем evaluator; benchmark training не запускать до фиксации protocol.

```bash
cd vendor/OpenDPD
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -v -m "not extended"
```

Проверка exact benchmark matrix без training:

```bash
bash benchmark/reproduce_benchmark_report.sh --dry-run \
  --output-dir /tmp/opendpd-benchmark-dry-run
```

После evaluator fixes — сначала один deterministic baseline:

```bash
python main.py \
  --dataset_name APA_200MHz \
  --step train_pa \
  --PA_backbone tres_gru \
  --PA_hidden_size 27 \
  --frame_length 200 \
  --frame_stride 1 \
  --n_epochs 300 \
  --seed 0 \
  --re_level hard \
  --eval_val 1 \
  --eval_test 0 \
  --accelerator cuda
```

Затем neural DPD с frozen checkpoint и test только в отдельной final command. Current CLI потребует небольшого evaluator/API изменения, потому что `eval_test=0` лишает standard logger test, а standalone evaluator с exact selected checkpoint отсутствует.

Полный published matrix:

```bash
bash benchmark/reproduce_benchmark_report.sh --device 0
```

Его recorded total — около 4.55 GPU-hours на RTX PRO 6000 Blackwell. Per-job durations не опубликованы, поэтому время на другой GPU или 3 seeds честно оценить по repository нельзя. CPU-only current host для полного neural matrix непрактичен без предварительного smoke timing.

## 17. Какие изменения делать первыми

До добавления spline model:

1. новый read-only evaluator с explicit direction `x_desired -> DPD -> PA`, common PA binding и separate physical/surrogate labels;
2. alignment module + tests complex gain/integer/fractional delay;
3. pooled + compatibility metrics и frozen spectral config;
4. stateful streaming interface, warm-up mask, no-wrap future context;
5. predistorted amplitude/PAPR/stability metrics;
6. exact operation schema и reference counters;
7. quantization assertion: hard fail вместо silent float fallback, module coverage test, bit-true integer reference;
8. deterministic split/hash/checkpoint manifest и test-once command.

Только после этого memoryless complex spline имеет смысл сравнивать с:

- no DPD;
- complex MP/GMP ILA;
- GRU-H16;
- TRes-GRU-H15;
- TRes-DeltaGRU-H15;

на одном PA checkpoint и одной metric implementation. Если spline дешевле, но хуже по pooled NMSE/ACLR, результат должен быть представлен как Pareto point, а не как «победа по параметрам».
