# Обзор литературы по Digital Predistortion: качество, сложность и аппаратная реализуемость

Дата проверки источников: 2026-07-28. Это рабочий research-note для последующей интеграции в общий отчёт. Он основан на первичных источниках: статьях авторов, arXiv, DOI-записях, официальных репозиториях и открытых авторских рукописях. Если величина не приведена в доступном первоисточнике, ниже стоит `NR` (not reported); значения не оцифровывались на глаз с графиков.

## 1. Как читать сравнения

Результаты DPD нельзя ранжировать между разными PA, уровнями мощности, полосами, сигналами, feedback-path и процедурами измерения. Особенно нельзя считать, что более низкий NMSE на differentiable PA surrogate лучше менее низкого NMSE, измеренного на физическом PA.

В этом обзоре использованы отдельные группы сопоставимости:

1. **APA_200MHz**: OpenDPDv2 и только его baselines на одном физическом 3.5 GHz GaN Doherty PA.
2. **DPA_200MHz**: TCN-DPD и его baselines на одном frozen DGRU PA surrogate.
3. **SparseDPD-20MHz**: 20 MHz capture, OpenDPD surrogate evaluation и отдельная FPGA post-implementation simulation.
4. **Spline E1/E2/E3**: три собственных физических стенда статьи о spline-interpolated LUT; сравнивать методы можно только внутри каждого E1, E2 или E3.
5. **FR3 feature-selection**: два 15 GHz DUT из одной работы; физическое DPD-измерение таблицей дано для DUT1.
6. **Прочие hardware reports**: только контекст для throughput/resources; это не единая рейтинговая таблица.

Принятая базовая метрика:

\[
\mathrm{NMSE}_{dB}=10\log_{10}
\frac{\sum_n |\hat y[n]-y_{\mathrm{ref}}[n]|^2}
     {\sum_n |y_{\mathrm{ref}}[n]|^2}.
\]

Если требование «ошибка меньше \(10^{-5}\)» означает именно normalized error **power**, оно эквивалентно NMSE < −50 dB. Если это обычный MSE, результат зависит от масштаба сигнала. Если это относительная RMS-amplitude error, то \(10^{-5}\) и −100 dB power NMSE — совсем другой порог. Требование следует зафиксировать до tuning.

По умолчанию один complex multiply считается как 4 real multiplications + 2 real additions; реализация Gauss даёт 3 multiplications + 5 additions. В статьях часто «6 FLOPs» означает четыре умножения и два сложения, а не шесть real multiplications. Parameter count, FLOPs, stored coefficients, active coefficients и реальная latency — разные величины.

## 2. Обязательные современные работы

### 2.1 OpenDPDv2 / TRes-DeltaGRU

Источники: [arXiv:2507.06849](https://arxiv.org/abs/2507.06849), [HTML v2](https://arxiv.org/html/2507.06849v2), [официальный репозиторий](https://github.com/lab-emi/OpenDPD). Первая версия: 2025-07-09; v2: 2025-12-16; на дату обзора статья обозначена как under review.

Эксперимент APA_200MHz:

- PA: 3.5 GHz GaN Doherty, внутренняя плата Ampleon AR211132; \(P_\mathrm{out,avg}=41.2\) dBm, \(P_{1dB}=46.5\) dBm, \(P_{3dB}=50\) dBm.
- Сигнал: TM3.1a, 5 × 40 MHz = 200 MHz, 256-QAM OFDM, 983.04 MS/s, PAPR 10.01 dB при CCDF 0.001%.
- Capture: 98 304 complex samples; split 60/20/20.
- Обучение: end-to-end DPD через frozen differentiable PA surrogate; PyTorch 2.4.1/CUDA 12.4/RTX 4090, AdamW, 240 epochs, lr \(5\cdot10^{-3}\), batch 64; checkpoint выбирается по validation ACPR.
- Модель: TRes front-end + DeltaGRU, hidden size 15. Конфигурации temporal convolution указаны как `(2,3,3,16,16)` и `(3,2,1,1,1)`.
- Финальный сигнал лучшей DPD измеряется на физическом PA; это сильнее, чем оценка только через surrogate.

Сопоставимые результаты APA_200MHz:

| Model | Trainable params | Paper “FLOPs”/sample | NMSE, dB | EVM, dB | Avg. ACPR, dBc |
|---|---:|---:|---:|---:|---:|
| No DPD | — | — | −20.5 | −24.7 (5.84%) | −28.3 |
| RVTDCNN | 1007 | 1587 | −32.9 | −33.9 | −50.8 |
| PG-JANET | 1130 | 1507 | −36.6 | −39.3 | −58.2 |
| DVR-JANET | 1097 | 1370 | −37.0 | −38.6 | −59.4 |
| BO-JANET | 1064 | 1535 | −39.8 | −42.9 | −58.7 |
| APNRRU | 1043 | 1328 | −36.0 | −38.1 | −58.6 |
| TRes-GRU | 999 | 1282 | −38.4 | −41.2 | −59.0 |
| TRes-DeltaGRU | 999 | 1324 | −39.6 | −42.1 | −59.9 |

В основной таблице OpenDPDv2 дан средний ACPR, а не отдельные left/right значения. Поэтому восстановить асимметрию соседних каналов из неё нельзя.

Temporal sparsity ablation:

| Target temporal sparsity | Mean active params | NMSE, dB | EVM, dB | ACPR, dBc |
|---:|---:|---:|---:|---:|
| 0% | 999 | −39.8 | −42.1 | −59.9 |
| 56% | 450 | −38.1 | −40.6 | −52.9 |
| 72.5% | 288 | −36.0 | −37.0 | −52.0 |
| 80.2% | 222 | NR | −35.1 | −46.7 |

Quantization ablation:

| Sparsity | Precision | EVM, dB | ACPR, dBc |
|---:|---|---:|---:|
| 0% | FP32 | −42.1 | −59.9 |
| 0% | W16A16 | −41.2 | −58.8 |
| 0% | W12A12 | −37.3 | −54.5 |
| 56% | W16A16 | −39.3 | −53.2 |
| 56% | W12A12 | −35.2 | −51.8 |
| 72.5% | W16A16 | −34.2 | −48.2 |
| 72.5% | W12A12 | −31.0 | −45.2 |

Аппаратная часть — gem5 simulation ARMv7-A 32-bit с NEON, GCC `-O3`, L1 I/D по 8 KB, DDR4 1 GB и 7 nm energy model, последовательность 10k samples. INT12 оценивается примерно в 2.8× экономии энергии, INT12 + 56/72.5% sparsity — в 4.5×/5.2×. Это не fabricated ASIC и не прямой RF-throughput measurement. Авторы отдельно отмечают доминирование memory traffic.

Ограничения:

- paper FLOPs — агрегированные scalar operations, не число real multiplications;
- “active parameters” — proxy ожидаемой работы, а не размер сохранённой модели;
- delta thresholding требует comparison, index/cache/state traffic, которые нельзя обнулить;
- training/calibration wall-clock не приведён;
- robustness к thermal/bias drift мотивирован, но не измерен;
- внутри статьи имеется гипотетическая power breakdown; её нельзя представлять как существующий 1 GHz neural-DPD chip.

Вывод: это главный apples-to-apples reference только на APA_200MHz. Для заявления о превосходстве новый метод должен пройти тот же physical PA, split, normalization, alignment и ACPR evaluator.

### 2.2 TCN-DPD

Источники: [arXiv:2506.12165](https://arxiv.org/abs/2506.12165), [HTML](https://arxiv.org/html/2506.12165v1), [IEEE DOI](https://doi.org/10.1109/IMS40360.2025.11103923). IMS 2025.

- Dataset: OpenDPD **DPA_200MHz**, 10 × 20 MHz 64-QAM OFDM от 40 nm digital transmitter.
- PA: fixed pretrained DGRU surrogate, PA simulation NMSE −31.84 dB.
- DPD: noncausal dilated depthwise-separable 1-D TCN, residual I/Q; четыре depthwise convolutions, kernel 5, dilation base 2; Hardswish выбран из 22 activations.
- Training/evaluation: E2E через frozen surrogate; 5 random seeds.
- Физическая PA verification отсутствует; она названа ongoing work.

Сопоставимые результаты только внутри DPA_200MHz:

| Model | Approx. params | NMSE, dB | ACPR L/R, dBc | EVM, dB |
|---|---:|---:|---:|---:|
| No DPD | — | NR | −31.90 / −30.45 | −34.02 |
| LSTM | ~500 | −35.22 ± 3.86 | −43.60 / −42.68 | −37.52 |
| GRU | ~500 | −40.01 | −44.95 / −43.76 | −42.70 |
| RVTDCNN | ~500 | −32.03 | −48.04 / −46.26 | −34.61 |
| VDLSTM | ~500 | −32.50 | −47.04 / −45.85 | −34.94 |
| PN-TDNN | ~500 | −35.49 | −49.25 / −48.43 | −37.70 |
| DGRU | ~500 | −41.82 | −50.57 / −49.16 | −44.04 |
| TCN-200 | ~200 | −41.27 | −45.83 / −46.76 | −43.81 |
| TCN-500 | ~500 | −44.61 ± 1.37 | −51.58 ± 2.84 / −49.26 ± 2.04 | −47.52 ± 1.49 |
| TCN-1000 | ~1000 | −46.37 | −52.58 / −50.84 | −49.40 |

Параметры не равны operations/sample; полная стоимость Hardswish, buffering и convolutions не дана. Noncausal convolutions требуют look-ahead и добавляют streaming latency. Эти числа нельзя ставить в один рейтинг с APA_200MHz OpenDPDv2.

### 2.3 SparseDPD

Источники: [arXiv:2506.16591](https://arxiv.org/abs/2506.16591), [HTML](https://arxiv.org/html/2506.16591v1), [официальный код](https://github.com/MannoVersluis/SparseDPD), [FPL 2025 DOI](https://doi.org/10.1109/FPL68686.2025.00031).

- Capture: 172 035 samples, MATLAB 20 MHz 64-QAM, 3.5 GHz GaN Doherty, \(P_\mathrm{out}=41.5\) dBm, split 60/20/20.
- PNTDNN: 2 hidden layers × 12 neurons; unstructured iterative magnitude pruning.
- QAT 400 epochs; шесть pruning rounds по 200 epochs, каждый удаляет 20% минимальных weights; batch 256, frame 500, stride 1, lr \(10^{-3}\) с plateau decay; один seed.
- Итог: 64 stored parameters, 74% zeros, заявлено 72 operations/sample.
- Fixed-point: input/weights/activations Q1.13 (14 bit), intermediate Q2.13, output Q2.27 (29 bit); inverse-square-root LUT + две Newton iterations.
- FPGA: Zynq-7Z010, 170 MHz/170 MS/s, 12.2 GOPS, 2298 LUT, 1724 FF, 66 DSP (82.5%), 13 BRAM; simulated dynamic 241 mW, total 405 mW.
- DPD + surrogate: NMSE −48.2 dB, EVM −54.0 dB, ACPR −59.4 dBc.

Критически важная оговорка: статья прямо говорит, что PNTDNN оценён “using OpenDPD framework with simulated results”. DPD metrics не являются повторным физическим измерением PA после predistortion. FPGA power/timing — post-implementation simulation. Поэтому нельзя писать «SparseDPD физически получил −59.4 dBc».

Ограничения:

- один seed и узкополосный 20 MHz сигнал;
- unstructured zeros ускоряют только движок, реально пропускающий нули; dense matrix multiply остаётся плотным;
- hard-coded sparse topology переносит irregular connectivity в routing/control;
- utilization 82.5% DSP на выбранном малом FPGA ограничивает запас;
- не сопоставимо с 200 MHz APA/DPA.

### 2.4 Gradient-adaptive spline-interpolated LUT DPD

Источники: [arXiv:1907.02350](https://arxiv.org/abs/1907.02350), [PDF](https://arxiv.org/pdf/1907.02350), [IEEE TCSI DOI](https://doi.org/10.1109/TCSI.2020.3034825). Campo et al., journal version 2020.

Две архитектуры:

- SPH: complex memoryless spline LUT, затем короткий complex FIR (spline-based Hammerstein);
- SMP: параллельные delayed complex spline LUT branches (spline memory polynomial).

В обоих случаях cubic B-splines с равномерными amplitude knots и complex control points; ILA, пять iterations. SPH обновляет LUT и FIR decoupled LMS, SMP — LMS.

Конвенция статьи: complex multiply = 6 FLOPs = 4 real multiplications + 2 additions; complex-real multiply = 2; complex add = 2. Тривиальные умножения на 0/1/степени двойки/0.5 исключены.

Пример \(P_\text{spline}=3,Q=7,M=4\):

| Model | Stored coefficients | Real mult./sample, inference | Real mult./sample, learning |
|---|---:|---:|---:|
| SPH | 14 | 40 (24 spline + 16 FIR) | 124 |
| SMP | 31 | 63 | 119 |
| MP, order 11, memory 4 | 24 complex | 112 | 2514 |

Физические эксперименты:

| Set | PA / signal | Model | Coeff. | FLOPs | Real mult. | EVM | Adjacent metric |
|---|---|---|---:|---:|---:|---:|---:|
| E1 | Mini-Circuits ZHL-4240, 3.5 GHz, 100 MHz NR DL OFDM, 30 kHz SCS, 264 PRB/95.04 MHz occupied, +27 dBm | No DPD | — | — | — | 7.82% | max adjacent PSD −23.8 dBm/MHz |
| E1 | same | SPH M3Q7 | 13 | 69 | 36 | 5.54% | −36.3 dBm/MHz |
| E1 | same | SMP M4Q7 | 31 | 99 | 63 | 5.57% | −37.8 dBm/MHz |
| E1 | same | MP O11M4 | 24 | 255 | 112 | 5.47% | −38.2 dBm/MHz |
| E2 | Skyworks SKY66293-21 n78, 3.65 GHz, 100 MHz NR, +24 dBm | SPH M4Q7 | 14 | 77 | 40 | 5.57% | −33.2 dBm/MHz |
| E2 | same | SMP M5Q7 | 38 | 111 | 75 | 5.55% | −33.1 dBm/MHz |
| E2 | same | MP O11M5 | 30 | 255 | 136 | 5.54% | −33.2 dBm/MHz |
| E3 | 64-element Anokiwave AWMF-0129, 28 GHz OTA, 100/200 MHz NR 64-QAM, EIRP 42.5 dBm | No DPD | — | — | — | 12.10/12.43% | TRP ACLR 26.10/26.30 dB |
| E3 | same | SPH M3Q7 | NR | 69 | 36 | 6.20/6.25% | 34.4/34.1 dB |
| E3 | same | SMP M4Q7 | NR | 99 | 63 | 6.15/6.20% | 34.8/34.4 dB |
| E3 | same | MP O11M4 | NR | 255 | 112 | 6.00/6.13% | 35.2/35.0 dB |

E1/E2 используют 491.52 MS/s; E1 — 5 × 100k training samples; E3 — 5 × 50k и 7× oversampling. EVM measurement floor около 4%. NMSE не приведён. Результаты подтверждают, что spline memory/Hammerstein дают большую часть линейзации MP при существенно меньшем числе умножений, но не доказывают превосходство над OpenDPD на его PA.

### 2.5 Piecewise closed-loop DPD

Источники: [arXiv:2003.06348](https://arxiv.org/abs/2003.06348), [PDF](https://arxiv.org/pdf/2003.06348). Brihuega et al., 2020.

- PA: 64-element AWMF-0129 active array, 28 GHz.
- Signal: 400 MHz 5G NR OFDM, 64-QAM, 120 kHz SCS, 3168 active subcarriers/FFT 4096, 5× oversampling, PAPR 7 dB @ 0.01%.
- Algorithm: piecewise closed-loop decorrelation/error minimization, adaptive partitioning и basis pruning.
- Baselines: PW-ILA, single-polynomial ILA и closed-loop variants.
- Update: closed-loop 10 blocks × 20k samples; ILA 4 iterations.
- Claimed system result: starting ACLR down to 21 dBc и примерно +4 dB usable EIRP против reference approaches.
- Pruning: 348 → 96 basis functions с близкой линеаризацией.

При +43.3 dBm:

| Variant | Forward FLOPs/sample | Learning normalized FLOPs/sample |
|---|---:|---:|
| PW-ILA | 935 | \(2.7\cdot10^9\) |
| CL self-orthogonalized | 935 | 54 520 |
| CL orthogonalized | 27 847 | 928 |
| PW-CL self-orthogonalized | 935 | 54 520 |
| PW-CL orthogonalized | 27 847 | 928 |
| PW-CL orthogonalized + pruned | 2219 | 323.2 |

Комплексное умножение считается как 6 FLOPs. Exact NMSE не дан; ключевые RF curves находятся на графиках, которые здесь не оцифровывались. Work показывает компромисс: orthogonalization резко ускоряет calibration convergence, но может сделать forward implementation очень дорогой; pruning частично возвращает приемлемую стоимость. Beam-dependent load modulation означает, что frequent online recalibration важнее одной статической точности.

### 2.6 Low-complexity DPD through feature selection

Источники: [arXiv:2607.15441](https://arxiv.org/abs/2607.15441), [HTML](https://arxiv.org/html/2607.15441v1), [dataset DOI](https://doi.org/10.48804/306IIM).

Существование и дата проверены: arXiv v1 отправлена **2026-07-16**. На дату обзора это свежий preprint/SiPS submission, не peer-reviewed archival result.

- DUT: два измеренных 15 GHz FR3 PA.
- Signal: 100 MHz single-carrier 64-QAM, RRC \(\alpha=0.35\), sample rate 1.28 GHz.
- Split: 33/33/33.
- ILA/postinverse learning.
- Candidate dictionary: 321 200 real features из \(N=15k,M=200,P=7\).
- Pipeline: LASSO сокращает примерно до 2000, MRMR ранжирует top 100–200, затем residual PNN.
- Offline feature selection исключён из runtime cost; значит, reported inference complexity не описывает calibration cost.

DUT1: ADL9006 GaAs, 15 GHz, \(P_\mathrm{in}=-2\) dBm, 10 dB output attenuation.

| Model | FLOPs/sample | Coefficients | Validation inverse NMSE, dB | Measured EVM | Measured worst ACLR, dBc |
|---|---:|---:|---:|---:|---:|
| No DPD | — | — | — | 2.50% | −33.0857 |
| PNN | 360 | 128 | −32.7998 | 1.88% | −39.9457 |
| PNN | 787 | 287 | −35.1755 | 1.54% | −41.0505 |
| PNN | 1839 | 725 | −37.4431 | 1.39% | −39.0286 |
| PNN | 2491 | 911 | −41.1065 | 1.39% | −41.4168 |
| Proposed | 407 | 98 | −35.3823 | 1.38% | −43.8113 |
| Proposed | 939 | 344 | −37.6726 | 1.24% | −41.2444 |
| Proposed | 1768 | 650 | −41.1615 | 1.08% | −41.8760 |

Около −42 dB validation inverse NMSE оба подхода достигают примерно при 1600 parameters; selected model около −37 dB требует 595 FLOPs против 797 у PNN. Таблица физического DPD+PA measurement относится к DUT1. Для DUT2 (TLPA2G22G-43-43-HS, до 53 dB gain, \(P_\mathrm{in}=-20\) dBm, 30 dB attenuation) показано примерно −34 dB inverse-model NMSE при 234 FLOPs, что на 32% ниже PNN; полной physical EVM/ACLR table для DUT2 нет.

Ограничения: огромный offline dictionary, отсутствуют selection/training wall-clock, один capture split и нет seed confidence intervals, EVM instrumentation equalization, validation NMSE относится к postdistorter prediction, а не непосредственно к physical cascade. Это сильный источник для feature-selected dictionary, но не прямой конкурент OpenDPDv2.

## 3. Классические модели и методы обучения

| Направление | Первичный источник | Что установлено | Главный риск/ограничение |
|---|---|---|---|
| ILA / Volterra inverse | Eun & Powers, 1997, [DOI](https://doi.org/10.1109/78.552219) | Postdistorter учится \(y\rightarrow x\), затем coefficients копируются в predistorter; RLS/Volterra | Требуется существование устойчивой применимой inverse; training direction не заменяет deployment test |
| ILA equivalence | Jiang & Wilford, 2013, [arXiv:1306.0037](https://arxiv.org/abs/1306.0037) | При сформулированных предпосылках postdistorter совпадает с predistorter | Теорема не оправдывает круговую test procedure на уже известном \(y_\text{test}\) |
| Memory polynomial | Ding et al., 2004, [DOI](https://doi.org/10.1109/TCOMM.2003.822188) | Диагональное Volterra-приближение; linear-in-coefficients LS | Не ловит все leading/lagging envelope cross-products |
| Measured MP precursor | Kim & Konstantinou, 2001, [DOI](https://doi.org/10.1049/el:20010940) | Measured single/multicarrier UMTS PA predistortion | Старый узкополосный стенд; не современная 200 MHz задача |
| GMP | Morgan et al., 2006, [DOI](https://doi.org/10.1109/TSP.2006.879264) | Добавляет leading/lagging cross terms; original 30 W, 2 GHz PA | Dictionary быстро растёт; exact open table не извлечена из paywall |
| DDR Volterra | Zhu et al., 2008, [DOI](https://doi.org/10.1109/TMTT.2008.925211) | Dynamic deviation reduction делает Volterra практичнее | Всё ещё комбинаторный рост terms/order/memory |
| Direct learning | Zhou & DeBrunner, 2007, [DOI](https://doi.org/10.1109/TSP.2006.882058) | NFXRLS, nonlinear-adjoint LMS/RLS оптимизируют forward output error | Нужны secondary-path/PA derivatives либо model; чувствительность к model bias |
| ILC | Chani-Cahuana et al., 2016, [DOI](https://doi.org/10.1109/TMTT.2016.2588483) | Итеративно инвертирует waveform; полезен как high-quality teacher для компактного DPD | Несколько PA acquisitions, waveform-specific target, слабая online пригодность |
| Concurrent decorrelation | Abdelaziz et al., 2017, [DOI](https://doi.org/10.1109/TMTT.2017.2706688), [author record](https://trepo.tuni.fi/handle/10024/213605) | Closed-loop adaptation с одним feedback receiver | Скорость/устойчивость зависят от conditioning и feedback distortion |
| Hybrid-MIMO decorrelation | [arXiv:1804.02178](https://arxiv.org/abs/1804.02178) | Расширяет closed-loop decorrelation на hybrid MIMO | Beam/load-dependent PA, limited observation paths |
| NLMS Volterra | 2012 simulation paper, [DOI](https://doi.org/10.1016/j.proeng.2012.04.172) | LMS/NLMS/VSS для 16-QAM OFDM Volterra predistorter | Simulation, не hardware/physical evidence |
| NFxPEM/NFxLMS | Gan dissertation, 2009, [institutional page](https://www.spsc.tugraz.at/phd-theses/adaptive-digital-predistortion-of-nonlinear-systems.html) | Систематический adaptive-DPD treatment | Нельзя переносить thesis simulations на современный wideband PA |

Отдельный канонический и хорошо верифицированный **block-RLS RF-DPD** источник в доступном корпусе не найден. Нельзя автоматически называть batch/block least squares или QR solve «block-RLS»: RLS рекурсивно обновляет inverse covariance, а block LS решает новую normal equation. Этот пробел следует явно оставить в literature matrix, а не заполнять вторичным пересказом.

### Почему круговая inverse/forward проверка недостаточна

Пусть физический или surrogate PA — \(f\), а inverse model — \(h\). ILA обучает

\[
h(y/g)\approx x.
\]

Проверка

\[
y_\text{test}/g \rightarrow h \rightarrow \hat x
\rightarrow f \rightarrow \hat y\approx y_\text{test}
\]

показывает, что \(f\circ h\) согласован с identity **на распределении уже наблюдённых PA outputs**. В реальной эксплуатации требуется другое:

\[
x_\text{desired}\rightarrow h\rightarrow z\rightarrow
f(z)\approx g x_\text{desired}.
\]

Первый тест заранее использует \(y_\text{test}\), может скрыть distribution mismatch, noninvertible compression и совместную ошибку forward/inverse surrogate. ILA остаётся допустимым способом обучения при предпосылках inverse-equivalence; неверным является направление теста, а не сама ILA.

## 4. Sparse regression, feature selection и piecewise models

### 4.1 OMP/LASSO на физическом PA

Barry, Li, Becerra и Gilabert, 2021: [DOI](https://doi.org/10.3390/s21175772), [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8433820/), [author PDF](https://idus.us.es/server/api/core/bitstreams/cbb4722e-77cd-46d4-837a-e5ad91bd4e26/content).

- Physical load-modulated balanced amplifier; 4 × LTE20 carrier aggregation, 64-QAM; occupied spread 120 MHz, sample rate 614.4 MS/s.
- Separate training и validation blocks по 307 200 samples.
- Initial GMP dictionary: 979 coefficients.
- Сравнение doubly orthogonal matching pursuit (DOMP), doubly matching selection (DMS) и LASSO.

При 17 selected coefficients:

| Method | Worst/best ACPR, dBc | Worst/best EVM |
|---|---:|---:|
| No DPD | −31.7 / −36.6 | 3.0 / 2.3% |
| DOMP-GMP | −45.0 / −46.1 | 1.2 / 0.9% |
| DMS | −44.4 / −45.5 | 1.3 / 1.0% |
| LASSO | −41.0 / −43.6 | 1.6 / 1.1% |

В их MATLAB/code/hardware DOMP был около 10× быстрее DMS, но около 2× медленнее LASSO; это не универсальная асимптотическая гарантия. Работа хорошо поддерживает greedy delay/branch selection, но не прямое сравнение с APA_200MHz.

Becerra et al., 2019, “Comparative Analysis of Greedy Pursuits…”, [institutional record](https://idus.us.es/items/af48568e-09c0-443d-ade6-c7b05e17bd8d), IEEE TMTT 67(9), 3575–3585: class-AB и class-J PA, 15 MHz LTE; при target −45 dB NMSE DOMP уменьшал коэффициенты примерно на 96%, PCA — на 85%. Exact table в доступной записи не извлечена.

### 4.2 Structured sparsity / group LASSO

Hemsi & Panazio, 2022, sparse flexible reduced-Volterra, [DOI](https://doi.org/10.1109/ACCESS.2022.3223369): LASSO и sparse-group-LASSO для structured dictionary. Sparse RV2 давал примерно на 3 dB лучший ACPR, чем GMP при сопоставимой или меньшей running cost. Offline SLEP до 5000 iterations на \(\lambda\), затем AIC model selection: inference дешёвый, но calibration не бесплатный.

Tanio et al., 2020, structured sparse neural DPD, [DOI](https://doi.org/10.1109/ACCESS.2020.3005146): 3.5 GHz GaN Doherty, 15 MHz LTE, 96 MS/s; group-lasso ETDNN примерно в 30 раз дешевле conventional NN при сходном ACLR. Не следует смешивать эту работу с отдельной статьёй 2022 года “A sparse neural network-based power adaptive DPD design and its hardware implementation”, на которую ссылается SparseDPD.

### 4.3 CPWL / decomposed vector rotation

Zhu, 2015, decomposed-vector-rotation behavioral model: [DOI](https://doi.org/10.1109/TMTT.2014.2387853), [author PDF](https://researchrepository.ucd.ie/rest/bitstreams/25439/retrieve). Это modified CPWL с phase-restored complex basis и linear-in-parameters LS.

Physical envelope-tracking test:

- in-house GaN class-AB + envelope modulator, 2.14 GHz, 20 MHz LTE, PAPR 6.5 dB, \(P_\mathrm{out}=45\) dBm, 122.88 MS/s;
- 16k samples, 4k extraction и separate evaluation, ILA;
- DVR \(K=8,M=3\), 84 coefficients: ACPR ±20 MHz 54.4/54.0 dBc против 30.8/32.1 без DPD; ±40 MHz 54.0/53.8 против 40.3/40.9; NRMSE 0.98% против 8.42%;
- piecewise Volterra, 146 coefficients: 51.5/52.1 dBc, 51.7/51.4 dBc, NRMSE 1.10%.

Physical Doherty test:

- in-house LDMOS, 2.14 GHz, \(P_\mathrm{out}=47\) dBm;
- 60 MHz mixed waveform: 18 MHz four-carrier GSM + 20 MHz LTE, 368.64 MS/s;
- DVR \(K=4,M=6\), 95 coefficients: LTE ACPR улучшен более чем на 28 dB, GSM IMD3 примерно на 37 dB; NRMSE 15.8% → 0.69%.

CPWL/DVR полезен как linear-regression dictionary с piecewise нелинейностью, но basis generation, magnitude и phase restoration тоже стоят операций. Coefficients нельзя выдавать за multiplications/sample.

Дополнительные первичные anchors: ранняя threshold CPWL работа [DOI](https://doi.org/10.1002/mmce.20498) и complex CPWL simulation [DOI](https://doi.org/10.1587/elex.8.1556).

## 5. Phase-equivariant и neural DPD

### 5.1 Phase-normalized recursive neural network

Fischer-Bühner et al., 2024: [DOI](https://doi.org/10.1109/LMWT.2024.3393859), [open author PDF](https://pure.tue.nl/ws/portalfiles/portal/353887253/Recursive_Neural_Network_With_Phase-Normalization_for_Modeling_and_Linearization_of_RF_Power_Amplifiers.pdf).

Modeling test: RTH18008S-30 GaN Doherty, 1.8 GHz, 100 MHz 5G OFDM, 38.1 dBm; 180k train/120k evaluation, five cycles. DPD test: QPA3503 GaN Doherty, 3.5 GHz, 200 MHz 5G multicarrier, PAPR 8 dB, 34.2 dBm; ILC teacher обновлялся каждые 400 epochs; test 210k samples. TensorFlow BPTT: sequence 40, batch 20T, 2200 epochs.

| Model | Params / states | NMSE, dB | ACLR, dBc | EVM |
|---|---:|---:|---:|---:|
| No DPD | — | −20.2 | −26.1 | 8.40% |
| ILC teacher | waveform | −51.9 | −57.9 | 1.81% |
| VDLSTM | 1080 / 15 | −36.7 | −43.1 | 2.18% |
| DVR-JANET | 1098 / 12 | −38.3 | −45.0 | 2.03% |
| PN-TDNN | 1035 / stateless | −40.8 | −49.1 | 1.94% |
| PN-RNN | 1094 / 13 | −42.8 | −50.0 | 1.89% |

PN-RNN показывает ценность phase-normalization и небольшого state, но operations/sample и hardware results не приведены. Предшествующий PNN: [DOI](https://doi.org/10.1109/LMWT.2023.3290980).

Phase equivariance

\[
\mathrm{DPD}(x e^{j\phi})=\mathrm{DPD}(x)e^{j\phi}
\]

естественно соблюдается моделью \(z=xC(|x|)\) и уменьшаeт гипотезное пространство. Две независимо обученные произвольные I/Q сети её в общем случае нарушают.

### 5.2 Complex-valued NN

Надёжного открытого peer-reviewed physical-DPD источника, который одновременно даёт full complex NN architecture, real-multiplication count и hardware result, в проверенном корпусе не найдено. В качестве воспроизводимого, но не peer-reviewed hardware evidence существует официальный MATLAB R2026a example: [Complex-Valued Neural Network for DPD](https://www.mathworks.com/help/comm/ug/complex-valued-neural-network-for-digital-predistortion-design-offline-training.html). Там 100 MHz 16-QAM OFDM и simulated complex PA; Cardioid complex network сообщает около −39.4 dB NMSE против −37.5 dB у real leaky-ReLU model. Это documentation example и не может конкурировать с physical OpenDPD result.

Ранняя dual-output amplitude/phase neural DPD: Naskas & Papananos, 2004, [DOI](https://doi.org/10.1109/TCSII.2004.837284); prototype показывает до 25 dB linearity improvement, но это не современная fully complex wideband architecture.

## 6. DeltaGRU, quantization и hardware evidence

### 6.1 DeltaDPD

Источники: [arXiv:2505.06250](https://arxiv.org/abs/2505.06250), [IEEE DOI](https://doi.org/10.1109/LMWT.2025.3565004).

- 3.5 GHz GaN Doherty, 41.5 dBm; TM3.1a 200 MHz 256-QAM OFDM; 98 304 samples, split 60/20/20.
- Frozen DGRU PA model: 2751 params, surrogate NMSE −40.04 dB.
- AdamW, 200 epochs.
- DeltaGRU: 573 params; при 52% temporal sparsity NMSE −37.22 dB, EVM −38.52 dB, ACPR −50.03 dBc.
- Estimated 7 nm energy 6.41 nJ/inference, примерно 1.7–1.8× saving.

Spectrum физически измерялся, но energy — model estimate. Temporal sparsity зависит от waveform и threshold; следует измерять cache/update distribution, worst-case, а не только mean active fraction.

### 6.2 Mixed-precision GRU DPD

Источник: [arXiv:2404.15364](https://arxiv.org/abs/2404.15364).

- 2.4 GHz 40 nm digital PA, 160 MHz (4 × 40 MHz) 1024-QAM OFDM, 640 MS/s, 491 520 samples, PAPR 10.38 dB, 13.75 dBm, split 60/20/20.
- E2E via surrogate; Adam, 100 epochs, lr \(10^{-3}\), batch 3200; models около 500 parameters.

| Model | ACPR L/R, dBc | EVM, dB | MUL/sample | ADD/sample | MEM/sample |
|---|---:|---:|---:|---:|---:|
| No DPD | −31.69 / −32.45 | −27.05 | — | — | — |
| GMP, 495 params | −40.79 / −40.86 | −29.27 | 2190 | 3668 | 517 |
| GRU FP32, 502 params | −43.36 / −45.30 | −38.46 | 502 | 1417 | 506 |
| GRU W16A16 | −43.75 / −45.27 | −38.72 | same topology | same topology | same topology |
| GRU W12A16 | −43.03 / −44.69 | −37.47 | same topology | same topology | same topology |
| GRU W12A12 | −42.36 / −43.79 | −37.45 | same topology | same topology | same topology |
| GRU W8A8 | −35.84 / −35.70 | −28.89 | same topology | same topology | same topology |

Feature extraction остаётся FP32 (14 MUL, 17 ADD). 7 nm powers 1.98 W FP32, 0.71 W W16A16 и 2.8×/3.7×/3.8×/4.5× savings для W16A16/W12A16/W12A12/W8A8 — estimates, не silicon measurement.

### 6.3 DPD-NeuralEngine ASIC study

Источник: [arXiv:2410.11766](https://arxiv.org/abs/2410.11766).

- 22 nm GF22FDX post-layout simulation, не fabricated chip.
- GRU: 502 params, 4 inputs/10 hidden/1 layer, W12A12 Q2.10.
- Physical GaN Doherty PA около 40 dBm; 80 MHz 64-QAM OFDM/PAPR 8.2 dB (hardware table также указывает 60 MHz baseband — эту неоднозначность надо сохранять).
- QAT 300 epochs, batch 64, frame 50, stride 1.
- Post-layout: 2 GHz clock, 250 MS/s, 1026 operations/sample, 7.5 ns, 256.5 GOPS, 195 mW, 0.2 mm², 6.58 TOPS/W/mm².
- RF: ACPR −45.3 dBc, EVM −39.8 dB.

Результат важен как hardware mapping proof, но power/timing — post-layout estimate и RF dataset отличен от OpenDPDv2.

### 6.4 FPGA anchors

Huang, Xia, Boumaiza, 2020: [DOI](https://doi.org/10.1109/TMTT.2020.2993236), IEEE TMTT 68(9), 4066–4076. ZCU102, parallel pruned CRV/GMP-family engine, 2.4 GS/s для 400 MHz signal, measured 28 GHz PA. Это реальный high-throughput FPGA design, однако его operation convention и PA dataset отличны от OpenDPD.

Real-time model switching DPD, 2022: [DOI](https://doi.org/10.1109/TMTT.2021.3132347), IEEE TMTT 70(3), 1500–1508. Decision tree выбирает один cross-term/coefficient set на sample, что уменьшает dynamic runtime; есть physical PA и FPGA/power evidence. Нельзя переносить абсолютный ACPR в общую таблицу.

Контекстная hardware-таблица из SparseDPD (не рейтинг):

| Work/model | Params | Ops/sample | Throughput | Signal BW | Reported RF metric | Nature of hardware result |
|---|---:|---:|---:|---:|---|---|
| SparseDPD PNTDNN | 64 | 72 | 170 MS/s | 20 MHz | NMSE −48.2, EVM −54.0, ACPR −59.4 | RF through surrogate; FPGA post-implementation sim |
| ETDNN | 40 | NR | 368.64 MS/s | 46.08 MHz | NMSE −39.5, EVM −42.3, ACPR ~−50 | Separate dataset |
| FPGA MP | 9 | 30 | 250 MS/s | 20 MHz | ACPR −49 | Separate dataset, 0.244 W |
| FPGA GMP | 38 | 149 | 400 MS/s | 100 MHz | NMSE −38.5, ACPR −46.5 | Separate dataset, 0.89 W |
| Pruned FPGA GMP/CRV | 36 | 17 | 2.4 GS/s | 400 MHz | EVM −39.2, ACPR −44.7 | Separate 28 GHz dataset, 0.96 W |
| GPU TDNN | 909 | ~1818 | 1 GS/s | 200 MHz | NMSE −38.3, EVM −35.3, ACPR −45.2 | Separate dataset |
| ASIC GRU | 502 | 1026 | 250 MS/s | 60/80 MHz | EVM −39.8, ACPR −45.3 | 22 nm post-layout, 0.195–0.2 W |

Нули ускоряют inference только при zero-skipping datapath. FPGA clock × samples/clock — throughput, но не end-to-end latency с buffering. Post-synthesis/post-layout power нельзя называть measured chip power.

## 7. Online adaptation, drift и generalization

| Problem | Primary source | Evidence | Limitation |
|---|---|---|---|
| Power/BW/waveform changes | Yu et al., 2022, continual-learning DPD, [DOI](https://doi.org/10.1109/TMTT.2022.3210199) | Doherty PA; retention/merging states under changing power, bandwidth and signal type | Exact result table paywalled; no values invented here |
| Dynamic complexity under operating point | Jiang et al., 2023, gated dynamic NN, [DOI](https://doi.org/10.1109/TMTT.2023.3241612) | Varying power/BW/PAPR; sparse gating reduces runtime >50% with small degradation | Absolute metrics not recovered from accessible primary text |
| Temperature | Jindal et al., 2022, [DOI](https://doi.org/10.1109/TMTT.2022.3175155), [open repository](https://eprints.whiterose.ac.uk/id/eprint/192792/) | 3.75 GHz 10 W GaN-on-SiC class-B, 20–80°C, PAPR 6–12 dB, multiple BWs; thermal model NRMSE <7%, ACPR +1–4 dB versus temperature-less model | Different task/data; not OpenDPD comparison |
| Bias/power changes | Dawar et al., 2016, [DOI](https://doi.org/10.1049/iet-com.2015.1048) | Direct-learning adaptive MP, 20 MHz LTE; input power/drain/gate bias changes | Exact table paywalled |
| Beam/load changes | Piecewise closed-loop, [arXiv:2003.06348](https://arxiv.org/abs/2003.06348) | 28 GHz active array; beam-dependent load modulation motivates frequent adaptation | High forward/calibration cost in some variants |
| Carrier frequency | 2025 coefficient-interpolation study, [DOI](https://doi.org/10.3390/app15189899) | 3.3–3.8 GHz measured modeling, interpolation from three frequencies; up to 9 dB average NMSE gain vs fixed MP | Behavioral model, not full DPD apples-to-apples |

OpenDPDv2 обсуждает thermal/electron-trapping/bias dynamics от microseconds до milliseconds, но не проводит drift/adaptation experiment. Поэтому online calibration speed должна измеряться отдельно: samples/acquisitions to target ACPR, wall-clock, feedback bandwidth и recovery after operating-point step.

## 8. Выводы для нового low-cost baseline

### 8.1 Memoryless complex linear spline

\[
z[n]=x[n]C(|x[n]|),
\qquad
C(r)=c_k+t(c_{k+1}-c_k), \quad t\in[0,1].
\]

Модель phase-equivariant, использует два соседних complex control points и обучается одним complex ridge solve. На sample при известном segment:

- interpolation в форме \(c_k+t\Delta c_k\): 2 real multiplications + 2 real additions;
- complex multiply \(xC\): 4 real multiplications + 2 real additions;
- итого ядро: примерно 6 real multiplications + 4 real additions;
- дополнительно: magnitude или power, segment lookup/comparison, reciprocal/division либо precomputed scale, две coefficient reads.

Если knots определены по \(r^2=I^2+Q^2\), magnitude вычисляется как 2 multiplications + 1 addition без square root; это привлекательный hardware variant. Однако linear interpolation по power — другая функция, чем linear interpolation по amplitude, поэтому его нужно считать отдельной knot strategy, а не бесплатной оптимизацией.

Это очень сильный cost baseline, но физические spline E1–E3 показывают, что memory branches/FIR часто нужны даже при 100 MHz. На 200 MHz нельзя заранее ожидать −50 dB NMSE от memoryless \(C(|x|)\).

### 8.2 Минимальная последовательность расширений

| Candidate | Expected accuracy | Inference cost | Calibration | Hardware fit | Main failure mode |
|---|---|---|---|---|---|
| Memoryless linear spline | AM/AM + AM/PM only | Очень низкая, local support | Один small complex ridge | Отличный LUT/interpolator | PA memory и hysteresis остаются |
| Spline → short FIR (SPH) | Linear memory после nonlinearity | \(O(K_\mathrm{active}+M)\) | Alternating/ridge/LMS | Хороший regular datapath | Не ловит envelope-dependent memory до spline |
| Sparse spline memory branches (SMP) | Branch-specific delayed envelope | \(O(B)\), 2 knots/branch | Group ridge/LASSO/OMP | Хороший, если delays fixed | Conditioning и branch duplication |
| Feature-selected GMP/CPWL | Выше при cross-memory | Пропорционально selected terms | OMP/group LASSO может быть дорог | MAC-friendly; irregular terms | Offline selection cost, generalization |
| State-conditioned spline | Slow thermal/bias envelope state | Несколько 1-pole states + local LUT | Joint/ridge при fixed beta | Очень хороший | Identifiability, mismatch timescale |
| Spline/MP + tiny residual TCN | Наиболее гибкий | Ниже full neural при малом residual | E2E сложнее | Буфер/activation/quantization | Surrogate exploitation и latency |

Рекомендованный порядок evidence:

1. memoryless complex spline на честном desired-\(x\) cascade;
2. SPH и delays \(m=\{0\},\{0,1\},\{0,1,2\}\);
3. greedy/group selection только если validation NMSE или left/right ACLR улучшаются;
4. state-conditioned branch только при доказанном slow residual correlation;
5. tiny neural residual только после ablation, показывающей, что dictionary extension дороже.

### 8.3 Что нужно измерять, а не оценивать параметрами

- left/right/average ACPR одним и тем же PSD definition;
- NMSE после integer/fractional delay и complex gain alignment;
- EVM, PAPR, maximum \(|z|\), AM/AM и AM/PM;
- stored vs active coefficients;
- real MUL, ADD, comparisons, LUT, nonlinear ops, reads/writes отдельно;
- batch throughput и streaming sample latency с warm-up/state;
- calibration wall-clock, PA acquisitions и samples-to-target;
- FP32, FP16-like, int16, int12 с saturation и accumulator width;
- worst-case, не только average temporal sparsity;
- три seeds для stochastic models и bootstrap/segment confidence intervals для deterministic regressions;
- новое power/waveform condition без retraining и после controlled recalibration.

## 9. Итоговая source-backed карта Pareto-кандидатов

Эта таблица не объединяет метрики разных datasets; она показывает, какой тип evidence существует.

| Method/work | Comparable group | Quality evidence | Cost evidence | Calibration evidence | Fixed-point/hardware | Directly comparable to OpenDPDv2? |
|---|---|---|---|---|---|---|
| OpenDPDv2 TRes-DeltaGRU | APA_200MHz | Physical PA: −39.6 NMSE, −42.1 EVM, −59.9 avg ACPR | 999 params, 1324 generic FLOPs; active-work ablation | 240 epochs; wall-clock NR | W16/W12 + gem5 7 nm estimate | Да, это reference |
| TCN-1000 | DPA_200MHz | Surrogate: −46.37 NMSE, −49.40 EVM, ACPR −52.58/−50.84 | ~1000 params; ops NR | 5 seeds; wall-clock NR | Нет | Нет: другой dataset, surrogate only |
| SparseDPD | SparseDPD-20MHz | Surrogate: −48.2 NMSE, −54 EVM, −59.4 ACPR | 64 params/72 ops | 400 + 6×200 epochs; one seed | 170 MS/s Zynq postimpl sim, int14 | Нет |
| SPH spline | E1/E2/E3 | Physical; near-MP EVM/adjacent metrics | 36–40 real MUL/sample in reported configs | 5 ILA, LMS; learning 124 real MUL/sample example | LUT + FIR friendly | Нет, но очень релевантный architecture prior |
| SMP spline | E1/E2/E3 | Physical; near-MP quality | 63–75 real MUL/sample | 5 ILA; learning 119 real MUL/sample example | Local LUT branches | Нет |
| Piecewise closed-loop | 28GHz-400MHz | Physical OTA; ACLR/EIRP gains | 935–27847 forward FLOPs; pruning | 323–54520 normalized learning FLOPs vs 2.7e9 ILA | Architecture-oriented, no common fixed-point table | Нет |
| Feature-selected residual PNN | FR3-DUT1 | Physical: best 1.08% EVM; worst ACLR −41.876 | 407–1768 FLOPs, 98–650 coeff | Huge 321,200 feature pool; time NR | Hardware NR | Нет |
| PN-RNN | PN-RNN-200MHz | Physical: −42.8 NMSE, −50 ACLR, 1.89% EVM | 1094 params, ops NR | ILC teacher, 2200 epochs | Hardware NR | Нет |
| OMP/GMP 17-term | LMBA-120MHz | Physical: ACPR −45/−46.1, EVM 1.2/0.9% | 17 coeff; exact ops depend terms | OMP selection; separate validation | MAC-friendly | Нет |
| DPD-NeuralEngine | ASIC-study | Physical RF data; −45.3 ACPR/−39.8 EVM | 1026 ops/sample | QAT 300 epochs | 22nm post-layout 250MS/s/195mW | Нет |

## 10. Нерешённые evidence gaps

1. Полные tables оригинальных Morgan GMP, Zhou direct-learning и части drift papers находятся за paywall; bibliographic claims подтверждены DOI, но точные RF numbers здесь намеренно не приведены.
2. Отдельный canonical block-RLS DPD source не верифицирован; требуется библиотечный доступ либо точное название от автора постановки.
3. OpenDPDv2 не сообщает left/right ACPR, end-to-end training time, stored/active memory traffic и real multiplication breakdown.
4. TCN-DPD не сообщает causal streaming latency и physical PA test.
5. SparseDPD не проводит physical closed-loop DPD measurement и использует один seed.
6. Работы по neural quantization в основном используют energy models/post-layout simulation, а не fabricated power measurements.
7. Ни одна рассмотренная внешняя работа не даёт честного apples-to-apples доказательства превосходства над OpenDPDv2 APA_200MHz.

Следовательно, literature evidence поддерживает complex local-support spline как крайне дешёвый baseline и SPH/SMP как минимальные memory extensions, но само заявление «лучше OpenDPD» возможно только после запуска обоих методов в одном исправленном evaluator и последующей физической PA verification.
