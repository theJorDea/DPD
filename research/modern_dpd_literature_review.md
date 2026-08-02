# Современные методы Digital Predistortion: обзор и выводы для проекта

Дата проверки: 2026-07-30
Охват: 89 проверенных первичных источников; 27 работ изучены подробно по
полному тексту, открытому manuscript/HTML, формулам и таблицам.
Машиночитаемая матрица: [`literature_matrix.csv`](literature_matrix.csv).

`not reported` означает, что значение не опубликовано или не подтверждено по
доступному первичному источнику. Числа не оцифровывались на глаз с графиков.

## 1. Что исследуется и что не является целью

Конечный объект проекта:

```text
desired x -> deployed DPD -> physical PA -> measured y
```

Основной результат — подавление нежелательного физического спектра при
ограниченном времени deployed DPD. PA model — только evaluator/calibration
tool; ограничение, описанное как «1000 real multiplications», к нему не
относится.

Формула \(E(f)<10^{-5}\) со слайда Huawei не используется как acceptance
criterion. В частности, \(-50\) dB NMSE не является требованием Huawei.
Пока точная spectral metric неизвестна, отдельно нужны:

- left/right adjacent leakage;
- absolute adjacent и integrated out-of-band power;
- main-band power change;
- PSD и worst spectral bin;
- EVM и auxiliary NMSE;
- peak, PAPR и clipping;
- RF harmonics \(2f_c/3f_c\), только если observation bandwidth их содержит.

## 2. Как классифицированы доказательства

| Класс | Что действительно было сделано | Допустимый вывод |
|---|---|---|
| `physical-DPD` | predistorted waveform подан в physical PA и выход измерен | физическая линеаризация в данном режиме |
| `physical-data/surrogate` | capture физического PA использован для fit, DPD scored через model | только model-mediated result |
| `simulation` | программный PA | алгоритмическая гипотеза |
| `hardware-real` | FPGA/DSP/ASIC выполнял путь и измерен | throughput/resources данного устройства |
| `hardware-post` | synthesis/post-layout/power simulation | implementation estimate |
| `transfer-only` | метод sequence/ML/control без DPD experiment | только механизм для ablation |
| `general-method` | фундаментальный метод без RF experiment | механизм/статистический baseline |
| `uncertain` | доступные первичные metadata не позволяют надёжно классифицировать setup | никаких physical/hardware claims |

В CSV используются контролируемые эквиваленты `physical-DPD`,
`physical-data/surrogate`, `simulation`, `hardware-real`, `hardware-post`,
`general-method` и `uncertain`; совместные классы соединены `+`. Статья может
иметь несколько классов: например, physical RF result и post-layout ASIC
estimate.

## 3. Почему результаты разных статей нельзя сложить в рейтинг

NMSE/ACLR зависят от PA, carrier, output back-off, bandwidth, waveform,
sample rate, observation receiver, alignment, gain и spectrum masks. Поэтому
сформированы отдельные strata:

1. `OpenDPDv2 APA_200MHz` — physical 3.5 GHz GaN Doherty.
2. `TCN DPA_200MHz` — frozen DGRU surrogate с fidelity \(-31.84\) dB.
3. `SparseDPD 20MHz` — physical capture, surrogate cascade, FPGA post-run.
4. `Spline E1/E2/E3` — три physical setups Campo et al.
5. `FR3 feature-selection` — physical 15 GHz DUT1/DUT2.
6. `PN-RNN/APNRRU` — собственные physical GaN Doherty setups.
7. `hardware reports` — отдельные PA и operation conventions.

Только внутри stratum допустим численный quality ranking. OpenDPDv2 не
«проигрывает» spline E1, а spline E1 не «проигрывает» TCN: это разные
эксперименты.

## 4. Исследовательские вопросы

1. Какая минимальная phase-equivariant структура покрывает short electrical
   memory?
2. Есть ли в остатке slow state, а не frame/reset artifact?
3. Может ли residual correlation предсказать spectral gain лучше random и не
   хуже exhaustive/DOMP?
4. Как разделить expensive calibration и cheap inference?
5. Как защитить physical PA от unsafe update?
6. Какой cost proxy действительно коррелирует с target latency?
7. Меняется ли ranking DPD между независимыми evaluators?
8. Какие современные AI-механизмы дают конкретный выигрыш, а какие только
   новое название классической adaptive filtering идеи?

## 4.1 Фактическая точка проекта, от которой строится roadmap

Аудит выполнен на `main` commit `91aedae` (`Add sealed fixed-point DPD
validation runner`), синхронизированном с `origin/main`.

Подтверждено кодом и frozen artifacts:

- контуры PA modeling и DPD deployment разделены;
- DPD test direction — `desired x -> DPD -> frozen PA evaluator`;
- splits, configs, checkpoints и результаты защищены manifests/hashes;
- текущий selected spline-memory DPD causal, phase-equivariant и
  streaming-compatible;
- float schedule трёх ветвей: 21 real MUL, 24 ADD, 1 `sqrt`, 6 LUT
  accesses/sample;
- fixed-point core и sealed validation runner существуют;
- DPA/APA spectral improvements получены только через frozen surrogate;
- pinned-host timing является diagnostic, не target-hardware acceptance.

Текущий блокер подтверждён собственными benchmark reports: causal GMP PA
evaluators имеют test fidelity около \(-35.385\) dB (DPA) и \(-38.608\) dB
(APA), а запас относительно остаточной DPD-ошибки лишь 5.521/5.867 dB,
меньше внутреннего 10-dB gate. Следовательно, дальнейшая fine optimization
DPD через тот же evaluator может использовать его ошибки.

Ни одна приведённая ниже архитектура не реализована в этом research-этапе;
frozen results не менялись.

## 5. Фундаментальные модели и learning architectures

### 5.1 ILA

Changsoo Eun и E. J. Powers,
[“A New Volterra Predistorter Based on the Indirect Learning Architecture”,
1997](https://doi.org/10.1109/78.552219), IEEE TSP.

Postdistorter обучается

\[
y/g\mapsto x
\]

и coefficients копируются в predistorter. Это законный calibration method
при условиях существования/единственности inverse. Но deployment test обязан
быть

\[
x_\mathrm{desired}\to D\to PA\to gx_\mathrm{desired}.
\]

Круг \(y_\mathrm{test}\to D_\mathrm{post}\to PA\to y_\mathrm{test}\) не
доказывает линеаризацию нового desired input.

### 5.2 Memory Polynomial

Lei Ding, G. Tong Zhou, Dennis R. Morgan, Zhengxiang Ma, J. Stevenson Kenney,
Jaehyeong Kim, C. R. Giardina,
[“A Robust Digital Baseband Predistorter Constructed Using Memory
Polynomials”, 2004](https://doi.org/10.1109/TCOMM.2003.822188), IEEE
Transactions on Communications.

\[
z[n]=\sum_{m,p\ \mathrm{odd}}a_{m,p}x[n-m]|x[n-m]|^{p-1}.
\]

Плюсы: linear-in-parameters, регулярный MAC, быстрый LS/RLS, fixed-point.
Минусы: branch explosion при большой memory/order; correlated bases.
Работа остаётся обязательным baseline, а не современной «устаревшей»
формальностью.

### 5.3 GMP

Dennis R. Morgan, Zhengxiang Ma, Jaehyeong Kim, Michael G. Zierdt,
John Pastalan,
[“A Generalized Memory Polynomial Model for Digital Predistortion of RF Power
Amplifiers”, 2006](https://doi.org/10.1109/TSP.2006.879264), IEEE TSP.

Добавляет lagging/leading envelope cross-terms:

\[
x[n-m]|x[n-m-d]|^{p-1}.
\]

Physical 2.14 GHz, 30-W PA, 11-carrier CDMA, около 15 MHz: MP с 20
coefficients дал spectral regrowth около 52 dB, GMP с 40 — около 59 dB в
собственной convention. Exact EVM/NMSE/runtime не опубликованы. Это
непосредственное обоснование branch-selection вместо полного GMP.

### 5.4 Direct Learning

Dayong Zhou, Victor E. DeBrunner,
[“Novel Adaptive Nonlinear Predistorters Based on the Direct Learning
Algorithm”, 2007](https://doi.org/10.1109/TSP.2006.882058), IEEE TSP.

DLA обновляет predistorter по cascade error, а не переносит postinverse.
Filtered-x/RLS-подобная learning path требует модели/оценки sensitivity PA.
Плюс — objective ближе deployment. Риск — instability и feedback noise.
Первичная статья в основном simulation; physical safe update следует
доказывать отдельно.

### 5.5 Separable inverse

Hong Jiang, Paul A. Wilford,
[“Digital Predistortion for Power Amplifiers Using Separable Functions”,
2010](https://doi.org/10.1109/TSP.2010.2049742),
[open copy](https://arxiv.org/abs/1306.0037).

Работа формализует условия совпадения postdistorter и predistorter. Journal
year — 2010; дата arXiv upload 2013 не должна заменять год публикации.
Теорема не оправдывает circular test при noninvertible compression,
distribution shift или совместной ошибке inverse/forward surrogates.

## 6. Volterra, DDR, CPWL, LUT и spline

### 6.1 Dynamic Deviation Reduction

Anding Zhu, Paul J. Draxler, Jonmei J. Yan, Thomas J. Brazil,
Donald F. Kimball, Peter M. Asbeck,
[“Open-Loop Digital Predistorter … Using Dynamic Deviation Reduction-Based
Volterra Series”, 2008](https://doi.org/10.1109/TMTT.2008.925211).

DDR организует Volterra terms по dynamic deviation, уменьшая число cross
terms. Это важный dictionary prior для short-memory residual branches.
Полная Volterra остаётся непрактичной; sparse selection и band limitation
обязательны.

### 6.2 Decomposed Vector Rotation / CPWL

Anding Zhu,
[“Decomposed Vector Rotation-Based Behavioral Modeling for Digital
Predistortion of RF Power Amplifiers”, 2015](https://doi.org/10.1109/TMTT.2014.2387853).

Modified CPWL сохраняет complex phase через rotation-restoration basis и
остаётся linear-in-parameters.

Physical ET GaN class-AB, 2.14 GHz, 20 MHz LTE:

- no DPD NRMSE 8.42%, ACPR 30.8/32.1 dBc;
- DVR, 84 coefficients: NRMSE 0.98%, ACPR 54.4/54.0 dBc.

Physical LDMOS Doherty, 60 MHz mixed GSM+LTE:

- DVR 95 coefficients;
- NRMSE 15.8% → 0.69%;
- LTE ACPR более чем +28 dB.

Это сильный CPWL baseline, но coefficient count не равен runtime.

### 6.3 Spline-interpolated LUT

Pablo Pascual Campo et al.,
[“Gradient-Adaptive Spline-Interpolated LUT Methods for Low-Complexity
Digital Predistortion”, 2021](https://doi.org/10.1109/TCSI.2020.3034825),
IEEE TCAS-I.

SPH:

\[
x\to\text{complex spline LUT}\to\text{short complex FIR}.
\]

SMP:

\[
z[n]=\sum_m x[n-m]C_m(|x[n-m]|).
\]

Три physical setups, включая 28 GHz OTA. Типовая стоимость:

| Model | Coefficients | Real MUL/sample |
|---|---:|---:|
| SPH | 14 | 40 |
| SMP | 31 | 63 |
| MP order 11, memory 4 | 24 complex | 112 |

На 28 GHz 200 MHz: no-DPD ACLR 26.3 dB; SPH/SMP/MP 34.1/34.4/35.0 dB.
Это самый близкий physical prior к текущему spline-memory core. Он не
сопоставим напрямую с APA_200MHz OpenDPD.

### 6.4 Piecewise closed-loop

Alberto Brihuega, Mahmoud Abdelaziz, Lauri Anttila, Matias Turunen,
Markus Allen, Thomas Eriksson, Mikko Valkama,
[“Piecewise Digital Predistortion for mmWave Active Antenna Arrays:
Algorithms and Measurements”, 2020](https://doi.org/10.1109/TMTT.2020.2994311),
[open copy](https://arxiv.org/abs/2003.06348).

Physical 64-element array, 28 GHz OTA, 400 MHz 5G NR 64-QAM; closed-loop
decorrelation, adaptive regions и pruning.

- starting ACLR до 21 dBc;
- около +4 dB usable EIRP;
- 348 → 96 basis functions с близкой линеаризацией;
- forward cost от 935 до 27,847 paper-FLOP/sample в зависимости от
  orthogonalization; pruned orthogonalized path 2219.

Важный вывод: self-orthogonalization ускоряет calibration, но может перенести
цену в forward path. Residual pruning уже известен.

## 7. Sparse regression и feature selection

### 7.1 DOMP/LASSO

Abdoul Barry, Wantao Li, Juan A. Becerra, Pere L. Gilabert,
[“Comparison of Feature Selection Techniques for Power Amplifier Behavioral
Modeling and Digital Predistortion Linearization”, 2021](https://doi.org/10.3390/s21175772),
Sensors.

Physical LMBA, 2 GHz, 4×LTE20 64-QAM, spread 120 MHz, 614.4 MS/s; separate
train/validation по 307,200 samples; initial GMP 979 terms.

При 17 selected terms:

| Method | ACPR range, dBc | EVM range |
|---|---:|---:|
| no DPD | −31.7 … −36.6 | 2.3–3.0% |
| DOMP | −45.0 … −46.1 | 0.9–1.2% |
| LASSO | −41.0 … −43.6 | 1.1–1.6% |

Это один из прямых prior к correlation-guided selection.

### 7.2 Doubly OMP

Juan A. Becerra, María J. Madero-Ayora, Javier Reina-Tosina,
Carlos Crespo-Cadenas, Javier García-Frías, Gonzalo Arce,
[“A Doubly Orthogonal Matching Pursuit Algorithm for Sparse Predistortion of
Power Amplifiers”, 2018](https://doi.org/10.1109/LMWC.2018.2845947).

DOMP выбирает следующий term по residual correlation после двойной
orthogonalization. Physical GaN/15 MHz OFDM evidence существует. Поэтому
«коррелировать residual с candidates и добавить лучший» — не новая идея.

### 7.3 Sparse Bayesian и robust power-varying selection

Jun Peng, Songbai He, Bingwen Wang, Zhijiang Dai, Jingzhou Pang,
[Sparse Bayesian DPD, 2016](https://doi.org/10.1109/TCSII.2016.2534718), и
Carlos Crespo-Cadenas, María J. Madero-Ayora, Juan A. Becerra, Sergio Cruces,
[robust sparse-Bayesian DPD, 2022](https://doi.org/10.1109/TMTT.2022.3157586).

Sparse Bayesian regression даёт posterior/relevance selection; robust variant
ориентирован на power-varying operation. Calibration тяжелее ridge, но
deployed active set может быть мал. Это обязательный baseline, если
correlation-guided method заявляет устойчивость по режимам.

### 7.4 Feature selection 2026

Cel Thys, Rodney Martinez Alonso, Ali H. Alsarraf, Dominique Schreurs,
Sofie Pollin,
[“Low Complexity Neural Network Digital Predistortion … through Feature
Selection”, 2026](https://arxiv.org/abs/2607.15441),
[dataset](https://doi.org/10.48804/306IIM).

Physical 15 GHz FR3 PA, 100 MHz 64-QAM, 1.28 GS/s. Offline dictionary:
321,200 real Volterra/GMP-like features; LASSO → около 2000, MRMR → 100–200,
затем residual phase-normalized NN.

DUT1:

| Model | Coefficients | FLOP/sample | inverse val NMSE | physical EVM | physical worst ACLR |
|---|---:|---:|---:|---:|---:|
| PNN | 128 | 360 | −32.80 dB | 1.88% | −39.95 dBc |
| proposed | 98 | 407 | −35.38 dB | 1.38% | −43.81 dBc |
| proposed | 650 | 1768 | −41.16 dB | 1.08% | −41.88 dBc |

Большой offline search не входит в runtime. Selected real I/Q features не
гарантируют exact phase-equivariance. Это наиболее сильный современный
competitor для rich-library selection.

## 8. Adaptive/closed-loop DPD и observation receiver

### 8.1 ILC

Jessica Chani-Cahuana, Per Niklas Landin, Christian Fager, Thomas Eriksson,
[“Iterative Learning Control for RF Power Amplifier Linearization”,
2016](https://doi.org/10.1109/TMTT.2016.2588483).

ILC повторно изменяет waveform, чтобы physical PA output приблизился к target,
затем deployable model может быть fitted к teacher waveform. Это сильный
teacher при heavy compression/noise, но требует повторяемых acquisitions.
Стоимость ILC — calibration, не sample inference.

Maarten Schoukens, Jules Hammenecker, Adam Cooman,
[“Obtaining the Preinverse of a Power Amplifier Using Iterative Learning
Control”, 2017](https://doi.org/10.1109/TMTT.2017.2694822),
[open copy](https://arxiv.org/abs/1606.08663), усиливает это основание.

### 8.2 Concurrent decorrelation

Mahmoud Abdelaziz, Lauri Anttila, Adnan Kiayani, Mikko Valkama,
[“Decorrelation-Based Concurrent Digital Predistortion With a Single Feedback
Path”, 2018](https://doi.org/10.1109/TMTT.2017.2706688).

Physical base-station и commercial LTE-A mobile PA; один feedback receiver.
Coefficients обновляются для декорреляции residual error и nonlinear basis.
Следовательно, observation receiver и residual decorrelation сами по себе
известны.

### 8.3 Dynamic parameter selection

Quynh Anh Pham, Gabriel Montoro, David Lopez-Bueno, Pere L. Gilabert,
[“Dynamic Selection and Estimation of the Digital Predistorter Parameters for
Power Amplifier Linearization”, 2019](https://doi.org/10.1109/TMTT.2019.2923186).

Active parameter set меняется по residual contribution. Это прямой prior к
advisor, который добавляет/удаляет branches; отличие проекта должно быть в
cross-frame statistics, cost model и safety protocol.

### 8.4 Observation bandwidth

N. Hammler, A. Cathelin, P. Cathelin, B. Murmann,
[“A Spectrum-Sensing DPD Feedback Receiver With 30× Reduction in ADC
Acquisition Bandwidth and Sample Rate”, 2019](https://doi.org/10.1109/TCSI.2019.2920828).

Это hardware prior для slow observation path. Он показывает, что controller
не обязан digitize весь RF bandwidth непрерывно, но reduced observation
должен соответствовать конкретной spectral metric.

Ahmed Ben Ayed, Eric Ng, Patrick Mitran, Slim Boumaiza,
[low-bit observation ADC DPD, 2021](https://doi.org/10.1109/ACCESS.2021.3096978)
подтверждает другой trade-off feedback fidelity/cost.

### 8.5 Safe RL

Christian Spano, Damiano Badini, Lorenzo Cazzella, Matteo Matteucci,
[“Local and Remote Digital Pre-Distortion for 5G Power Amplifiers with Safe
Deep Reinforcement Learning”, 2025](https://doi.org/10.3390/s25196102),
[open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12527039/).

Huawei Milan hardware test. CRE-DDPG:

- ограничивает действие при приближении ACLR к safety threshold;
- хранит actions и выполняет reverse/recovery при нарушении;
- 2000 episodes; improvement примерно после 850 updates;
- reported 25 runs safe/improved к не более чем 950 updates;
- для exploration предложен TDD switch с antenna на receiver path.

Значит safety threshold и rollback уже не новы. Возможное преимущество
deterministic advisor — 1–несколько auditable candidate fits, независимый
shadow capture, multi-metric/worst-bin gates и hard cost, но это ещё нужно
экспериментально показать.

## 9. Neural DPD

### 9.1 Residual TDNN

Yibo Wu, Ulf Gustavsson, Alexandre Graell i Amat, Henk Wymeersch,
[“Residual Neural Networks for Digital Predistortion”, 2020](https://arxiv.org/abs/2005.05655).

R2TDNN учит nonlinear residual вокруг identity shortcut и проверен на physical
PA. Это прямое основание tiny residual path, но не основание пропускать
structured branch ablation.

### 9.2 Phase-normalized NN/RNN

Arne Fischer-Bühner, Lauri Anttila, Manil Dev Gomony, Mikko Valkama:

- [PN-NN, 2023](https://doi.org/10.1109/LMWT.2023.3290980);
- [PN-RNN, 2024](https://doi.org/10.1109/LMWT.2024.3393859).

Physical QPA3503 GaN Doherty, 3.5 GHz, 200 MHz:

| Model | Parameters/state | NMSE | ACLR | EVM |
|---|---:|---:|---:|---:|
| no DPD | — | −20.2 dB | −26.1 dBc | 8.40% |
| PN-RNN | 1094 / 13 | −42.8 dB | −50.0 dBc | 1.89% |
| ILC teacher | waveform | −51.9 dB | −57.9 dBc | 1.81% |

Польза phase normalization и compact state подтверждена; operation count и
fixed-point этой PN-RNN не опубликованы.

### 9.3 APNRRU

Arne Fischer-Bühner, Lauri Anttila, Matias Turunen, Manil Dev Gomony,
Mikko Valkama,
[“Augmented Phase-Normalized Recurrent Neural Network for RF Power Amplifier
Linearization”, 2025](https://doi.org/10.1109/TMTT.2024.3484581).

Три physical GaN Doherty PA, 983.04 MS/s, до 400 MHz noncontiguous 256-QAM.
APNRRU сочетает phase normalization, input FIR, compact residual recurrence и
envelope states.

Около 1031 parameters:

- 1187 real MUL;
- 1153 ADD;
- 39 nonlinear activations;
- 1 phase decomposition/sample.

320 MHz: NMSE −40.62 dB, worst ACLR −49.14 dBc, EVM 0.93%.
400 MHz noncontiguous: −35.93 dB, −45.24 dBc, 1.24%.

Это сильный teacher/reference, но published path уже превышает условный
1000-MUL proxy до учёта nonlinear/memory latency. Авторы обучали пять
экземпляров и публиковали лучший, а не mean/std; эти числа не доказывают
robustness по random seeds.

### 9.4 MoE

Arne Fischer-Bühner, Alberto Brihuega, Lauri Anttila, Matias Turunen,
Vishnu Unnikrishnan, Manil Dev Gomony, Mikko Valkama,
[“Sparsely Gated Mixture of Experts Neural Network for Linearization of RF
Power Amplifiers”, 2024](https://doi.org/10.1109/TMTT.2023.3341616).

Physical 1.8/3.5 GHz. На 100 MHz:

- \(N=4,K=2\): 3960 stored / 2012 active parameters, NMSE −41.8 dB,
  ACLR −49.1 dBc, EVM 1.60%;
- \(N=4,K=1\): 3960/1038, −41.0/−47.9/1.61%.

Generic MoE/top-K routing уже применён. Для проекта frame-rate hard routing
предпочтительнее sample-level soft/top-K: исполняется один cheap expert и
switch можно защитить hysteresis.

### 9.5 OpenDPDv2

Yizhuo Wu, Ang Li, Chang Gao,
[“OpenDPDv2: A Unified Learning and Optimization Framework for Neural Network
Digital Predistortion”, 2025](https://arxiv.org/abs/2507.06849).

Physical 3.5 GHz GaN Doherty, 200 MHz TM3.1a 256-QAM, 983.04 MS/s,
98,304 samples, 60/20/20; end-to-end training через frozen PA surrogate,
финальный physical measurement.

| Model | Params | Paper FLOP/sample | NMSE | EVM | Avg ACPR |
|---|---:|---:|---:|---:|---:|
| no DPD | — | — | −20.5 | −24.7 | −28.3 |
| TRes-GRU | 999 | 1282 | −38.4 | −41.2 | −59.0 |
| TRes-DeltaGRU | 999 | 1324 | −39.6 | −42.1 | −59.9 |

W16/W12 и temporal sparsity изучены. Hardware energy — gem5/7-nm model, не
fabricated silicon. Left/right ACPR, training wall-time и full real-operation
breakdown не опубликованы. Это главный apples-to-apples neural reference
только на APA_200MHz.

### 9.6 TCN-DPD

Huanqiang Duan, Manno Versluis, Qinyu Chen, Leo C. N. de Vreede, Chang Gao,
[TCN-DPD, 2025](https://arxiv.org/abs/2506.12165),
[IMS DOI](https://doi.org/10.1109/IMS40360.2025.11103923).

OpenDPD DPA_200MHz, frozen DGRU PA fidelity −31.84 dB, five seeds.
TCN-500: NMSE \(-44.61\pm1.37\) dB, ACPR L/R
\(-51.58\pm2.84/-49.26\pm2.04\) dBc, EVM \(-47.52\pm1.49\) dB.

Опубликованные dilated convolutions noncausal. Physical PA, fixed-point и
complete operation count отсутствуют. Высокий surrogate NMSE ниже evaluator
fidelity — явный риск model exploitation.

### 9.7 DeltaDPD

Yizhuo Wu et al.,
[DeltaDPD, 2025](https://arxiv.org/abs/2505.06250),
[DOI](https://doi.org/10.1109/LMWT.2025.3565004).

Physical 3.5 GHz GaN Doherty, 200 MHz. При 52% temporal sparsity:

- 573 parameters;
- NMSE −37.22 dB;
- EVM −38.52 dB;
- ACPR −50.03 dBc;
- 589 MUL, 2005 ADD, 710 memory accesses/sample;
- 6.41 nJ — estimated 7-nm energy.

Эта работа особенно полезна тем, что отделяет MUL/ADD/memory. Average temporal
sparsity не гарантирует worst-case deadline.

### 9.8 SparseDPD

Manno Versluis, Yizhuo Wu, Chang Gao,
[SparseDPD, 2025](https://arxiv.org/abs/2506.16591),
[FPL DOI](https://doi.org/10.1109/FPL68686.2025.00031).

Physical capture: 3.5 GHz GaN Doherty, 20 MHz 64-QAM, 172,035 samples.
PNTDNN 2×12, QAT + pruning:

- 64 stored parameters;
- 74% zeros;
- 72 reported operations/sample;
- Q1.13 input/weights/activations;
- surrogate cascade NMSE −48.2, EVM −54.0, ACPR −59.4;
- Zynq-7Z010 post-implementation: 170 MS/s, 66 DSP, 13 BRAM,
  405 mW total estimate.

RF score — surrogate result; FPGA — post-implementation simulation.
Unstructured zeros требуют zero-skipping engine.

## 10. Hardware-oriented evidence

### 10.1 Parallel FPGA DPD

Hai Huang, Jingjing Xia, Slim Boumaiza,
[“Novel Parallel-Processing-Based Hardware Implementation of Baseband
Digital Predistorters for Linearizing Wideband 5G Transmitters”,
2020](https://doi.org/10.1109/TMTT.2020.2993236).

Physical 28 GHz/OTA evidence, ZCU102, pruned CRV/GMP-family:

- 2.4 GS/s для 400 MHz signal;
- 300 MHz clock;
- около 0.96 W;
- own-experiment ACPR −44.7 dBc, EVM −39.2 dB.

Это сильное hardware-real доказательство: regular parallel datapath может
обработать широкий сигнал. Абсолютный RF result не сравнивается с OpenDPD.

### 10.2 Mixed precision

Yizhuo Wu et al.,
[“MP-DPD”, 2024](https://arxiv.org/abs/2404.15364),
[DOI](https://doi.org/10.1109/LMWT.2024.3386330).

160 MHz 1024-QAM; 502-param GRU:

- 502 MUL, 1417 ADD, 506 MEM/sample;
- FP32 ACPR −43.36/−45.30, EVM −38.46 dB;
- W16A16 −43.75/−45.27, −38.72 dB;
- 7-nm energy — model estimate.

16-bit robustness одной GRU не доказывает 12/14/16-bit suitability spline;
нужен bit-true local sweep.

### 10.3 DPD-NeuralEngine

Ang Li, Haolin Wu, Yizhuo Wu, Qinyu Chen, Leo C. N. de Vreede, Chang Gao,
[DPD-NeuralEngine](https://arxiv.org/abs/2410.11766),
[ISCAS DOI](https://doi.org/10.1109/ISCAS56072.2025.11043563).

Physical GaN Doherty RF data; GF22FDX post-layout:

- W12A12, 502 parameters;
- 1026 operations/sample;
- 2 GHz, 250 MS/s, 7.5 ns;
- 195 mW, 0.2 mm²;
- ACPR −45.3 dBc, EVM −39.8 dB.

ASIC не fabricated; paper имеет 80-MHz RF / 60-MHz hardware-table ambiguity.

## 11. Adaptation по режимам

### 11.1 Real-time model switching

Yue Li, Xiaoyu Wang, Anding Zhu,
[2022](https://doi.org/10.1109/TMTT.2021.3132347).

Decision tree выбирает cross-term branch/coefficient set; active одна ветвь.
Physical PA/FPGA evidence. Это прямой prior к conditional computation.
Frame-rate router проекта должен сравниваться с этой схемой и простой LUT.

### 11.2 Temperature

Gautam Jindal, Gavin T. Watkins, Kevin Morris, Tommaso A. Cappello,
[2022](https://doi.org/10.1109/TMTT.2022.3175155).

3.75 GHz, 10-W GaN-on-SiC class-B, 20–80°C, несколько PAPR/bandwidth.
Temperature-aware DPD улучшал ACPR примерно на 1–4 dB относительно
temperature-less model в собственном experiment. Это physical motivation, но
\(q_\beta\) нельзя называть температурой без sensor/correlation experiment.

### 11.3 Continual learning

Yucheng Yu, Peng Chen, Xiao-Wei Zhu, Jianfeng Zhai, Chao Yu,
[2022](https://doi.org/10.1109/TMTT.2022.3210199).

Physical Doherty, changing power/bandwidth/waveform; retention/transfer/merge
states. Поддерживает slow control-plane и coefficient memory, но не требует
continual training на каждый sample.

### 11.4 Transfer-learning adapter

Feridoon Jalili, Felice Francesco Tafuri, Ole Kiel Jensen, Qingyue Chen,
Ming Shen, Gert F. Pedersen,
[2023](https://doi.org/10.1109/ACCESS.2023.3242648).

Physical OTA 4×4 APA, 20–100 MHz, upconversion до 28 GHz. 100 MHz:
ACLR improvement 8.5 dB, EVM improvement 8.6 percentage points.

Headline 199,168 → 160 MUL относится только к added adaptation layer; frozen
20-MHz backbone также исполняется. Работа доказывает дешевое обновление, не
160-MUL полный DPD.

## 12. Современные state-space/LLM идеи

Первичные mechanism sources:

- S4: [Gu, Goel, Ré, 2021](https://arxiv.org/abs/2111.00396);
- Mamba: [Gu, Dao, 2023](https://arxiv.org/abs/2312.00752);
- HyperNetworks: [Ha, Dai, Le, 2016](https://arxiv.org/abs/1609.09106);
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685);
- distillation: [Hinton, Vinyals, Dean, 2015](https://arxiv.org/abs/1503.02531);
- sparse MoE: [Shazeer et al., 2017](https://arxiv.org/abs/1701.06538);
- MAML: [Finn, Abbeel, Levine, 2017](https://proceedings.mlr.press/v70/finn17a.html);
- deep Koopman: [Lusch, Kutz, Brunton, 2018](https://doi.org/10.1038/s41467-018-07210-0);
- Neural ODE: [Chen et al., 2018](https://arxiv.org/abs/1806.07366).

Прямые выводы:

- full Transformer/attention неприемлем для sample path;
- full Mamba уже имеет DPD prior APN-Mamba
  ([2026 DOI](https://doi.org/10.1016/j.jestch.2026.102408)), но нет
  подтвержденного operations/fixed-point/hardware Pareto;
- полезен simplified diagonal state, сравниваемый с \(q_\beta\) и FIR;
- hypernetwork допустим только как slow coefficient generator;
- LoRA бессмыслен для десятков coefficients без доказанного low-rank drift;
- distillation/ILC полезны в calibration path;
- retrieval — обычный verified coefficient bank;
- speculative inference не переносится; shadow validation переносится;
- sample-level soft MoE хуже frame hard routing для hard real-time.

Полный разбор находится в
[`ai_methods_transfer_to_dpd.md`](ai_methods_transfer_to_dpd.md).

## 13. Спектральное обучение

Полезна multi-objective loss:

\[
\begin{aligned}
L={}&\lambda_tL_\mathrm{time}
+\lambda_LL_\mathrm{left}
+\lambda_RL_\mathrm{right}
+\lambda_oL_\mathrm{OOB}\\
&+\lambda_eL_\mathrm{EVM}
+\lambda_pL_\mathrm{peak}
+\lambda_mL_\mathrm{main}
+\lambda_cL_\mathrm{cost}.
\end{aligned}
\]

Differentiable FFT позволяет gradient, но не решает:

- падение main-band power;
- worst-frame leakage;
- узкий spike вне интегральной полосы;
- wrong window/mask;
- adversarial surrogate drive;
- physical RF harmonics вне baseband capture.

Нужны explicit main-band constraint, percentile/worst-frame objective,
spectral mask max penalty, peak guard и independent evaluator. Spectral
fine-tuning не разрешён через текущий недостаточно точный единичный surrogate.

## 14. Ближайший prior art к residual controller

Уже известны:

- residual decorrelation;
- OMP/DOMP correlation selection;
- sparse Bayesian/group-LASSO;
- dynamic active set;
- piecewise pruning;
- operating-point model switching;
- MoE/configuration-conditioned coefficients;
- safe RL threshold и recovery;
- observation receiver с reduced bandwidth/bit width.

Поэтому самостоятельный вклад может быть только в проверенной комбинации:

1. phase-equivariant local spline main path;
2. causal group dictionary;
3. QR/SVD partial complex scores после PA/spectral sensitivity
   \(J_P(z)\Phi_G\); raw correlation только diagnostic baseline;
4. cross-frame stability, whole-frame/capture null и max-statistic/FDR;
5. explicit branch timing cost;
6. independent spectral shadow capture;
7. main-band/EVM/peak/PAPR/worst-bin constraints;
8. atomic coefficient-bank swap/rollback;
9. physical drift experiment;
10. сравнение с DOMP/group-LASSO/dynamic selection/safe RL при равном
    acquisition и deployed-cost budget.

До этого формулировка — «потенциально новая комбинация известных методов».

## 15. Pareto-выводы

| Семейство | Качество | Fast-path cost | Calibration | Fixed-point | Главный риск |
|---|---|---|---|---|---|
| MP/GMP | сильный baseline | растёт с terms | быстрый LS/RLS | structurally favorable | basis explosion |
| spline/SPH/SMP | близко MP при малой цене в physical priors | очень низкий | LS/LMS | structurally favorable | memory coverage |
| sparse GMP/CPWL | хороший Pareto | выбранные terms | дорогой search | favorable but model-specific | unstable selection |
| PN-RNN/APNRRU | сильное physical качество | ~1k+ MUL и nonlinear | долго/ILC | возможно | timing/memory |
| TCN | сильный surrogate | regular conv, но buffer | SGD | возможно | noncausal/no physical |
| temporal sparse GRU | quality/energy trade-off | variable | SGD | показано 12–16 bit | worst-case work |
| expert bank | режимная robustness | один expert при hard routing | per regime | structurally favorable | switching/OOD |
| hypernetwork | cheap fast path возможен | base DPD | heavy slow path | fast path requires proof | unsafe extrapolation |
| diagonal state | потенциально длинная memory | десятки MUL | простой | requires state-bound proof | не лучше one-pole |
| reservoir | flexible | сотни тысяч MUL при dense R=600×2 | ridge readout | плохо для budget | recurrence dominates |

## 16. Рекомендация для проекта

1. **Первый путь:** structured spline-memory + correlation-guided sparse
   residual branches.
2. **Второй:** slow-state spline или frame-rate coefficient bank после
   controlled state evidence.
3. **Высокий риск:** slow hypernetwork/low-rank generator либо tiny diagonal
   state.

Первый эксперимент — observer-only на frozen validation:

- residual correlations;
- block null/bootstrap;
- frame stability;
- artifact controls;
- cost annotation;
- evaluator disagreement.

Никакого fit и test opening на этом шаге. Если нет устойчивой residual
structure, усложнение DPD прекращается. Если есть, advisor предлагает ровно
одну branch, а её spectrum проверяется независимо.

## 17. Нерешённые вопросы

От Huawei нужны:

- exact spectral bands/components;
- absolute или dBc reference;
- required suppression;
- main-band/EVM/peak limits;
- RBW/VBW/window/detector/averaging;
- target sample rate/clock/hardware/numeric format;
- exact reference 1000-MUL timing kernel;
- allowed latency/update cadence;
- observation receiver specification.

Для научного результата нужны:

- high-fidelity independent PA evaluator;
- controlled physical captures по power/temperature/bandwidth;
- paired repeated no-DPD/DPD physical measurement;
- target-hardware timing;
- untuned sealed physical test.

До этого нельзя утверждать превосходство над OpenDPD, физическую
линеаризацию или соответствие Huawei.
