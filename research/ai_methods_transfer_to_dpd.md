# Перенос современных AI, state-space и adaptive-control идей в DPD

Дата: 2026-07-30
Статус: source-backed design analysis; новые модели не реализованы.

## 1. Критерий полезности

Идея считается применимой к DPD только если можно явно ответить:

1. что вычисляется на каждом complex sample;
2. что вычисляется редко в calibration/control path;
3. causal ли deployed path;
4. сколько состояния и coefficient memory требуется;
5. как соблюдается либо осознанно нарушается phase-equivariance;
6. как выполняется fixed-point;
7. какое физическое PA evidence существует;
8. чем идея лучше простой таблицы коэффициентов, spline/GMP или short FIR.

Архитектурное сходство с LLM не является доказательством пригодности при
сотнях миллионов отсчётов в секунду.

Используются три категории:

- **прямо применимо** — механизм уже имеет DPD evidence или добавляет малую
  и измеримую стоимость;
- **применимо после серьёзной адаптации** — механизм полезен только вне
  sample-rate path либо после сильного структурирования;
- **аналогия без практического преимущества** — цена механизма превышает
  реалистичную пользу.

## 2. Что уже показали neural DPD

### 2.1 Полносвязные и time-delay сети

RVFTDNN/TDNN доказали, что real-valued сеть над delayed I/Q может моделировать
нелинейную память, но dense layers требуют работы для всех weights. Residual
R2TDNN ([Wu et al., 2020](https://arxiv.org/abs/2005.05655)) обучает только
отклонение от identity и имеет физический PA experiment. Это полезный training
bias, но не отменяет dense cost.

Вердикт: tiny residual network имеет смысл только после structured path и
только при сравнении с дополнительными GMP/spline branches той же latency.

### 2.2 GRU/LSTM/RNN

GRU, JANET и LSTM компактно представляют память, но gates требуют нескольких
matrix-vector products и nonlinear activations. OpenDPDv2
([Wu, Li, Gao, 2025](https://arxiv.org/abs/2507.06849)) и DeltaDPD
([Wu et al., 2025](https://arxiv.org/abs/2505.06250)) показывают сильное
physical-PA качество, однако:

- temporal sparsity — средняя, waveform-dependent величина;
- dense kernels не ускоряются от нулевых deltas без специального datapath;
- worst-case latency и memory traffic нельзя заменить active-parameter count;
- конкретные OpenDPD backbones/configs нельзя автоматически считать
  streaming-ready: для каждого отдельно проверяются look-ahead, framing,
  reset/warm-up и state continuity.

Вердикт: хороши как teacher/reference; fast-path winner не доказан.

### 2.3 Phase normalization

Phase-normalized TDNN/RNN:

- [PN network, 2023](https://doi.org/10.1109/LMWT.2023.3290980);
- [PN-RNN, 2024](https://doi.org/10.1109/LMWT.2024.3393859);
- [APNRRU, 2025](https://doi.org/10.1109/TMTT.2024.3484581).

Они нормализуют глобальную фазу и обучают amplitude/memory correction.
Физические эксперименты показывают, что это сильный inductive bias. APNRRU
даёт явный ориентир цены: конфигурация около 1031 параметра требует
приблизительно 1187 real MUL, 1153 ADD, 39 activations и phase decomposition
на sample. Это уже выше условного arithmetic proxy проекта до учёта памяти и
nonlinear latency.

Вердикт: **прямо применим принцип эквивариантности**, но не обязательно сама
рекуррентная сеть. Форма \(z=xC(|x|,\text{state})\) реализует принцип дешевле.

### 2.4 TCN/CNN

[TCN-DPD](https://arxiv.org/abs/2506.12165) показывает сильные surrogate
metrics при 200 MHz, но опубликованная dilated TCN noncausal. Causal version
потребует либо отказа от future context, либо явного look-ahead/latency.
Convolution regular для FPGA, однако buffer и activation cost остаются.

Вердикт: causal tiny convolution возможна как residual ablation; исходную
TCN нельзя называть готовой real-time архитектурой.

### 2.5 Sparse networks

[SparseDPD](https://arxiv.org/abs/2506.16591) показывает QAT и FPGA
post-implementation mapping, но unstructured zero имеет стоимость ноль только
при zero-skipping hardware. Structured sparse ETDNN
([Tanio et al., 2020](https://doi.org/10.1109/ACCESS.2020.3005146)) переносится
лучше: целые neurons/groups можно исключить.

Вердикт: structured sparsity — прямо применима; unstructured pruning без
конкретного engine — только parameter compression.

## 3. State-space модели

### 3.1 S4

[Structured State Spaces for Sequence Modeling](https://arxiv.org/abs/2111.00396)
решает long-sequence задачи через специально параметризованный linear state и
convolution/recurrent duality. Это не DPD paper.

Полный S4 включает complex/structured kernels и offline convolutional training.
Для DPD полезен более узкий принцип: несколько устойчивых diagonal states

\[
h_i[n]=a_i h_i[n-1]+b_i^Tu[n]
\]

с bounded \(a_i\). Такой recurrence causal, потоковый и fixed-point-friendly.
После quantization требуется \(|a_i|\le1-\epsilon\), bounded input,
рассчитанный state/accumulator bound и zero-input limit-cycle test.

Вердикт: **применимо после сильного упрощения**. Научный вопрос — даёт ли
compact state лучшее spectrum/cost, чем набор one-pole \(q_\beta\) и short
FIR. Если нет, SSM terminology ничего не добавляет.

### 3.2 Mamba/selective state space

[Mamba](https://arxiv.org/abs/2312.00752) делает state parameters
input-dependent. Это полезно для языковых последовательностей, но в DPD
sample-level selection требует projections, gates, exponentials/softplus и
state updates на каждом sample.

Математическая recurrent форма Mamba может быть causal. Однако на дату обзора
существующая DPD-работа APN-Mamba
([official DOI](https://doi.org/10.1016/j.jestch.2026.102408)), но одного
публикационного результата недостаточно для вывода о её конкретных streaming
boundary semantics, fixed-point degradation и hardware throughput в нашем
режиме. Её нужно сравнивать с diagonal state и APNRRU по реальному
operation/memory count.

Вердикт: full Mamba — **не первый fast-path кандидат**. Допустим только
маленький selective-state ablation после linear-state baseline.

### 3.3 Neural ODE

[Neural ODE](https://arxiv.org/abs/1806.07366) моделирует непрерывную динамику,
но adaptive numerical solver имеет variable latency, сложный fixed-point и
непредсказуемый worst case. Физические thermal states можно описать обычным
discrete one-pole update гораздо дешевле.

Вердикт: полезная offline modeling analogy; практически бесполезна в deployed
sample path.

### 3.4 Koopman

Deep Koopman representations
([Lusch, Kutz, Brunton, 2018](https://doi.org/10.1038/s41467-018-07210-0))
ищут observable space с линейной динамикой. Для DPD это мотивирует learned
invariant features + small linear state, но encoder может быть дороже самой
модели.

Вердикт: применимо после адаптации, если observables заранее структурированы
как \(|x|^2,\Delta|x|^2,q_\beta\). Произвольный neural encoder пока не
обоснован.

## 4. Mixture of experts и conditional computation

### 4.1 Что уже известно в DPD

Attention-guided memory-polynomial network
([Cioba et al., 2020](https://arxiv.org/abs/2003.13361)) смешивает локальные
AOMPM experts на sample level. Sparsely gated MoE для PA linearization
([Fischer-Bühner et al.](https://doi.org/10.1109/TMTT.2023.3341616)) и
real-time model switching
([Li, Wang, Zhu](https://doi.org/10.1109/TMTT.2021.3132347)) являются прямыми
prior art.

Следовательно, «использовать MoE» не является новой идеей.

### 4.2 Рекомендуемая адаптация

Router выбирает один coefficient bank раз в кадр:

\[
e_t=\arg\max_e\left(
a_e^T[s_t^\mathrm{known\ before\ frame},s_{t-1}^\mathrm{feedback}]+b_e
\right).
\]

Преимущества:

- выполняется только один expert;
- router amortized по кадру;
- switch можно ограничить границей кадра;
- fixed-point path совпадает с обычной таблицей коэффициентов.

Нужно доказать преимущество над nearest operating-point LUT. Heterogeneous
soft experts с sample-rate routing вычисляют несколько paths и создают
variable work. Но при одинаковой topology и frame-static weights можно
однократно смешать coefficients и выполнить один DPD — это обязательный
baseline. Sample-level switch может породить новые spectral components.

Вердикт: frame-rate hard expert selection — прямо применимо при нескольких
режимах. Soft token-like MoE — не рекомендуется.

## 5. Hypernetworks, adapters и low-rank updates

### 5.1 Hypernetwork

[HyperNetworks](https://arxiv.org/abs/1609.09106) генерируют weights другой
сети. Для DPD генератор должен быть медленным:

\[
\theta_{t+1}=H_\phi(s_{\le t}),\qquad
z[n]=D_{\theta_t}(x)[n].
\]

Fast path остаётся spline/GMP. Generator может работать на control processor
раз в кадр. Feedback текущего кадра влияет только на следующий; coefficients
обязательно проходят projection, shadow validation и atomic bank
swap/rollback. Statistics должны быть rotation-invariant.

Вердикт: применимо после серьёзной адаптации; полезно только при наличии
multi-condition captures.

### 5.2 LoRA

[LoRA](https://arxiv.org/abs/2106.09685) экономит adaptation больших dense
матриц. У spline/GMP десятки коэффициентов, поэтому factorization
\(\Delta W=AB\) часто хранит не меньше исходного update.

Полезная версия:

\[
\theta_c=\theta_0+Uh_c,\qquad r=\dim h_c\ll\dim\theta,
\]

если SVD коэффициентов по controlled regimes действительно показывает
low-rank drift.

Вердикт: direct LoRA analogy почти бесполезна; empirical low-rank coefficient
subspace может быть полезен для safe adaptation.

### 5.3 Adapters

Маленькая additive branch, обучаемая для нового operating point, является
DPD-аналогом adapter. В нашей задаче это уже sparse residual branch.

Вердикт: прямо применимо, но новизна определяется physical protocol, а не
термином adapter.

## 6. Distillation и teacher–student

[Knowledge Distillation](https://arxiv.org/abs/1503.02531) переносит поведение
тяжёлого teacher в маленький student. DPD-specific evidence включает
Walsh-domain cross-domain distillation
([Thys et al., 2024](https://arxiv.org/abs/2402.09964)) и physical ILC teacher
в phase-normalized работах.

Лучший teacher для physical DPD — ILC waveform или хорошо контролируемая
feedback optimization, а не один PA surrogate. Student остаётся
structured/causal.

Риски:

- teacher может иметь noncausal look-ahead;
- student копирует unsafe peak;
- surrogate teacher передаёт model error;
- хорошая imitation loss не гарантирует spectral result.

Вердикт: прямо применимый calibration mechanism, не deployed architecture.

## 7. Meta-learning и continual learning

### 7.1 Meta-learning

[MAML](https://proceedings.mlr.press/v70/finn17a.html) ищет initialization для
быстрой адаптации. DPD-specific meta-learning:
[Falempin et al., 2022](https://doi.org/10.1109/TBC.2022.3204229) и
[CCNC 2022 work](https://doi.org/10.1109/CCNC49033.2022.9700529).

Сильный эксперимент должен сравнить:

- meta-initialization;
- warm-start предыдущими коэффициентами;
- nearest operating-point bank;
- direct ridge/RLS на том же числе calibration samples.

Без этого «быстрая адаптация» может быть обычным warm start.

Вердикт: применимо в slow path; physical evidence пока слабее classical
adaptive DPD.

### 7.2 Continual learning

[Continual-learning DPD](https://doi.org/10.1109/TMTT.2022.3210199) прямо
рассматривает изменения power/bandwidth/waveform и забывание. Для проекта
важнее coefficient bank + replay/regularization, чем большой continual NN.

Вердикт: механизмы retention и regime memory применимы; sample-level training
не требуется.

### 7.3 In-context adaptation

LLM in-context learning не даёт переносимого механизма без большого
sequence model. Практический аналог — оценить operating statistics текущего
кадра и выбрать/интерполировать коэффициенты.

Вердикт: красивая аналогия; использовать термины system identification и
frame-rate adaptation.

## 8. Retrieval и external memory

«Retrieval» в DPD сводится к:

- key: power, temperature, bandwidth, PA id;
- value: проверенный coefficient bank;
- nearest-neighbor lookup;
- safe fallback при большом расстоянии.

Это полезная инженерная coefficient database. Она не требует vector database,
attention или LLM. Memory bandwidth и atomic bank switch нужно измерять.

Вердикт: прямо применимо как LUT по режимам; LLM-формулировка ничего не
добавляет.

## 9. Speculative execution

Speculative decoding ускоряет autoregressive model, когда дешёвый draft
предлагает токены, а teacher проверяет их блоком. DPD output нельзя задерживать
до проверки тяжёлым teacher на каждый sample.

Допустимая аналогия — shadow coefficient bank, проверяемый на следующем
calibration capture. Это safe deployment protocol, а не speculative
sample-level inference.

Вердикт: прямая LLM идея непрактична; asynchronous shadow validation полезна.

## 10. Dynamic sparsity и gating

Delta updates могут экономить энергию при медленно меняющемся input/state, но
нужны:

- worst-case active fraction;
- zero-skipping datapath;
- comparison/index/cache cost;
- отсутствие variable-latency deadline misses;
- waveform/power robustness.

Для hard real-time предпочтительнее static structured sparsity: заранее
известно, какие branches/taps отсутствуют.

Вердикт: static structured sparsity — прямо применима; average dynamic sparsity
— дополнительная optimization, не gate.

## 11. Pruning, tensor decomposition и low-rank neural layers

- neuron/channel pruning уменьшает регулярные dimensions — полезно;
- unstructured weight pruning без sparse engine — только память;
- low-rank factorization dense layer выгодна, если
  \(r(F+H)<FH\) и дополнительная intermediate memory не доминирует;
- tensor decomposition имеет смысл для больших convolution kernels, которых в
  рекомендуемом baseline нет.

Вердикт: применять только после profiling конкретной residual network.

## 12. Quantization-aware training

Physical/surrogate DPD работы показывают рабочие 12–16 bit точки, но format
зависит от architecture и capture. Для проекта QAT не заменяет bit-accurate
verification:

- input/coeff/state formats;
- rounding;
- accumulator width;
- saturation;
- LUT interpolation;
- limit cycles;
- chunk equivalence;
- spectral degradation, а не только coefficient MSE.

Вердикт: прямо применимо после float topology freeze.

## 13. Hardware-aware NAS

NAS полезен только с target-derived latency/resource model. Parameter count
как objective недостаточен. Search space должен состоять из причинных
операторов: spline branch, short FIR, diagonal state, PWL residual neuron.
Candidate с look-ahead автоматически невалиден.

Growing complex CNN DPD
([IJCNN 2024](https://doi.org/10.1109/IJCNN60899.2024.10651335)) — релевантный
research reference, но наш первый search должен быть малым exhaustive/greedy
structured search, который легче воспроизвести.

Вердикт: применимо позже; сейчас cost-aware branch selection информативнее.

## 14. Адаптивная фильтрация и управление

### 14.1 LMS/NLMS

Плюсы: малая стоимость update, streaming, простота fixed-point. Минусы:
медленная сходимость при коллинеарном nonlinear dictionary, чувствительность
к step size и feedback noise. Normalization должна учитывать group energy.

Применение: редкий block update уже выбранных коэффициентов.

### 14.2 RLS с forgetting

Плюсы: быстрая адаптация к drift. Минусы:

- \(O(C^2)\) calibration arithmetic и \(O(C^2)\) covariance memory;
- численная устойчивость в fixed-point;
- forgetting может усиливать шум и сделать update неустойчивым.

Для deployed DPD стоимость RLS не учитывается, если update идёт на отдельном
контроллере; но acquisition/time-to-recover публикуются.

### 14.3 Kalman/EKF/UKF

Если coefficients — state с линейным observation equation, RLS/Kalman
эквивалентны в подходящей постановке. EKF/UKF оправданы только при явно
нелинейной state evolution и измеримом выигрыше. Sigma points UKF быстро
становятся дорогими.

Вердикт: linear coefficient Kalman/RLS — допустим; EKF/UKF не первый выбор.

### 14.4 Iterative learning control

ILC имеет сильное physical DPD evidence:

- [Chani-Cahuana et al.](https://doi.org/10.1109/TMTT.2016.2588483);
- [preinverse via ILC](https://arxiv.org/abs/1606.08663).

ILC может создавать high-quality teacher waveform, но требует повторяемого
сигнала и нескольких PA acquisitions. Это calibration path, не deployed
inference.

### 14.5 Adaptive observers

Observer полезен для slow thermal/bias state, если state наблюдаем из
baseband residual и датчиков. Нельзя объявлять estimated \(q_\beta\)
температурой без temperature measurement.

### 14.6 MPC

MPC с online optimization на каждый RF sample нереалистичен. Frame-rate
constraint management для coefficient update — применим, но тогда это
supervisory control, не sample-level MPC.

### 14.7 Extremum seeking и Bayesian optimization

Могут настраивать несколько slow hyperparameters по physical spectral metric,
но требуют безопасных PA probes. Bayesian optimization плохо масштабируется
на десятки spline/GMP coefficients; ridge/RLS использует структуру лучше.

### 14.8 Bandits

Bandit может выбирать один из уже безопасных coefficient banks при drift.
Exploration на live transmitter опасен; допустим только constrained/shadow
режим. Если режим наблюдаем датчиками, supervised router проще.

### 14.9 Reinforcement learning

RL оправдан для discrete PA configuration/self-healing, но DPD coefficients
имеют известную differentiable/linear structure. RL sample-inefficient,
трудно гарантирует spectrum/peak safety и не имеет преимущества над
LS/closed-loop decorrelation без специальной постановки.

Прямой prior существует: [Spano et al., 2025](https://doi.org/10.3390/s25196102)
проверили CRE-DDPG на стенде Huawei Milan. Контроллер ограничивает coefficient
action около ACLR threshold и выполняет recovery через обратную
последовательность сохранённых actions; improvement появлялся примерно после
850 updates. Это делает safe RL обязательным comparator, но также подчёркивает
ценность детерминированного advisor с 1–несколькими auditable shadow fits.

Вердикт: не рекомендуемый основной путь; high-risk safety comparator.

## 15. Итоговая transfer matrix

| Идея | Категория | Где выполняется | Causal/state | Fixed-point | Что проверить |
|---|---|---|---|---|---|
| Phase-equivariant scalar correction из rotation-invariant features | прямо | каждый sample | causal, delays | structurally favorable | spline/GMP vs PN-RNN |
| Tiny residual MLP | после ablation | каждый sample | causal, малый state | feasible with PWL/QAT; unverified | равная timing cost с branches |
| GRU/LSTM | teacher/reference | каждый sample | recurrent | сложно, но показано | worst-case latency/memory |
| Causal tiny TCN | после адаптации | каждый sample | buffer | structurally feasible; unverified | убрать look-ahead |
| Diagonal SSM | после адаптации | каждый sample | compact state | needs state-bound proof | vs one-pole/FIR |
| Full Mamba | не первый путь | каждый sample | selective state | тяжело | доказать cost advantage |
| Frame hard experts | прямо | router/frame; один expert/sample | bounded | fast path structurally favorable | vs coefficient LUT |
| Soft sample MoE | непрактично | несколько experts/sample | switching | дорого | обычно не запускать |
| Slow hypernetwork | после адаптации | frame/control | delayed update | fast path needs bit-true proof | held-out conditions |
| Low-rank coefficient subspace | после evidence | frame/control | bounded | fast path structurally favorable | SVD controlled captures |
| Distillation/ILC teacher | прямо как training | calibration | teacher arbitrary | student-dependent | direct-fit baseline |
| Continual learning bank | прямо как control | rare update | regime memory | model-specific | forgetting/rollback |
| Retrieval coefficient bank | прямо | frame lookup | no sample state | structurally favorable | OOD distance |
| Speculative sample inference | аналогия | — | incompatible | — | не рекомендуется |
| Shadow validation | прямо как safety | asynchronous | rollback state | не fast path | forced fault tests |
| Static structured sparsity | прямо | каждый sample | fixed | target-dependent | target kernel |
| Dynamic unstructured sparsity | после hardware proof | каждый sample | indices/cache | medium | worst-case active work |
| Neural ODE | непрактично | solver/sample | variable | плохо | не рекомендуется |
| RL coefficient control | высокий риск | calibration | policy state | medium | safe sample efficiency |

## 16. Рекомендуемый порядок

1. Использовать не «AI-модель», а AI-полезный механизм: residual observer,
   group selection и safe shadow update.
2. Проверить slow-state spline и frame-rate coefficient bank.
3. Использовать ILC/high-quality neural model как teacher только после
   physical feedback.
4. Затем сравнить tiny phase-equivariant residual MLP и diagonal SSM с
   structured branches при одинаковом измеренном timing.
5. Hypernetwork/meta-learning исследовать только после появления нескольких
   контролируемых operating points.

Главный вывод: наиболее ценные переносимые идеи — **разделение fast/slow path,
структурная разреженность, phase-equivariance, distillation и безопасные
низкоразмерные updates**. Transformer attention, full Mamba, speculative
decoding и in-context terminology не дают автоматического преимущества для
DPD.
