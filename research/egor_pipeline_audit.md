# Независимый аудит `DPD_for_PA` и `chaotic_library`

Дата аудита: 2026-07-28. Нумерация ячеек notebook ниже нулевая (`cell 0` — первая ячейка). Результаты экспериментов, которых нет в репозитории или которые не были запущены в ходе этого аудита, не приписываются авторам.

## 1. Зафиксированные версии и просмотренные артефакты

| Репозиторий | Commit | Состояние |
|---|---|---|
| `vendor/DPD_for_PA` | `8e8127cfbea4b2d67cc3d944514b4835e4c7e947` | `main`, clean |
| `vendor/chaotic_library` | `f4ebc3e7c302e83d2eb1c44244f5ecd6e2d884ce` | `main`, clean |
| `vendor/OpenDPD` — только для проверки происхождения данных | `7426bbf8a47624b59bd7f045a86641b403023f3c` | `main`, clean |

Основные просмотренные файлы:

- `vendor/DPD_for_PA/MY_PA_DPD.ipynb`, все 17 ячеек, включая сохранённые outputs;
- `vendor/DPD_for_PA/README.md`;
- `vendor/DPD_for_PA/DPD_3.pdf`, все 14 страниц;
- четыре CSV в `vendor/DPD_for_PA/data1/`;
- четыре PNG в `vendor/DPD_for_PA/imgs/`;
- `vendor/chaotic_library/src/chaotic_library/enhanced_esn_fan.py`, полностью;
- `vendor/chaotic_library/README.md`, `pyproject.toml`, `requirements.txt`, `tests/test_imports.py`;
- для идентификации dataset: `vendor/OpenDPD/datasets/DPA_200MHz/{spec.json,*.csv}` и `vendor/OpenDPD/datasets/README.md`.

SHA-256 ключевых файлов:

```text
d5cc89168cea930656b15062b9de650473a5dfd317696c80eb97a6e4fdbf2b68  MY_PA_DPD.ipynb
0282149263a28ba0fcbd3903b1dcc902b91c1e2e49e87fbfde4f1df062c1d79f  enhanced_esn_fan.py
05da7e48c14d1eadb003a43363d6b9f658a164b564b660f5edee21b9e9d4d0f3  train_input.csv
f3d9b3173e78a4c292028b304271ee16c52c2262dd122f593ca7b8b65097dff3  train_output.csv
262e1fe4365992735ef9d6f67c4ff557e8e4d8a4f4885b7bc5c7011d07f1b4c0  test_input.csv
57d13dc65abf2b8cf77a05d0f1cf90ca0b3520f2da6bb85eb7ba31048d46d9a8  test_output.csv
```

Все четыре CSV Егора побайтно совпадают с соответствующими файлами OpenDPD `DPA_200MHz`. Однако Egor-репозиторий не скопировал `val_input.csv`/`val_output.csv`.

Среда машины аудита:

```text
Linux 7.0.5-2-cachyos x86_64
Intel Core i5-12450H, 8 physical / 12 logical CPUs
15 GiB RAM
system Python 3.14.6
```

System Python не содержит `pandas`/`scikit-learn`, поэтому notebook целиком не переисполнялся. Это не мешает статически подтвердить направления mapping/state/runtime и выполнить NumPy-проверки данных и точный аналитический operation count. Сам notebook сообщает другую исходную среду: metadata — Python 3.13.7; cached output cell 0 — Windows venv, `opendpd==2.0.0`, `numpy==2.4.6`, `pandas==3.0.3`, `scipy==1.17.1`, `matplotlib==3.10.9`, `torch==2.12.0`. Версия `scikit-learn` и commit `chaotic_library` в notebook не зафиксированы.

Ключевые команды аудита:

```bash
git clone https://github.com/EgorMa1tsev/DPD_for_PA.git vendor/DPD_for_PA
git clone https://github.com/CapitalistGeorge/chaotic_library.git vendor/chaotic_library
git -C vendor/DPD_for_PA rev-parse HEAD
git -C vendor/chaotic_library rev-parse HEAD
jq -r '.cells | to_entries[] | ...' vendor/DPD_for_PA/MY_PA_DPD.ipynb
nl -ba vendor/chaotic_library/src/chaotic_library/enhanced_esn_fan.py
pdftotext -layout vendor/DPD_for_PA/DPD_3.pdf -
sha256sum vendor/DPD_for_PA/data1/*.csv vendor/OpenDPD/datasets/DPA_200MHz/*.csv
cmp vendor/DPD_for_PA/data1/train_input.csv vendor/OpenDPD/datasets/DPA_200MHz/train_input.csv
```

Upstream-код не изменялся.

## 2. Краткий итог

1. **ILA mapping выбран в принципе допустимо:** postdistorter обучается как `y/g -> x`. Но один из двух test paths действительно круговой.
2. **Cell 10 — круговая reconstruction-проверка:** `y_test/g -> DPD -> PA surrogate -> y_test`. Она не доказывает predistortion нового desired input.
3. **Cell 11 исправляет направление:** `x_test -> DPD -> PA surrogate -> g*x_test`, но даёт только AM/AM, AM/PM и PSD plots — без NMSE/EVM/ACLR и без real-PA validation.
4. **Cell 14 и отслеживаемый `imgs/psd2.png` снова круговые.** README показывает правильный и круговой PSD рядом, не отмечая принципиальную разницу.
5. **Validation split OpenDPD полностью выброшен.** Есть только train/test; значит нет честного места для выбора hyperparameters/checkpoint.
6. **PSD ось и framing неверны для этого dataset:** notebook задаёт `fs=200`, `nperseg=256`; dataset требует `fs=800 MHz`, `nperseg=2560`.
7. **`sparsity=0.1` не даёт sparse runtime.** `W` — dense `numpy.ndarray`; вызывается dense `W @ state`. Развёрнутый DPD стоит примерно **728,622 real multiplications/sample**, то есть примерно в 729 раз выше ограничения 1000, ещё до учёта стоимости `tanh`, `sin`, `cos`, memory traffic и Python.
8. **Каждый `predict()` сбрасывает reservoir в ноль.** `last_state_` записывается после `fit`, но обычным prediction path не используется. Chunked/one-sample streaming неэквивалентен full-array prediction.
9. **Две независимые I/Q модели не обеспечивают phase equivariance.** Они видят обе координаты, но используют разные random reservoirs и ничем не связанные readouts.
10. **Fourier block не является спектральным преобразованием сигнала.** Это pointwise `sin/cos` от стандартизованных I и Q; он не вычисляет FFT и не кодирует временную частоту напрямую.
11. Сохранённые результаты соответствуют приблизительно **-31.67 dB PA NMSE** и **-32.09 dB circular-cascade NMSE**, а не -50 dB. Для честного correct-direction cascade scalar metric в репозитории отсутствует.
12. Заявления о 10x/100x training speed и меньшей памяти не имеют воспроизводимых timing/memory logs, baseline config или checkpoints. В inference memory доминируют фиксированные dense reservoirs, а не маленький ridge readout.

## 3. Dataset, splits и preprocessing

### 3.1 Что это за данные

По SHA-256 это точная копия OpenDPD `DPA_200MHz`:

- 10 × 20 MHz LTE carriers;
- 200 MHz occupied bandwidth;
- sampling rate 800 MHz;
- 64-QAM;
- IFFT frame size 2560;
- split `train/val/test = 0.6/0.2/0.2`.

Источники: `vendor/OpenDPD/datasets/DPA_200MHz/spec.json:2-18`, `vendor/OpenDPD/datasets/README.md:145-157`. OpenDPD описывает CSV как time-aligned PA input/output (`datasets/README.md:3`).

Число samples:

| Split | Samples | IFFT frames по 2560 |
|---|---:|---:|
| train | 23,040 | 9 |
| validation | 7,680 | 3 — отсутствует у Егора |
| test | 7,680 | 3 |

Точные train/test I/Q rows не пересекаются. Это не обнаруживает утечку через происхождение waveform, но исключает простое дублирование строк.

### 3.2 Загрузка и воспроизводимость

- Notebook cell 3 читает `train_input.csv`, `train_output.csv`, ... из текущей директории (`MY_PA_DPD.ipynb:108`), но tracked files находятся в `data1/`. Чистый checkout notebook не запускает без смены path/copy/symlink.
- Нет `requirements.txt`, lockfile или install-команды для `chaotic_library`. Cell 0 ставит unpinned latest `opendpd matplotlib scipy`, хотя `opendpd` затем не используется.
- Все `execution_count` равны `null`, хотя outputs сохранены. Порядок исполнения по metadata доказать нельзя.
- Нет сохранённых PA/DPD checkpoints, `predictions.csv`, `cascade_predictions.csv`, timing logs или configs.
- Seeds reservoirs заданы, но версия библиотеки не pin'нута.

### 3.3 Отсутствующий validation split

Notebook использует train и test напрямую (cell 3) и строит все итоговые plots на test (cells 11, 14). Validation CSV из исходного dataset не включены. Даже если показан только один набор hyperparameters, невозможно проверить, не выбирались ли `R`, `ridge_alpha`, `fan_terms`, `sparsity` и выводы по test. Для apples-to-apples воспроизведения надо вернуть оригинальный val split и заморозить test до финального запуска.

### 3.4 Alignment и gain

Ни integer/fractional delay search, ни DC/IQ/frequency-response correction в notebook нет. Простая диагностическая complex cross-correlation в диапазоне ±100 samples имеет максимум на lag 0 (`0.99488` train, `0.99476` test), что согласуется с заявлением OpenDPD о предварительном time alignment, но не проверяет fractional delay или feedback-path response.

Gain в cell 6 (`MY_PA_DPD.ipynb:220`) вычисляется как

```text
(mean(|y_I|) + mean(|y_Q|)) /
(mean(|x_I|) + mean(|x_Q|)) = 3.210296519.
```

Это не least-squares complex gain. Корректный минимум для `||y-gx||²`:

```math
g_\mathrm{LS} = \frac{\sum_n x^*[n] y[n]}{\sum_n |x[n]|^2}.
```

На этих CSV:

| Split | `g_LS` | phase | notebook-style gain |
|---|---:|---:|---:|
| train | `3.165638314 - j3.52e-11` | `-6.4e-10°` | `3.210296519` |
| test | `3.159012167 - j9.44e-11` | `-1.7e-9°` | `3.214079956` |

Здесь комплексная фаза практически нулевая, но notebook gain завышен примерно на 1.4% относительно train LS gain. Это меняет desired output power, DPD drive и PAPR. Выбор gain должен быть частью фиксированного protocol, а не marginal-amplitude heuristic.

### 3.5 Distribution shift, скрытый ILA

Для DPD train input используется `u_train=y_train/3.2103`. Его максимальная amplitude `0.7852`, тогда как desired `x_test` достигает `1.0`; 79 из 7680 test samples (1.03%) превышают максимум DPD-train amplitude. 49 samples выходят хотя бы по одной Cartesian координате за min/max DPD train input; примерно 1.00% samples имеют хотя бы одну координату дальше `3σ` и подвергаются clipping в deterministic-feature path.

Это особенно важно, потому что эти редкие большие amplitudes находятся в compression region. Circular test `y_test/g` остаётся на postdistorter training manifold и не испытывает такого extrapolation.

## 4. Фактический PA/DPD pipeline

### 4.1 PA

Cell 8:

```text
[I_in, Q_in] -> esn_I(R=800, seed=42) -> I_out
[I_in, Q_in] -> esn_Q(R=800, seed=43) -> Q_out
```

То есть каждая scalar-output модель видит обе I/Q coordinates; это не две строго одномерные модели. Но reservoirs, states, scalers и ridge readouts независимы. Hyperparameters находятся в `MY_PA_DPD.ipynb:281-298`, size — строка 287. Mapping input→measured output корректен.

Сохранённый output cell 8:

```text
I: MSE=0.000396, R²=0.999431
Q: MSE=0.000529, R²=0.999202
```

PA test prediction начинается с zero reservoir state, warm-up не отбрасывается.

### 4.2 DPD training

Cell 10, `MY_PA_DPD.ipynb:370-399`:

```text
u_train = y_train / coeff
target  = x_train

[Re(u), Im(u)] -> dpd_I(R=600, seed=100) -> I_PA_drive
[Re(u), Im(u)] -> dpd_Q(R=600, seed=101) -> Q_PA_drive
```

Это стандартная форма indirect learning/postdistorter fit. Само направление обучения не является ошибкой при условии, что inverse однозначен, postdistorter переносится в predistorter position, а train domain покрывает desired-input domain.

### 4.3 Три разных evaluation path

| Ячейка | Реальный вход DPD | Target | Verdict |
|---|---|---|---|
| cell 10, `:370-418` | `y_test/coeff` | measured `y_test` | круговая inverse/forward reconstruction |
| cell 11, `:493-502` | desired `x_test` | `coeff*x_test` | направление эксплуатации корректно, но только surrogate plots |
| cell 14, `:643-658` | `y_test/coeff` (через ранее вычисленный `pa_out_pred`) | `y_test` | снова круговая |

Cell 10 cached output:

```text
I: MSE=0.000246, R²=0.999646
Q: MSE=0.000594, R²=0.999104
```

Cell 11 не вычисляет ни MSE/NMSE, ни EVM, ни ACLR. Следовательно, scalar claim для честного `x_test -> DPD -> PA` в repository отсутствует.

## 5. Шесть заданных гипотез

### H1. Test может быть круговым

**Verdict: частично подтверждена.** Круговая проверка точно есть в cell 10 и cell 14, но утверждение «весь notebook тестирует только reconstruction» было бы неверным: cell 11 подаёт на DPD именно `X_test`.

Код:

- `X_dpd_test=[y_test_I/coeff,y_test_Q/coeff]`: `MY_PA_DPD.ipynb:371`;
- inverse prediction: `:403-404`;
- PA surrogate: `:407-409`;
- target снова `y_test`: `:412-418`;
- правильный вход `dpd_I.predict(X_test)`: `:493`;
- правильный ideal `coeff*x_complex`: `:502`;
- cell 14 возвращается к `u=y_test/coeff`: `:643-644`.

Математически, пусть `P` — physical PA, `P_hat` — surrogate, `y_i=P(x_i)`, `u_i=y_i/g`. ILA обучает

```math
D = \arg\min_D \sum_i |D(u_i)-x_i|^2.
```

Circular score:

```math
\hat P(D(P(x_i)/g)) \approx P(x_i)
```

мал по построению, если `D(u_i)≈x_i` и `P_hat(x_i)≈P(x_i)`. В первом порядке:

```math
e_\mathrm{circ}\approx J_P(x_i)e_D(u_i)+e_{\hat P}(x_i).
```

Он проверяет согласованность inverse и forward models на известных `u_i`; из него не следует

```math
P(D(s))\approx gs
```

для нового desired signal `s`. Для этого нужны domain coverage, устойчивый inverse и independent/physical PA evaluation.

`imgs/psd1.png` по title соответствует correct-direction cell 11 (`"PSD comparison"`). `imgs/psd2.png` соответствует circular cell 14 (`"Power Spectral Density comparison"`). README `:28-30` показывает их как два обычных DPD результата, не помечая, что второй использует measured PA output для создания DPD input.

Минимальный эксперимент:

1. Fit только на train; выбирать параметры только по val.
2. На val/test задать `s=x_val/x_test`, вычислить `z=D(s)`, затем `P_real(z)` или хотя бы независимый `P_hat2(z)`.
3. Сравнить строго с `g*s`; `y_val/y_test` не использовать как DPD input.
4. Отдельно посчитать circular score и показать разницу.

### H2. Высокий R² не гарантирует хороший NMSE/ACLR

**Verdict: подтверждена.**

Код использует только component MSE и R² (`MY_PA_DPD.ipynb:311-314`, `:415-418`). ACLR/EVM/NMSE не вычисляются.

На test output power `mean(|y|²)=1.359229996`. Из округлённых cached MSE:

| Path | Average real-component MSE | Complex MSE | Approx. complex NMSE |
|---|---:|---:|---:|
| PA surrogate | `4.625e-4` | `9.25e-4` | `-31.67 dB` |
| circular DPD→PA surrogate | `4.20e-4` | `8.40e-4` | `-32.09 dB` |

Даже circular результат имеет normalized error power около `6.18e-4`, далеко от `1e-5`/`-50 dB`. Correct-direction NMSE неизвестен.

R² нормирует SSE на centered component variance; он не контролирует spectral location ошибки. Маленькая широкополосная или adjacent-band ошибка может почти не менять R², но сильно ухудшить ACLR. Separate I/Q R² также не заменяет complex gain/phase-aligned NMSE.

Минимальный эксперимент: тем же evaluator вычислить complex NMSE, EVM, left/right/average ACPR по `spec.json`, PAPR и peak DPD amplitude для PA-only, circular и correct-direction outputs. Не выбирать checkpoint по test.

### H3. Reservoir 600/800 значительно дороже лимита

**Verdict: подтверждена.**

`W` создаётся как dense NumPy array (`enhanced_esn_fan.py:164`), маскируется нулями (`:166-168`), сохраняется без sparse conversion (`:184`) и вызывается как `self.W @ state` (`:196`). Dense BLAS не пропускает нулевые coefficients. README библиотеки одновременно:

- неверно называет `sparsity` долей zeroed connections (`README.md:302`), хотя code сохраняет примерно долю `sparsity` nonzero;
- пишет `O(T*N_r*s)` (`README.md:314`), тогда как sparse recurrence должна быть `O(T*s*N_r²)`, а текущая dense — `O(T*N_r²)`.

Полный operation count приведён в разделе 7. Deployed DPD I+Q: около **728,622 normalized real multiplications/sample**, **726,152 additions/sample**, 1,200 `tanh` и 64 trig calls. Даже идеальная CSR-реализация тех же четырёх seed matrices оставляет DPD около **80,410 multiplications/sample**, то есть >80× целевого лимита.

Минимальный эксперимент: после `fit` вывести `type(W)`, `W.dtype`, `W.nbytes`, `count_nonzero(W)`; benchmark dense `W@state` и CSR на batch/chunk size 1 с одним thread, затем сравнить outputs bitwise/tolerance. Отдельно benchmark full `predict`, включая trig/scalers/readout.

### H4. Две I/Q модели могут нарушать phase equivariance

**Verdict: структурно подтверждена; фактическая величина нарушения не измерена.**

Models имеют разные seeds (`MY_PA_DPD.ipynb:297-298`, `:392-393`), разные reservoirs/states/scalers/readouts и не имеют weight tying. Cartesian polynomial/Fourier features также не rotation-equivariant.

Для `F(I,Q)=(F_I,F_Q)` equivariance требует

```math
F(R_\phi v)=R_\phi F(v)
```

для всех `v,φ`. Уже при `φ=π/2` необходимо

```math
F_I(-Q,I)=-F_Q(I,Q),\qquad F_Q(-Q,I)=F_I(I,Q),
```

чего независимые random models не обеспечивают. Это не означает, что две-output Cartesian model всегда запрещена: если measurement path имеет IQ imbalance, строгая equivariance может быть слишком сильной. Но здесь нарушение не является осознанно измеренным trade-off.

Минимальный эксперимент: для нескольких `φ` вращать целую complex sequence, сохранять одинаковые state/reset rules и считать

```math
\epsilon_\mathrm{eq}(\phi)=
\frac{\sum_n|D(e^{j\phi}x)[n]-e^{j\phi}D(x)[n]|^2}
{\sum_n|D(x)[n]|^2}.
```

Сравнить с phase-equivariant spline/MP baseline и проверить наличие IQ imbalance в measurements.

### H5. Hidden state сбрасывается при каждом predict/frame

**Verdict: подтверждена.**

- `_compute_states(..., initial_state=None)` объявлен в `enhanced_esn_fan.py:201-205`;
- при `None` создаётся zero state (`:219-220`);
- `fit` сохраняет `last_state_` (`:319-330`);
- open-loop `predict` вызывает `_compute_states(X)` без initial state (`:365-373`);
- `last_state_` в prediction path не читается.

Поэтому:

```text
predict(X)
!= concatenate(predict(X[:k]), predict(X[k:]))
!= concatenate(predict(X[n:n+1]) for n)
```

в общем случае. Notebook подаёт весь test array одним вызовом, поэтому state сохраняется внутри этих 7680 samples, но сбрасывается между каждым вызовом/cascade stage. Публичного streaming state API нет; one-sample эксплуатация фактически уничтожит заявленную memory dynamics.

Минимальный эксперимент: сравнить full-array output с 2 chunks, frames по 2560 и 7680 отдельных one-sample calls. Затем добавить явно передаваемый `(state_in,state_out)`, warm-up/context и unit test chunk equivalence.

### H6. Cartesian Fourier features — слабый inductive bias для PA/DPD

**Verdict: точная форма подтверждена; утверждение о худшем качестве требует ablation.**

Source строит для каждого стандартизованного I и Q:

```math
\sin(2\pi k I_s),\cos(2\pi k I_s),
\sin(2\pi k Q_s),\cos(2\pi k Q_s),\quad k=1,\dots,8,
```

`enhanced_esn_fan.py:244-262`, после input scaling/clipping `:235-242`, `:282-290`. Это 32 pointwise nonlinear features. Это **не FFT, не PSD и не явный учёт temporal spectrum**, вопреки формулировкам `DPD_for_PA/README.md:7`, `:30`, `DPD_3.pdf` pages 4/6.

`PolynomialFeatures(degree=2)` (`enhanced_esn_fan.py:74`, `:284-287`) даёт 5 features: `I,Q,I²,IQ,Q²`. Reservoir получает raw, не scaled/clipped input (`:319` → `_compute_states`; scaling выполняется только позже в `_build_feature_matrix`).

Для обычного complex PA более естественны phase-equivariant radial/memory terms `x[n-m]|x[n-d]|^{p-1}`. Cartesian periodicity по I/Q:

- зависит от выбранной фазы координат;
- вводит осциллирующую экстраполяцию;
- не соответствует монотонной compression области;
- требует дорогих `sin/cos`;
- усложняет fixed-point.

Минимальный эксперимент: при одинаковом split, state size/seed и ridge protocol сравнить `reservoir-only`, `+poly`, `+Cartesian FAN`, radial spline/MP features и phase-normalized features. Выбирать по val NMSE+ACLR, минимум 3 seeds. Текущая validation запрещает `fan_terms=0`, поэтому poly-only ablation надо реализовать явно, а не подменять `fan_terms=1`.

## 6. `EnhancedESN_FAN`: scaling, ridge, state и conditioning

Фактический feature path:

1. Reservoir state от raw `[I,Q]`;
2. отдельно `StandardScaler` + clip ±3 только для polynomial/FAN;
3. concatenate `[state, polynomial, Fourier]`;
4. второй `StandardScaler` всех `D` features;
5. scalar `Ridge(alpha=1e-2, fit_intercept=True)`.

Источники: `enhanced_esn_fan.py:74-77`, `:195-199`, `:235-242`, `:264-296`, `:317-330`.

Положительное: readout fit convex и deterministic при фиксированном reservoir. Ограничения:

- washout samples не удаляются;
- ridge alpha не выбирается по validation;
- I/Q fits дважды строят почти одинаковые expensive matrices;
- target scaling отсутствует;
- full feature matrix и all states хранятся в RAM;
- random dense eigendecomposition нужна для spectral-radius normalization (`:170-175`);
- `StandardScaler` combined path можно algebraically fold в readout для inference, но текущий source этого не делает;
- default dtype — `float64` (`:109`, `:126`, `:217`, `:220`, `:222`); quantization/fixed-point/export отсутствуют;
- `tanh`, 32 `sin/cos` и scikit-learn pipeline не являются FPGA-ready.

Unit tests библиотеки проверяют только import (`tests/test_imports.py:1-7`); state continuity, sparsity semantics, numerical regression и serialization не тестируются.

## 7. Пошаговый operation count

### 7.1 Конвенция

Один PA/DPD scalar-output model имеет:

```text
d = 2 inputs
R = reservoir size
P = C(d+2,2)-1 = 5 polynomial features
F = 2*fan_terms*d = 32 Fourier features
D = R+P+F = R+37 readout features
```

Считаются scalar real operations одного sample. Dense dot считает zero coefficients. FMA разложен как 1 multiplication + 1 addition. `tanh`, `sin`, `cos`, division, comparison и memory access выделены отдельно.

`StandardScaler` в source делает divisions. Для hardware-normalized столбца предполагается заранее сохранённый reciprocal, поэтому каждая division заменена одной multiplication. Это благоприятно для implementation, но требует coefficients reciprocals.

Fourier source дважды формирует angle для sin/cos: 32 vector-scalar multiplications/sample в batch. При строго one-sample Python call добавляются ещё 32 scalar constant multiplications; они амортизируются в текущем whole-array call. Оптимизированный hardware мог бы precompute `2πk` и разделить angle между sin/cos, сократив этот пункт с 32 до 16.

### 7.2 Один scalar-output model

| Операция | Multiplications | Add/sub | Other |
|---|---:|---:|---:|
| `Win@[1,I,Q]` | `3R` | `2R` | — |
| dense `W@state` | `R²` | `R(R-1)` | `R²+R` coefficient/state reads минимум |
| sum двух matvec | — | `R` | — |
| `tanh` | — | — | `R tanh` |
| leak `(1-a)s+a*s_new` | `2R` | `R` (+1 scalar subtraction) | — |
| input standardization | — | `2` | `2 divisions`, 4 clip comparisons |
| degree-2 polynomial | `3` | — | — |
| FAN arguments | `32` | — | `32 trig` |
| combined standardization | — | `D` | `D divisions` |
| ridge dot + intercept | `D` | `D` | — |

Итого:

```text
source real multiplications = R² + 6R + 72
real divisions              = R + 39
hardware-normalized mult    = R² + 7R + 111
real additions/subtractions = R² + 5R + 76  (+1 scalar)
nonlinear calls             = R tanh + 32 trig
comparisons                 = 4
```

### 7.3 Четыре фактические модели

| Model/path | R | Hardware-normalized mult/sample | Add/sample | Nonlinear/sample |
|---|---:|---:|---:|---:|
| PA I | 800 | 645,711 | 644,076 | 800 tanh + 32 trig |
| PA Q | 800 | 645,711 | 644,076 | 800 tanh + 32 trig |
| **PA pair** | — | **1,291,422** | **1,288,152** | 1,600 tanh + 64 trig |
| DPD I | 600 | 364,311 | 363,076 | 600 tanh + 32 trig |
| DPD Q | 600 | 364,311 | 363,076 | 600 tanh + 32 trig |
| **deployed complex DPD** | — | **728,622** | **726,152** | 1,200 tanh + 64 trig |
| **software DPD→PA surrogate cascade** | — | **2,020,044** | **2,014,304** | 2,800 tanh + 128 trig |

Physical deployment не исполняет PA surrogate, поэтому основной gate надо применять к строке deployed complex DPD: 728,622, а не к full software cascade. Она превышает `<1000` примерно в 729 раз.

При 800 MS/s, указанной в dataset spec, это формально `5.83e14` real multiplications/s только для DPD; даже notebook-ошибка 200 MS/s дала бы `1.46e14`. Это не practical real-time architecture.

### 7.4 Фактические nonzeros и идеальный sparse lower bound

Exact masks воспроизведены из `default_rng(seed)` с тем же числом draw для `Win`:

| Model | R/seed | `nnz(W)` | Density | Ideal CSR normalized mult/sample |
|---|---:|---:|---:|
| PA I | 800/42 | 63,620 | 9.9406% | 69,331 |
| PA Q | 800/43 | 64,114 | 10.0178% | 69,825 |
| DPD I | 600/100 | 35,903 | 9.9731% | 40,214 |
| DPD Q | 600/101 | 35,885 | 9.9681% | 40,196 |

Идеальный CSR DPD pair всё равно требует примерно **80,410 mult/sample**, не считая sparse indices/memory irregularity и nonlinear cost. Текущий code не достигает этого lower bound.

### 7.5 Stored coefficients и RAM

Trainable readout parameters:

| Pair | Trainable ridge weights + intercepts |
|---|---:|
| PA I/Q | `2*(837+1)=1,676` |
| DPD I/Q | `2*(637+1)=1,276` |
| Total | `2,952` |

Но inference должен хранить fixed random `W`, `Win`, readout и scaling coefficients. Нижняя оценка essential dense scalars/model:

```text
R² (W) + 3R (Win) + (D+1) (ridge) + 2D+4 (two scaler mean/scale)
= R² + 6R + 116.
```

| Pair | Essential dense scalars | float64 bytes | MiB |
|---|---:|---:|---:|
| PA I/Q | 1,289,832 | 10,318,656 | 9.84 |
| deployed DPD I/Q | 727,432 | 5,819,456 | 5.55 |
| full software cascade | 2,017,264 | 16,138,112 | 15.39 |

Это optimistic lower bound: actual sklearn objects дополнительно хранят scaler variances, `last_state_`, counts и Python metadata. Parameter count только trainable ridge weights скрывает основную storage/runtime стоимость.

### 7.6 Training complexity

Для каждого fit source:

- вычисляет dense eigenvalues `O(R³)`;
- сохраняет `T×R` states;
- выполняет `T` dense `R×R` matvec;
- строит dense `T×D` matrix;
- решает dense ridge.

При `T=23,040`:

| Pair | Только reservoir `W@state` multiplications | `T*D²` ridge-crossproduct order |
|---|---:|---:|
| PA I/Q | 29.49 billion | 32.28 billion |
| DPD I/Q | 16.59 billion | 18.70 billion |
| Total | 46.08 billion | 50.98 billion |

Это order estimate, не измеренный runtime; конкретный solver/version может изменить алгоритм. Per-model feature matrices занимают около 154.3 MB (PA) или 117.4 MB (DPD) float64, states — 147.5/110.6 MB, до временных copies и ridge workspace.

## 8. AM/AM, AM/PM, PSD и RF metrics

### AM/AM

Cell 11 использует desired `x_test` на горизонтальной оси и surrogate predictions на вертикальной. Measured `y_test` не показывается. Scatter без amplitude binning/median/confidence intervals смешивает memory trajectories.

### AM/PM

Phase error вычисляется относительно input phase и wrap'ится корректно, но:

- нет amplitude threshold, поэтому near-zero samples дают бессмысленные ±180° outliers;
- не вычитается выбранный complex linear gain/phase;
- нет binning по amplitude;
- показывается surrogate, не real PA.

### PSD

Cell 11 `MY_PA_DPD.ipynb:542-547` и cell 14 `:703-707` используют:

```text
fs=200
nperseg=256
Welch defaults otherwise
```

Для DPA_200MHz metadata требует `fs=800e6`, `nperseg=2560`. Поэтому frequency axis сжата в 4 раза, а main 200 MHz channel визуально выглядит как примерно ±25 вместо ±100 MHz. `nperseg` в 10 раз меньше IFFT frame. Нет явных window/overlap/nfft/detrend/scaling settings, channel masks или left/right integration.

Не вычислены:

- ACLR/ACPR left/right/average;
- EVM;
- complex NMSE;
- PAPR;
- peak/maximum predistorted amplitude;
- stability/clipping;
- confidence intervals.

По PSD plot нельзя заявлять apples-to-apples преимущество. Особенно нельзя использовать near-ideal `psd2` как evidence predistortion: это circular cell 14.

## 9. Проверка заявлений о скорости и памяти

`DPD_for_PA/README.md:36` заявляет примерно 100× faster training и меньшую память, тогда как `DPD_3.pdf`, page 12 заявляет примерно 10× меньше training time/compute. В repository нет:

- `time.perf_counter`/profiler;
- hardware/thread/BLAS configuration;
- OpenDPD command/config/checkpoint;
- training epochs/early stopping для сравнения;
- peak RSS/serialized-size measurements;
- inference latency/throughput;
- saved models.

Поэтому обе цифры **неподтверждены и взаимно несогласованы**. Convex ridge действительно может обучаться за один solve вместо многих gradient epochs, но текущий pipeline повторяет dense state rollout/eigendecomposition/ridge четыре раза. Deployed DPD хранит минимум 727k dense float64 scalars и исполняет ~729k multiplications/sample. Это не поддерживает заявление о компактном real-time inference.

Сравнивать надо отдельно:

1. calibration wall-clock и energy;
2. trainable coefficient count;
3. all stored coefficient count;
4. serialized bytes;
5. inference operations/sample;
6. measured latency/throughput при batch=1;
7. fixed-point resource/timing.

## 10. Что подтверждено и что нет

Подтверждено кодом/данными:

- dataset — exact OpenDPD DPA_200MHz train/test;
- validation split пропущен;
- PA mapping правильный;
- ILA postdistorter mapping `y/g -> x`;
- cell 10/14 circular, cell 11 correct-direction;
- two scalar I/Q models with both I/Q inputs;
- R=800 PA, R=600 DPD;
- dense `W@state`, около 10% numerical nonzeros;
- state reset при каждом `predict`;
- Cartesian Fourier/poly features и два уровня scaling;
- ridge alpha `1e-2`;
- wrong PSD `fs/nperseg`;
- cached R²/MSE недостаточны и не достигают -50 dB normalized error;
- нет checkpoints/timing/fixed-point artifacts.

Не подтверждено:

- correct-direction complex NMSE/EVM/ACLR;
- работа на physical PA после DPD;
- преимущество над любой OpenDPD architecture;
- 10× или 100× calibration speed;
- меньшая deployment memory;
- real-time throughput;
- устойчивость между seeds/power/waveforms;
- numerical phase-equivariance error;
- польза FAN над polynomial/radial baseline;
- fixed-point degradation.

## 11. Минимальный честный план исправления evaluator

До изменения модели:

1. Вернуть оригинальный `val` split и metadata OpenDPD.
2. Исправить file paths; pin Python/package/commit versions.
3. Зафиксировать LS complex gain, integer/fractional alignment и warm-up policy.
4. Разделить три именованных path:
   - `PA_ONLY: x -> P`;
   - `CIRCULAR_DIAGNOSTIC: y/g -> D -> P`, только diagnostic;
   - `PREDISTORTION: x -> D -> P`, основной score.
5. Никогда не формировать DPD test input из `y_test`.
6. Использовать `fs=800e6`, `nperseg=2560`, фиксированные channel masks и единый RF evaluator OpenDPD.
7. Добавить complex NMSE, EVM, ACPR L/R/avg, PAPR, peak drive и saturation.
8. Добавить stateful streaming API и тест full-vs-chunk equivalence; определить reset только на реальных sequence boundaries.
9. Добавить operation/storage counter, который считает dense zeros и nonlinear ops.
10. Сначала воспроизвести current model с 3 seeds, затем ablate reservoir/FAN/poly; test открыть один раз.

Acceptance gate для текущего reservoir DPD уже нарушен по аналитическому operation count. Поэтому дальнейшая оптимизация его quality без архитектурного сокращения/замены не отвечает требованию `<1000 real multiplications/sample`.
