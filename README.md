# Low-complexity Digital Predistortion

Репозиторий содержит аудит Digital Predistortion, воспроизводимую
экспериментальную среду и дешёвый фазово-эквивариантный
spline-memory DPD. Ближайшая воспроизводимая поставка — это
**validation-only surrogate demo**, а не результат на физическом PA.

Проверяемый тракт имеет правильное направление:

```text
desired validation x -> frozen DPD -> frozen PA surrogate -> output
```

Измеренный выход PA не подаётся на вход DPD. В demo нет fitting,
model selection или доступа к test split.

## Метод

Быстрый тракт — причинная трёхветвевая комплексная spline-memory
модель:

\[
z[n]=\sum_{m\in\{0,1,2\}}x[n-m]C_m(|x[n]|),
\]

где \(C_m\) — complex local-linear splines с quantile knots. Локальная
поддержка basis даёт регулярный LUT-friendly datapath, а общая
комплексная коррекция сохраняет фазовую эквивариантность:

\[
D(xe^{j\phi})=D(x)e^{j\phi}.
\]

Медленный `observer/advisor`, который по residual реального PA может
предлагать одну дополнительную ветвь и проверять её в shadow,
пока является **research proposal**. Он не входит в demo и не считается
реализованным результатом.

## Быстрый запуск

```bash
git clone --recurse-submodules git@github.com:theJorDea/DPD.git
cd DPD
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-baseline.txt
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.run_surrogate_demo \
  --output-root experiments/results/surrogate_demo_local_01
```

`--output-root` должен указывать на новый, ещё не существующий каталог.
Это защищает прежние evidence bundles от незаметной перезаписи. Запуск
проверяет pinned `numpy==2.5.1`, hashes конфигов, кода и входов,
а также замороженные reference metrics.

Ожидаемый консольный финал:

```text
DPA_200MHz: NMSE -20.338 -> -30.532 dB; adjacent relative L/R +4.749/+7.737 dB
APA_200MHz: NMSE -19.969 -> -32.380 dB; adjacent relative L/R +16.480/+13.864 dB
PASS: validation-only surrogate demo; no physical-PA claim
```

## Воспроизводимые surrogate-результаты

Все числа ниже относятся к ранее использованному validation split и frozen legacy
PA surrogate. DPA и APA — разные PA/captures и не объединяются.

| Dataset | No DPD NMSE | Float DPD NMSE | Левая/правая configured adjacent-region improvement |
|---|---:|---:|---:|
| DPA_200MHz | -20.338 dB | -30.532 dB | +4.749 / +7.737 dB |
| APA_200MHz | -19.969 dB | -32.380 dB | +16.480 / +13.864 dB |

Bit-accurate integer reference воспроизводит три заранее замороженных формата;
precision по этим результатам не выбирается.

| Format | DPA cascade NMSE | DPA adjacent suppression L/R | APA cascade NMSE | APA adjacent suppression L/R |
|---|---:|---:|---:|---:|
| signed 16-bit | -30.532 dB | 4.798 / 7.788 dB | -32.385 dB | 16.511 / 13.906 dB |
| signed 14-bit | -30.534 dB | 4.793 / 7.776 dB | -32.370 dB | 16.535 / 13.871 dB |
| signed 12-bit | -30.515 dB | 4.782 / 7.692 dB | -32.379 dB | 16.369 / 13.982 dB |

Для всех шести dataset/format combinations пройдены zero
saturation/knot-collision, exact configured chunked-streaming equivalence и
bit-exact 90-degree phase-rotation checks. Это software arithmetic evidence,
а не synthesized RTL или выбор лучшей разрядности для target.

## Стоимость DPD

Это аналитические operation schedules **на один complex sample**. Они не
заменяют измерение latency и throughput на целевом FPGA/DSP/ASIC.

| Schedule | MUL | ADD | DIV | Nonlinear | LUT | Compare DPA/APA | Memory read/write | State reals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Float reference | 21 | 24 | 0 | 1 amplitude | 6 | 5 / 3 | 18 / 2 | 4 |
| Fixed integer | 20 | 25 | 1 | 1 integer sqrt | 8 | 5 / 3 | 28 / 2 | 4 |

Хранится 144 real coefficients для DPA и 48 для APA. Само число
`MUL/sample` не является доказательством выполнения customer timing gate:
в него также входят memory traffic, LUT, division и nonlinear operations.

## Артефакты запуска

В новом output root публикуются:

- `summary.json` — metrics, operation vectors, environment, method и provenance;
- `completion_manifest.json` — финальный sealed manifest с hashes;
- 12 child `completion_manifest.json` — float/fixed replay и spectral stages
  для двух datasets;
- дочерние reports, replay waveforms и PSD arrays в `datasets/`, а также
  sealed config copies в `sealed_inputs/`.

Корневой `completion_manifest.json` публикуется последним. Его отсутствие
означает, что run незавершён и не должен цитироваться как evidence.

## Границы доказательств

Текущее demo подтверждает только воспроизводимую численную работу
frozen spline-memory DPD через legacy PA surrogate на validation. Оно
**не доказывает**:

- линеаризацию физического PA или Huawei base station;
- customer-defined harmonic/spur attenuation;
- RF harmonic suppression около \(2f_c\), \(3f_c\) и выше;
- apples-to-apples превосходство над OpenDPD;
- соответствие target timing budget, RTL resources или power;
- generalization между PA, power levels, waveforms и temperature.

Пока не определены customer integration bands/reference/threshold,
приведённые left/right числа называются только **configured
complex-baseband adjacent-region diagnostics**, а не ACLR/RF-harmonic
certification.

## Карта документов

- [`REQUIREMENTS.md`](REQUIREMENTS.md) — известные и неизвестные требования;
- [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md) — общий исследовательский аудит;
- [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) — только фактически выполненные benchmark runs;
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) и [`ROADMAP.md`](ROADMAP.md) — протокол,
  gates и порядок работ;
- [`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md) — API, изменения и ограничения
  реализации;
- [`FINAL_GAP_ANALYSIS.md`](FINAL_GAP_ANALYSIS.md) — что доказано и чего не хватает;
- [`research/FINAL_MODERN_DPD_RESEARCH_CONCLUSION.md`](research/FINAL_MODERN_DPD_RESEARCH_CONCLUSION.md)
  — вывод из 89 первичных источников;
- [`research/opendpd_audit.md`](research/opendpd_audit.md) и
  [`research/egor_pipeline_audit.md`](research/egor_pipeline_audit.md) — аудиты baseline;
- [`research/residual_observer_and_controller.md`](research/residual_observer_and_controller.md)
  — proposed slow observer/advisor, shadow validation и rollback.
