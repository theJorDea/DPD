from __future__ import annotations

import base64
import copy
import contextlib
import io
import json
import textwrap
from pathlib import Path

import numpy as np


def markdown(source: str) -> dict:
    rendered = textwrap.dedent(source).strip("\n")
    # Google Colab's Markdown renderer is most reliable with $$...$$ display
    # delimiters.  Normalize the source once instead of relying on a local
    # Jupyter renderer's handling of \[...\].
    rendered = rendered.replace("\\[", "$$").replace("\\]", "$$")
    rendered = rendered.replace("\\(", "$").replace("\\)", "$")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": rendered.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


cells = [
    markdown(
        r'''
        # Комплексный spline-memory DPD: воспроизводимый BlackBox-эксперимент

        **Краткий итог.** Notebook воспроизводит отдельный эксперимент на
        BlackBox capture: обучает вспомогательный GMP evaluator усилителя,
        замораживает его, калибрует причинный комплексный spline-memory DPD и
        проверяет каскад в правильном направлении. Итоговые числа формируются
        при полном запуске, а не перенесены из презентации.

        **Статус доказательности: МОДЕЛЬ PA.** Это не `DPA_200MHz` и не
        `APA_200MHz`. После подачи предыскажённого сигнала физический PA не
        измерялся, поэтому результаты не доказывают физическую линеаризацию,
        превосходство над OpenDPD, выполнение требований Huawei или готовность
        к внедрению.

        > **Конфиденциальность данных.** Эта самодостаточная версия содержит
        > встроенную копию измеренных train/validation-данных. Перед внешней
        > отправкой необходимо подтвердить разрешение на распространение capture.

        | Пункт | Результат полного запуска |
        |---|---|
        | Набор данных | BlackBox capture, train 92 000 / validation 23 000 |
        | PA evaluator | causal GMP, 189 комплексных коэффициентов; `__PA_FIDELITY__ dB` NMSE |
        | DPD | `__BRANCHES__`, `K=__KNOT_COUNT__`, ridge `1e-4` |
        | Validation без / с DPD | `__NO_DPD_NMSE__ / __DPD_NMSE__ dB` pooled NMSE |
        | Улучшение | `__IMPROVEMENT__ dB` на frozen PA evaluator |
        | Fast path | `__MUL__ MUL + __ADD__ ADD + __MAG__ magnitude + __LUT__ LUT reads` / complex sample |
        | Fixed-point | в этом BlackBox notebook не проверен |

        ```text
        x_desired → DPD → frozen PA evaluator → y_hat ≈ g·x_desired
        ```

        Правильный путь проверки:

        ```text
        desired x_val → frozen DPD → frozen PA evaluator → y_hat → сравнение с g·x_val
        ```

        Измеренный `y_val` в DPD при этой проверке не подаётся.
        ''',
    ),
    markdown(
        r'''
        # 0. Теоретический учебник кампании −50 dB

        Этот раздел — самодостаточный учебник по всей теории и математике
        кампании улучшения DPD. Исполняемого кода здесь нет; для синтетических
        демонстраций см. `docs/notebooks/DPD_theory_campaign.ipynb`.

        ## 0.1 Физика проблемы: нелинейность и память усилителя

        Идеальный PA: `y = g·x`. Реальный PA (1) компрессирует крупные
        отсчёты и порождает гармоники — нелинейность, удобно описываемая
        полиномом от амплитуды

        \[
        y[n] = \sum_k a_k\, x[n]\,|x[n]|^{2k},
        \]

        и (2) помнит предыдущие отсчёты (фильтры, ёмкости, тепло) — это
        оператор с памятью на десятки отсчётов. В частотной области
        нелинейность даёт спектральный рост: энергия выплёскивается в
        соседние каналы, что жёстко нормируют стандарты.

        ## 0.2 NMSE: метрика кампании

        \[
        \mathrm{NMSE} = \frac{\sum_n |y[n] - g\,x[n]|^2}{\sum_n |g\,x[n]|^2},
        \qquad \mathrm{dB} = 10\log_{10}\mathrm{NMSE}.
        \]

        Это доля мощности сигнала, испорченная ошибкой; каждые −3 dB —
        ошибка вдвое меньше. Точки отсчёта: −28.3 dB — старт кампании,
        −31.9 — её финал (DPA), −50 — цель Huawei (0.001 % ошибки).
        Гейн `g` подгоняется комплексными наименьшими квадратами
        (`g = ⟨x,y⟩/⟨x,x⟩`), первые `W` отсчётов отбрасываются (warmup —
        заполнение памяти моделей).

        Важно: NMSE каскада зависит от модели PA, через которую он измерен.
        Исторические числа репозитория (−30.5/−32.4) сняты через слабую
        MP-модель; через сильный GMP-суррогат тот же DPD даёт −28.3.

        ## 0.3 Каскад DPD и рост пиков

        Каскад: `x → u = D(x) → PA → y = P(u) ≈ g·x`. DPD обязан
        предрастягивать сигнал (PA сжимает пики — DPD их заранее растит),
        поэтому PAPR драйва выше PAPR входа. Если пик драйва выходит за
        диапазон, виденный моделью PA при обучении, модель экстраполирует
        и врёт — отсюда «опора»: `max|u| ≤ max|x|·(1+headroom)`.

        ## 0.4 Модель DPD: сплайновая память

        Выход DPD — сумма веток:

        \[
        u[n] = \sum_{\text{ветки}} x[n-m]\cdot C\big(|x[n-d]|\big).
        \]

        `x[n−m]` — задержанный отсчёт (память по сигналу), `|x[n−d]|` —
        огибающая с задержкой d, `C(·)` — сплайн: ломаная, заданная
        значениями в узлах (knots). Коэффициенты модели — узловые значения
        (комплексные). Baseline: 3 ветки × 24 узла = 72 коэффициента; финал
        кампании: 13 × 24 = 312. Узлы лучше ставить по квантилям амплитуды
        сигнала — равномерные проигрывают на 4–20 dB, тратя разрешение там,
        где сигнала нет.

        Фазовая эквивариантность: при `x → j·x` огибающая `|j·x| = |x|`
        не меняется (I²+Q² коммутативна), поэтому выход каждой ветки
        поворачивается ровно на 90°. Модель не привносит асимметрии I/Q.
        (Float-комплексное умножение не бит-точно под поворотом на уровне
        последнего бита IEEE; в целочисленном ядре перестановки точны.)

        ## 0.5 Обучение: LS и ILA

        Если модель линейна по параметрам (`u[n] = Σ θ_j φ_j[n]`, где
        `φ_j` — словарные функции), обучение — наименьшие квадраты:

        \[
        \theta = (\Phi^H\Phi + \lambda I)^{-1}\Phi^H t.
        \]

        Прямая постановка «минимизируй ошибку каскада» содержит PA внутри.
        Трюк репозитория — ILA (indirect learning): обучить обратную модель
        `G(y/g) ≈ x/g` на измеренных парах; G ≈ P⁻¹ и есть DPD.

        ## 0.6 Суррогаты («приборы») и потолок фиделити

        Для оценки кандидата нужен PA. Физического нет — есть модель
        «вперёд», обученная на записях: GMP
        `ŷ[n] = Σ c_{m,d,k} x[n−m] (|x[n−d]|²)^k` — снова LS. Её точность
        против измерений — фиделити (DPA −35.4, APA −38.5 dB). Каскад
        меряется как NMSE(P_модель(D(x)), g·x), а собственная ошибка модели
        не зависит от входа — её не скомпенсировать. Следствие: каскад не
        может превысить фиделити прибора. **Фиделити суррогата = потолок
        видимости.** Идеальный DPD покажет ≈ −35, не −50.

        ## 0.7 Прямое обучение Гаусса–Ньютона

        Минимизируем `‖g·x − P(D_θ(x))‖²` итеративно. Линеаризация:

        \[
        P(D_{\theta+\Delta}(x)) \approx P(D_\theta(x)) + J\Delta,
        \qquad J_{n,i} \approx \frac{P(D_{\theta+\varepsilon e_i}(x))_n - P(D_\theta(x))_n}{\varepsilon}.
        \]

        Якобиан — конечными разностями (313 прогонов модели на 288
        коэффициентов). Шаг — демпфированный LS:
        `min_Δ ‖JΔ − r‖² + λ‖Δ‖²` с перебором λ и длины шага на
        непересекающихся парах блоков («ротации»). Joint stacked objective —
        стек якобианов и остатков обоих суррогатов: один LS, общие
        коэффициенты, оба прибора сразу.

        ## 0.8 Ловушка: подгонка под суррогат

        Оптимизация под один суррогат компенсирует его личные ошибки, а не
        реальную нелинейность. Доказано экспериментально: кандидат дал
        +0.78 dB через суррогат A и −1.3 dB через независимый B. Причина:
        ошибки двух суррогатов частично коррелированы, но не совпадают, и
        оптимизатор цепляется за несовпадающую часть.

        ## 0.9 Защитный протокол

        1. Worst-case ранжирование: балл кандидата = max(NMSE_A, NMSE_B);
           принимается только улучшение обоих.
        2. Консенсус: целевой вектор словарных кандидатов — средний остаток
           двух суррогатов (эквивалент одного LS по двум системам).
        3. Разнесённые блоки: fit / advisor / selection — непересекающиеся
           куски train; validation — только отчётный; тест не открывался.
        4. Опора: ограничение пика драйва.

        ## 0.10 Ёмкость — главный прорыв

        Проверка «а не мало ли у DPD веток?»: та же ILA-машина по семействам
        4–13 веток, отбор по протоколу 0.9, затем GN-полировка. Лестница
        (DPA): −28.29 → −29.83 (5 веток) → −30.17 (7) → −31.13 (9) →
        −31.43 (12) → −31.93 (13 веток + joint GN). Затем плато: последняя
        ветка +0.08 dB; композиты поверх финала перестали проходить
        кросс-гейт — сплайн впитал их роль.

        ## 0.11 Fixed-point

        Деплой на 16-битной арифметике: масштаб по пику коэффициентов
        (guard 1.001), округление ties-to-even, насыщение. Гейт: 0
        насыщений; деградация каскада ≤ 0.05 dB через оба суррогата;
        потоковая обработка бит-в-бит; эквивариантность 90° не хуже float.

        ## 0.12 Информационный потолок

        Цепочка измеренных потолков: записи шумные (−39/−40 dB) → суррогаты
        точны до −35.4/−38.5 → идеальный DPD покажет ≈ −35 → мы на −31.9
        (≈91 % доступного диапазона). Чтобы доказать −50, нужны зерно и
        прибор на −55…−60: повторный физический захват (VSA +15 dB,
        дробная синхронизация ≥ 1/64, DC/IQ-коррекции) и/или GPU-нейросуррогат
        фиделити ≥ −55 (обёртки в репозитории).

        ## 0.13 Итоги кампании

        DPA: −28.29 → −31.93 (worst-case A/B; 312 коэфф., ≈81 MUL;
        fixed-point PASS; +3.64 dB). APA: −28.15 → −31.37 (216 коэфф.,
        63 MUL; PASS; +3.22 dB). Закрытые направления: ёмкость (насыщение),
        стратегии узлов, joint GN (датасет-зависим), композиты (HOLD),
        широкие словари (HOLD), ILC (негатив), нейросудья на CPU (мало),
        SPH-судья (без запаса).
        ''',
    ),
    markdown(
        r'''
        ## 1. Цель, два этапа и границы утверждений

        Комплексный отсчёт основной полосы имеет вид

        \[
        x[n] = I[n] + jQ[n].
        \]

        Реальный PA реализует нелинейный оператор `F`. Мы хотим найти такой
        предыскажатель `D`, чтобы

        \[
        F(D(x[n])) \approx g\,x[n].
        \]

        **Этап A — идентификация PA evaluator.** По измеренным парам
        $x\rightarrow physical\ PA\rightarrow y$ на train обучается
        $\hat P(x)\approx y$. После fit evaluator замораживается.

        **Этап B — калибровка DPD.** Коэффициенты DPD оцениваются на train, после
        чего проверяется только deployment-like путь
        $x_{desired}\rightarrow DPD\rightarrow\hat P\rightarrow\hat y$.
        Круговая реконструкция известного `y_val` не используется.

        В метаданных BlackBox неизвестны частота дискретизации, несущая,
        waveform и официальные границы полос. Поэтому PSD ниже — только
        визуальная диагностика в циклах на отсчёт; ACLR/ACPR не вычисляется.
        ''',
    ),
    code(
        r'''
        # Базовые импорты и настройки воспроизводимости.
        from __future__ import annotations

        import base64
        import io
        import hashlib
        import json
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np

        np.random.seed(0)
        np.set_printoptions(precision=5, suppress=True)
        plt.style.use("seaborn-v0_8-whitegrid")
        print("NumPy:", np.__version__)
        ''',
    ),
    code(
        r'''
        # Встроенные train/validation массивы BlackBoxData (сжатый NPZ, base64).
        # Генератор notebook подставляет payload перед сохранением файла.
        EMBEDDED_DATA_B64 = """__EMBEDDED_PAYLOAD__"""


        def load_embedded_splits() -> dict[str, np.ndarray]:
            """Decode the portable train/validation payload from this notebook.

            Returns:
                A dictionary with raw complex ``train_x``, ``train_y``,
                ``val_x`` and ``val_y`` arrays plus a human-readable source.
            """
            binary = base64.b64decode(EMBEDDED_DATA_B64.encode("ascii"))
            with np.load(io.BytesIO(binary), allow_pickle=False) as payload:
                result = {name: np.asarray(payload[name], dtype=np.complex128) for name in (
                    "train_x_raw", "train_y_raw", "val_x_raw", "val_y_raw"
                )}
            result["source"] = "embedded BlackBoxData train/validation payload"
            return result


        print("embedded payload ready; source data are portable with this notebook")
        ''',
    ),
    markdown(
        r'''
        ## 2. Данные, загрузка и дисциплина разбиения

        Хронологическое разбиение зафиксировано заранее: 92 000 отсчётов train
        (`5000:97000`) и 23 000 validation (`97000:120000`). Sealed test не
        встроен и не открывается. Validation используется для сравнения
        архитектур и числа узлов, поэтому её итоговая метрика не является
        независимой test-оценкой.

        При `USE_EMBEDDED_DATA=True` используется встроенный payload. Для MAT
        установите `False` и задайте `MAT_PATH`; ожидаются комплексные массивы
        `x`, `y`, `eRef` одинаковой длины. Для CSV задайте `SELECTION_DIR` с
        файлами `train_input.csv`, `train_output.csv`, `val_input.csv`,
        `val_output.csv`, каждый с колонками `I,Q` и заголовком.
        ''',
    ),
    code(
        r'''
        # Фиксированные параметры запуска. В portable-режиме менять пути не нужно:
        # данные уже находятся внутри notebook.
        PROJECT_CANDIDATES = [Path.cwd(), *Path.cwd().parents]
        PROJECT_ROOT = next(
            (p for p in PROJECT_CANDIDATES
             if (p / "data/private/blackbox_v3/selection").is_dir()),
            Path.cwd(),
        )
        SELECTION_DIR = PROJECT_ROOT / "data/private/blackbox_v3/selection"
        USE_EMBEDDED_DATA = True
        MAT_PATH = None  # пример: Path("/content/BlackBoxData.mat")
        OPENDPD_JSON = PROJECT_ROOT / "vendor/OpenDPD/benchmark/results/benchmark_report_results.json"
        print("external project data directory detected =", SELECTION_DIR.is_dir())
        print("USE_EMBEDDED_DATA =", USE_EMBEDDED_DATA)
        ''',
    ),
    code(
        r'''
        def read_iq_csv(path: Path) -> np.ndarray:
            """Read a two-column ``I,Q`` CSV and return ``I + jQ`` samples.

            Args:
                path: CSV file with a header and two numeric columns.

            Returns:
                One-dimensional finite complex128 vector.
            """
            raw = np.loadtxt(path, delimiter=",", skiprows=1)
            if raw.ndim != 2 or raw.shape[1] < 2:
                raise ValueError(f"Expected I,Q columns in {path}")
            signal = np.asarray(raw[:, 0] + 1j * raw[:, 1], dtype=np.complex128)
            if not np.all(np.isfinite(signal)):
                raise ValueError(f"Non-finite samples in {path}")
            return signal


        def load_mat_splits(path: Path) -> dict[str, np.ndarray]:
            """Load the optional MAT fallback and apply the frozen split.

            The source variables are validated, while ``eRef`` is retained only
            for the shape check and is intentionally not used by the model.
            """
            from scipy.io import loadmat

            payload = loadmat(path)
            vectors = {}
            for name in ("x", "y", "eRef"):
                if name not in payload:
                    raise ValueError(f"MAT file lacks {name}")
                value = np.asarray(payload[name]).reshape(-1)
                if not np.iscomplexobj(value):
                    raise ValueError(f"MAT variable {name} is not complex")
                vectors[name] = np.asarray(value, dtype=np.complex128)
            if len({v.size for v in vectors.values()}) != 1:
                raise ValueError("x, y and eRef must have equal lengths")
            return {
                "train_x_raw": vectors["x"][5000:97000],
                "train_y_raw": vectors["y"][5000:97000],
                "val_x_raw": vectors["x"][97000:120000],
                "val_y_raw": vectors["y"][97000:120000],
                "source": path,
            }


        def load_blackbox_splits(selection_dir: Path, mat_path: Path | None) -> dict[str, np.ndarray]:
            """Load embedded data or explicit CSV/MAT files with clear errors."""
            if USE_EMBEDDED_DATA:
                return load_embedded_splits()
            files = {
                "train_x_raw": selection_dir / "train_input.csv",
                "train_y_raw": selection_dir / "train_output.csv",
                "val_x_raw": selection_dir / "val_input.csv",
                "val_y_raw": selection_dir / "val_output.csv",
            }
            if all(path.is_file() for path in files.values()):
                result = {name: read_iq_csv(path) for name, path in files.items()}
                result["source"] = selection_dir
                return result
            if mat_path is not None and Path(mat_path).is_file():
                return load_mat_splits(mat_path)
            raise FileNotFoundError(
                "USE_EMBEDDED_DATA=False, но данные не найдены. "
                "Укажите каталог с четырьмя I,Q CSV в SELECTION_DIR или "
                "существующий BlackBoxData.mat в MAT_PATH."
            )


        raw = load_blackbox_splits(SELECTION_DIR, MAT_PATH)
        train_x_raw, train_y_raw = raw["train_x_raw"], raw["train_y_raw"]
        val_x_raw, val_y_raw = raw["val_x_raw"], raw["val_y_raw"]
        assert train_x_raw.size == train_y_raw.size == 92000
        assert val_x_raw.size == val_y_raw.size == 23000
        print("source:", raw["source"])
        print("train:", train_x_raw.shape, "validation:", val_x_raw.shape)
        print("raw train input peak:", np.max(np.abs(train_x_raw)))
        ''',
    ),
    code(
        r'''
        # Alignment, normalization и gain оцениваются только на train.
        def overlap_at_delay(x: np.ndarray, y: np.ndarray, delay: int):
            """Return overlapping vectors when y is delayed by ``delay`` vs x."""
            if delay > 0:
                return x[:-delay], y[delay:]
            if delay < 0:
                return x[-delay:], y[:delay]
            return x, y


        def normalized_alignment_score(x: np.ndarray, y: np.ndarray, delay: int) -> float:
            """Return magnitude of normalized complex correlation at one delay."""
            xa, ya = overlap_at_delay(x, y, delay)
            denominator = np.sqrt(np.vdot(xa, xa).real * np.vdot(ya, ya).real)
            return float(abs(np.vdot(xa, ya)) / denominator)


        ALIGNMENT_DELAYS = np.arange(-32, 33)
        alignment_scores = np.array([
            normalized_alignment_score(train_x_raw, train_y_raw, int(delay))
            for delay in ALIGNMENT_DELAYS
        ])
        integer_delay = int(ALIGNMENT_DELAYS[np.argmax(alignment_scores)])
        peak_index = int(np.argmax(alignment_scores))
        fractional_offset = 0.0
        if 0 < peak_index < alignment_scores.size - 1:
            left_score, peak_score, right_score = alignment_scores[peak_index - 1:peak_index + 2]
            curvature = left_score - 2.0 * peak_score + right_score
            if abs(curvature) > 1e-15:
                fractional_offset = float(0.5 * (left_score - right_score) / curvature)
        fractional_delay_diagnostic = integer_delay + fractional_offset

        # Один train-only integer delay применяется без переоценки на validation.
        train_x_aligned, train_y_aligned = overlap_at_delay(train_x_raw, train_y_raw, integer_delay)
        val_x_aligned, val_y_aligned = overlap_at_delay(val_x_raw, val_y_raw, integer_delay)

        TRAIN_INPUT_PEAK = float(np.max(np.abs(train_x_aligned)))
        train_x = train_x_aligned / TRAIN_INPUT_PEAK
        train_y = train_y_aligned / TRAIN_INPUT_PEAK
        val_x = val_x_aligned / TRAIN_INPUT_PEAK
        val_y = val_y_aligned / TRAIN_INPUT_PEAK

        gain = np.vdot(train_x, train_y) / np.vdot(train_x, train_x)
        print(f"train-only integer delay = {integer_delay} samples")
        print(f"fractional-delay diagnostic = {fractional_delay_diagnostic:+.6f} samples (not applied)")
        print(f"alignment peak score = {alignment_scores[peak_index]:.6f}")
        print(f"train-only scale = {TRAIN_INPUT_PEAK:.6f}")
        print(f"train-only complex gain = {gain.real:.8f} {gain.imag:+.8f}j")
        print("normalized peaks:", *(f"{np.max(np.abs(v)):.6f}" for v in (train_x, train_y, val_x, val_y)))

        fig, ax = plt.subplots(figsize=(9.5, 3.6))
        ax.plot(ALIGNMENT_DELAYS, alignment_scores, marker=".", ms=4)
        ax.axvline(integer_delay, color="tab:red", ls="--", label=f"выбрано: {integer_delay}")
        ax.set(xlabel="задержка y относительно x [отсчёты]", ylabel="|нормированная корреляция|",
               title="Диагностика временного выравнивания только по train")
        ax.legend()
        plt.tight_layout()
        plt.show()
        ''',
    ),
    code(
        r'''
        # Наглядная проверка входа/выхода PA.
        sample = slice(None, None, 30)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        axes[0].scatter(train_x.real[sample], train_y.real[sample], s=2, alpha=0.25, label="train")
        axes[0].scatter(val_x.real[sample], val_y.real[sample], s=2, alpha=0.25, label="validation")
        axes[0].set(xlabel="Re{x}", ylabel="Re{y}", title="I/Q-компонента: вход → выход PA")
        axes[0].legend()
        axes[1].scatter(np.abs(train_x)[sample], np.abs(train_y)[sample], s=2, alpha=0.25)
        r = np.linspace(0, max(np.max(np.abs(train_x)), np.max(np.abs(val_x))), 200)
        axes[1].plot(r, abs(gain) * r, color="tab:orange", label="|g|·|x| (идеал)")
        axes[1].set(xlabel="|x|", ylabel="|y|", title="AM/AM: компрессия усилителя")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        ''',
    ),
    markdown(
        r'''
        ## 3. Метрики

        Для основной оценки используем pooled complex NMSE:

        \[
        \mathrm{NMSE}_{dB} = 10\log_{10}
        \frac{\mathbb{E}|\hat y[n]-y_\mathrm{ref}[n]|^2}
             {\mathbb{E}|y_\mathrm{ref}[n]|^2}.
        \]

        Чем меньше (более отрицательно) NMSE, тем лучше. Первые отсчёты после
        сброса причинного состояния исключаются из оценки.

        Временное отношение RMS-ошибки к RMS-эталону математически даёт то же
        dB-число, что pooled NMSE, поэтому оно не выдаётся за отдельную EVM.
        Стандартная QAM EVM потребовала бы демодуляции, синхронизации,
        эквализации и известных идеальных символов; этих данных здесь нет.
        ''',
    ),
    code(
        r'''
        def pooled_nmse_db(estimate: np.ndarray, reference: np.ndarray, warmup: int = 0) -> float:
            """Return pooled complex NMSE in dB after a causal warm-up.

            Args:
                estimate: Predicted complex output.
                reference: Desired/reference complex output.
                warmup: Number of record-start samples excluded from scoring.

            Returns:
                ``10*log10(MSE / reference_power)``.
            """
            estimate = np.asarray(estimate, dtype=np.complex128)
            reference = np.asarray(reference, dtype=np.complex128)
            if estimate.shape != reference.shape or estimate.ndim != 1:
                raise ValueError("metric vectors must be equal 1-D arrays")
            e = estimate[warmup:] - reference[warmup:]
            ref = reference[warmup:]
            return float(10.0 * np.log10(np.mean(np.abs(e) ** 2) / np.mean(np.abs(ref) ** 2)))


        def normalized_rms_error_db(estimate: np.ndarray, reference: np.ndarray, warmup: int = 0) -> float:
            """Return time-domain normalized RMS error; equal to NMSE in dB."""
            e = np.asarray(estimate)[warmup:] - np.asarray(reference)[warmup:]
            ref = np.asarray(reference)[warmup:]
            return float(20.0 * np.log10(np.sqrt(np.mean(np.abs(e) ** 2)) / np.sqrt(np.mean(np.abs(ref) ** 2))))


        def summary(signal: np.ndarray) -> dict[str, float]:
            """Summarize RMS amplitude, peak amplitude and PAPR of a signal."""
            power = float(np.mean(np.abs(signal) ** 2))
            peak = float(np.max(np.abs(signal)))
            return {
                "rms": float(np.sqrt(power)),
                "peak": peak,
                "papr_db": float(10.0 * np.log10(peak * peak / power)),
            }


        def metric_row(name: str, estimate: np.ndarray, reference: np.ndarray, warmup: int) -> dict:
            """Build one readable report row from the common metric functions."""
            return {
                "name": name,
                "nmse_db": pooled_nmse_db(estimate, reference, warmup),
                **summary(estimate),
            }


        # Algebraic identity check; this is why no independent EVM column is shown.
        _metric_probe = np.array([1 + 1j, 0.5 - 0.2j])
        assert abs(pooled_nmse_db(0.9 * _metric_probe, _metric_probe)
                   - normalized_rms_error_db(0.9 * _metric_probe, _metric_probe)) < 1e-12
        ''',
    ),
    markdown(
        r'''
        ## 4. Frozen GMP-модель PA

        Для evaluator используем causal generalized memory polynomial:

        \[
        \begin{aligned}
        \hat y[n] ={}&\sum_{k,q}a_{kq}x[n-q]|x[n-q]|^k\\
        &+\sum_{k,q,l}b_{kql}x[n-q]|x[n-q-l]|^k\\
        &+\sum_{k,q,l}c_{kql}x[n-q]|x[n-q+l]|^k.
        \end{aligned}
        \]

        Последняя группа ограничена условием `lead ≤ signal_delay`, поэтому
        в ноутбуке нет look-ahead. Коэффициенты находятся комплексной ridge-
        регрессией на train; после этого PA больше не дообучается.
        ''',
    ),
    code(
        r'''
        def causal_delay(signal: np.ndarray, delay: int) -> np.ndarray:
            """Apply a causal integer delay with zero padding at record start."""
            out = np.zeros_like(signal, dtype=np.complex128)
            if delay == 0:
                out[:] = signal
            elif delay > 0:
                out[delay:] = signal[:-delay]
            return out


        def gmp_terms(ka=9, la=9, kb=3, lb=7, mb=3, kc=3, lc=7, mc=3):
            """Enumerate causal GMP terms as ``(branch, order, q, envelope_q)``."""
            terms = []
            for exponent in range(ka):
                for q in range(la):
                    terms.append(("aligned", exponent, q, q))
            for exponent in range(1, kb + 1):
                for q in range(lb):
                    for lag in range(1, mb + 1):
                        terms.append(("lagging", exponent, q, q + lag))
            for exponent in range(1, kc + 1):
                for q in range(lc):
                    for lead in range(1, mc + 1):
                        if lead <= q:  # causal-leading policy
                            terms.append(("leading", exponent, q, q - lead))
            return tuple(terms)


        def gmp_design(signal: np.ndarray, terms: tuple) -> np.ndarray:
            """Construct the dense complex GMP design matrix in term order."""
            signal = np.asarray(signal, dtype=np.complex128)
            columns = []
            for _branch, exponent, signal_delay, envelope_delay in terms:
                delayed = causal_delay(signal, signal_delay)
                if exponent == 0:
                    columns.append(delayed)
                else:
                    envelope = np.abs(causal_delay(signal, envelope_delay))
                    columns.append(delayed * envelope ** exponent)
            return np.column_stack(columns)


        def fit_complex_ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
            """Fit complex coefficients with column scaling and ridge regularization.

            The data term is normalized by the number of samples, matching the
            frozen PA-selection protocol used for the BlackBox experiment.
            """
            design = np.asarray(design, dtype=np.complex128)
            target = np.asarray(target, dtype=np.complex128)
            column_rms = np.sqrt(np.mean(np.abs(design) ** 2, axis=0))
            if np.any(column_rms <= 0) or not np.all(np.isfinite(column_rms)):
                raise ValueError("invalid design column scale")
            # The project protocol minimizes mean squared error + ridge:
            # ||D c - q||²/N + ridge*||c||².  Dividing both data and target
            # by sqrt(N) is therefore part of the definition, not cosmetic
            # numerical scaling.
            normalization = np.sqrt(float(design.shape[0]))
            scaled = design / column_rms / normalization
            normalized_target = target / normalization
            if ridge > 0:
                regularizer = np.sqrt(ridge) * np.eye(
                    scaled.shape[1], dtype=np.complex128
                )
                augmented = np.vstack((scaled, regularizer))
                augmented_target = np.concatenate(
                    (normalized_target, np.zeros(scaled.shape[1], dtype=np.complex128))
                )
            else:
                augmented, augmented_target = scaled, normalized_target
            coeff_scaled = np.linalg.lstsq(augmented, augmented_target, rcond=None)[0]
            return coeff_scaled / column_rms


        # Frozen evaluator configuration selected on the BlackBox validation split.
        # The PA is auxiliary: its complexity is not the deployment DPD budget.
        pa_terms = gmp_terms()
        assert len(pa_terms) == 189
        # Forward identification: measured PA input x -> measured output y.
        started = time.perf_counter()
        pa_design = gmp_design(train_x, pa_terms)
        pa_coefficients = fit_complex_ridge(pa_design, train_y, ridge=1e-10)
        pa_fit_seconds = time.perf_counter() - started
        del pa_design
        pa_train_hat = gmp_design(train_x, pa_terms) @ pa_coefficients
        pa_val_design = gmp_design(val_x, pa_terms)
        pa_val_hat = pa_val_design @ pa_coefficients
        del pa_val_design
        print(f"GMP terms: {len(pa_terms)} complex coefficients")
        print(f"fit time: {pa_fit_seconds:.2f} s")
        print(f"PA validation fidelity: {pooled_nmse_db(pa_val_hat, val_y, warmup=11):.3f} dB NMSE")
        ''',
    ),
    code(
        r'''
        # До DPD: каскад frozen PA на желаемом validation input.
        PA_WARMUP = 9
        DPD_WARMUP = 2
        CASCADE_WARMUP = PA_WARMUP + DPD_WARMUP
        ideal_val = gain * val_x
        measured_no_dpd = metric_row("measured y (diagnostic)", val_y, ideal_val, CASCADE_WARMUP)
        surrogate_no_dpd = metric_row("frozen PA, no DPD", pa_val_hat, ideal_val, CASCADE_WARMUP)
        print(measured_no_dpd)
        print(surrogate_no_dpd)
        ''',
    ),
    markdown(
        r'''
        ## 5. Архитектура spline-memory DPD

        DPD имеет три causal-ветви:

        \[
        z[n] = \sum_{m=0}^{2} x[n-m]\,C_m(|x[n-m]|).
        \]

        Здесь $x[n]$ — желаемый комплексный I/Q-отсчёт, $z[n]$ — отсчёт перед
        PA, а $C_m(r)$ — комплексная коррекция ветви задержки $m$. Модуль
        $C_m$ задаёт амплитуду вклада, аргумент — фазу.

        Каждая комплексная функция `C_m(r)` — кусочно-линейный сплайн с общими
        квантильными узлами `r_0,…,r_{K-1}`:

        \[
        C_m(r)=(1-\alpha)C_{m,k}+\alpha C_{m,k+1},
        \qquad
        \alpha=\frac{r-r_k}{r_{k+1}-r_k}.
        \]

        На каждом отсчёте активны только два соседних коэффициента. При общем
        повороте $x'[n]=x[n]e^{j\varphi}$ модули не меняются:
        $|x'[n-m]|=|x[n-m]|$. Поэтому каждый член получает один множитель
        $e^{j\varphi}$ и сохраняется фазовая эквивариантность:

        \[
        D(xe^{j\varphi})=D(x)e^{j\varphi}.
        \]

        Калибровка ILA — только способ получить коэффициенты:

        \[
        u[n]=y[n]/g,\qquad
        \min_C\|\Phi(u)C-x\|_2^2+\lambda\|C\|_2^2.
        \]

        На validation/test deployment-вход меняется на `x_desired`; measured
        `y_val` не подаётся внутрь DPD.

        В основной DPA/APA конфигурации проекта исследовалась другая связь
        $x[n-m]C_m(|x[n]|)$, то есть ветви `((0,0),(1,0),(2,0))`. Ниже обе
        топологии проверяются на BlackBox. Общая текущая амплитуда здесь не
        принимается заранее как лучшая.
        ''',
    ),
    code(
        r'''
        def make_quantile_knots(signal: np.ndarray, count: int) -> np.ndarray:
            """Place strictly increasing interpolation knots by amplitude quantiles."""
            radius = np.abs(np.asarray(signal, dtype=np.complex128))
            unit = np.linspace(0.0, 1.0, count)
            knots = np.quantile(radius, unit)
            knots[0] = 0.0
            knots[-1] = np.max(radius)
            knots = np.unique(knots)
            if knots.size < 2 or np.any(np.diff(knots) <= 0):
                raise ValueError("quantile knots are not strictly increasing")
            return knots


        def spline_coordinates(radius: np.ndarray, knots: np.ndarray):
            """Return left-knot indices and local linear interpolation weights."""
            clipped = np.clip(np.asarray(radius, dtype=float), knots[0], knots[-1])
            left = np.searchsorted(knots, clipped, side="right") - 1
            left = np.clip(left, 0, knots.size - 2)
            weight = (clipped - knots[left]) / (knots[left + 1] - knots[left])
            return left.astype(np.int64), weight


        def spline_memory_design(signal: np.ndarray, knots: np.ndarray, branches: tuple[tuple[int, int], ...]) -> np.ndarray:
            """Build ``Phi[n,bK+k] = x[n-m] B_k(|x[n-d]|)`` for spline-memory DPD."""
            signal = np.asarray(signal, dtype=np.complex128)
            columns = []
            for signal_delay, envelope_delay in branches:
                delayed_signal = causal_delay(signal, signal_delay)
                delayed_envelope = causal_delay(signal, envelope_delay)
                left, weight = spline_coordinates(np.abs(delayed_envelope), knots)
                basis = np.zeros((signal.size, knots.size), dtype=float)
                rows = np.arange(signal.size)
                basis[rows, left] = 1.0 - weight
                basis[rows, left + 1] += weight
                columns.extend((delayed_signal[:, None] * basis).T)
            return np.column_stack(columns)


        def fit_spline_memory_dpd(calibration_input, target, knots, branches, ridge):
            """Fit the ILA spline-memory coefficients on the causal steady region."""
            warmup = max(max(b) for b in branches)
            design = spline_memory_design(calibration_input, knots, branches)[warmup:]
            # The frozen project protocol intentionally uses the raw spline
            # dictionary here (no column scaling), with the same normalized
            # objective and ridge as factorize_spline_group().
            normalization = np.sqrt(float(design.shape[0]))
            solve_design = design / normalization
            solve_target = target[warmup:] / normalization
            augmented = np.vstack(
                (solve_design, np.sqrt(ridge) * np.eye(design.shape[1], dtype=np.complex128))
            )
            augmented_target = np.concatenate(
                (solve_target, np.zeros(design.shape[1], dtype=np.complex128))
            )
            flat = np.linalg.lstsq(augmented, augmented_target, rcond=None)[0]
            return flat.reshape(len(branches), knots.size), warmup


        def predict_spline_memory(signal, knots, branches, coefficients):
            """Evaluate the causal spline-memory DPD for one independent record."""
            signal = np.asarray(signal, dtype=np.complex128)
            output = np.zeros(signal.size, dtype=np.complex128)
            for branch_index, (signal_delay, envelope_delay) in enumerate(branches):
                delayed_signal = causal_delay(signal, signal_delay)
                delayed_envelope = causal_delay(signal, envelope_delay)
                left, weight = spline_coordinates(np.abs(delayed_envelope), knots)
                c = coefficients[branch_index]
                correction = c[left] + weight * (c[left + 1] - c[left])
                output += delayed_signal * correction
            return output
        ''',
    ),
    code(
        r'''
        # Model selection uses train for fit and validation for ranking.
        # The smallest K within 0.01 dB of the best validation NMSE is retained.
        TOPOLOGIES = {
            "общая текущая амплитуда": ((0, 0), (1, 0), (2, 0)),
            "согласованные задержки": ((0, 0), (1, 1), (2, 2)),
        }
        K_CANDIDATES = (4, 8, 12, 16, 24, 32)
        RIDGE = 1e-4
        NMSE_TOLERANCE_DB = 0.01
        ila_input = train_y / gain
        ideal_val = gain * val_x
        no_dpd_nmse_db = pooled_nmse_db(pa_val_hat, ideal_val, CASCADE_WARMUP)
        selection_rows = []

        for topology_name, candidate_branches in TOPOLOGIES.items():
            for candidate_k in K_CANDIDATES:
                candidate_knots = make_quantile_knots(ila_input, candidate_k)
                started = time.perf_counter()
                candidate_coefficients, candidate_warmup = fit_spline_memory_dpd(
                    ila_input, train_x, candidate_knots, candidate_branches, RIDGE
                )
                fit_seconds = time.perf_counter() - started
                candidate_drive = predict_spline_memory(
                    val_x, candidate_knots, candidate_branches, candidate_coefficients
                )
                candidate_output = gmp_design(candidate_drive, pa_terms) @ pa_coefficients
                selection_rows.append({
                    "topology": topology_name,
                    "branches": candidate_branches,
                    "K": candidate_k,
                    "nmse_db": pooled_nmse_db(candidate_output, ideal_val, CASCADE_WARMUP),
                    "real_coefficients": 2 * len(candidate_branches) * candidate_k,
                    "fit_seconds": fit_seconds,
                    "finite": bool(np.all(np.isfinite(candidate_coefficients))),
                    "max_drive": float(np.max(np.abs(candidate_drive))),
                })

        best_nmse = min(row["nmse_db"] for row in selection_rows)
        eligible = [row for row in selection_rows if row["nmse_db"] <= best_nmse + NMSE_TOLERANCE_DB]
        selected = min(eligible, key=lambda row: (row["real_coefficients"], row["nmse_db"]))
        branches = selected["branches"]
        knot_count = selected["K"]
        ridge = RIDGE

        print(f"no DPD: {no_dpd_nmse_db:.6f} dB")
        print(f"selection rule: smallest model within {NMSE_TOLERANCE_DB:.2f} dB of best")
        print(f"{'topology':29s} {'K':>3s} {'NMSE [dB]':>11s} {'real coef':>10s} {'fit [s]':>9s}")
        for row in selection_rows:
            print(f"{row['topology']:29s} {row['K']:3d} {row['nmse_db']:11.6f} "
                  f"{row['real_coefficients']:10d} {row['fit_seconds']:9.4f}")
        print("selected:", selected)

        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        for topology_name in TOPOLOGIES:
            rows = [row for row in selection_rows if row["topology"] == topology_name]
            ax.plot([row["K"] for row in rows], [row["nmse_db"] for row in rows],
                    marker="o", label=topology_name)
        ax.axhline(no_dpd_nmse_db, color="tab:red", ls="--", label="без DPD")
        ax.scatter([knot_count], [selected["nmse_db"]], s=110, facecolors="none",
                   edgecolors="black", linewidths=2, label="выбрано")
        ax.set(xlabel="число узлов K", ylabel="validation cascade NMSE [dB]",
               title="Выбор топологии и числа узлов (ниже — лучше)")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Refit exactly the chosen configuration on train; deployment receives val_x.
        knots = make_quantile_knots(ila_input, knot_count)
        dpd_coefficients, dpd_fit_warmup = fit_spline_memory_dpd(
            ila_input, train_x, knots, branches, ridge
        )
        predistorted_val = predict_spline_memory(val_x, knots, branches, dpd_coefficients)
        cascade_dpd = gmp_design(predistorted_val, pa_terms) @ pa_coefficients
        dpd_metrics = metric_row("зафиксированный spline-memory DPD", cascade_dpd, ideal_val, CASCADE_WARMUP)
        no_dpd_mse = np.mean(np.abs(pa_val_hat[CASCADE_WARMUP:] - ideal_val[CASCADE_WARMUP:]) ** 2)
        dpd_mse = np.mean(np.abs(cascade_dpd[CASCADE_WARMUP:] - ideal_val[CASCADE_WARMUP:]) ** 2)
        improvement_db = float(10 * np.log10(no_dpd_mse / dpd_mse))
        print("branches:", branches, "knots:", knots.size, "ridge:", ridge)
        print("coefficients:", dpd_coefficients.shape, "fit warmup:", dpd_fit_warmup)
        print(dpd_metrics)
        print(f"improvement over frozen-PA no-DPD: {improvement_db:.3f} dB")
        ''',
    ),
    markdown(
        r'''
        **Результат выбора.** На BlackBox общая текущая амплитуда
        `((0,0),(1,0),(2,0))` ухудшает каскад относительно режима без DPD при
        всех проверенных `K`. Это полезный отрицательный результат: архитектура,
        успешная на DPA/APA, не переносится автоматически на другой capture.
        Выбрана топология согласованных задержек. `K=24` — минимальная модель в
        пределах 0,01 dB от лучшего `K=32`, а не безусловно оптимальное число
        узлов. Поскольку выбор сделан по validation, нужен отдельный sealed test
        или физический PA для независимого финального утверждения.
        ''',
    ),
    code(
        r'''
        # AM/AM и AM/PM residual после каскада. Идеальный выход — g*x.
        n = slice(None, None, 25)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].scatter(np.abs(val_x)[n], np.abs(pa_val_hat)[n], s=3, alpha=0.22, label="no DPD")
        axes[0].scatter(np.abs(val_x)[n], np.abs(cascade_dpd)[n], s=3, alpha=0.22, label="spline-memory DPD")
        axes[0].plot(np.abs(val_x)[n], np.abs(ideal_val)[n], ".", ms=1.0, color="black", alpha=0.18, label="ideal g·x")
        axes[0].set(xlabel="|x desired|", ylabel="|y cascade|", title="AM/AM: приближение идеала")
        axes[0].legend(markerscale=3)
        nonzero = np.abs(ideal_val) > 1e-12
        phase_no = np.degrees(np.angle(pa_val_hat[nonzero] / ideal_val[nonzero]))
        phase_dpd = np.degrees(np.angle(cascade_dpd[nonzero] / ideal_val[nonzero]))
        radius_nonzero = np.abs(val_x[nonzero])
        axes[1].scatter(radius_nonzero[n], phase_no[n], s=3, alpha=0.20, label="без DPD")
        axes[1].scatter(radius_nonzero[n], phase_dpd[n], s=3, alpha=0.20, label="с DPD")
        axes[1].axhline(0, color="black", lw=1, ls="--", label="идеал")
        axes[1].set(xlabel="|x desired|", ylabel="фазовая ошибка [градусы]",
                    title="AM/PM: остаточная фазовая ошибка")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        ''',
    ),
    code(
        r'''
        # Комплексные сплайн-коэффициенты показываются по ветвям и единицам.
        dense_radius = np.linspace(knots[0], knots[-1], 500)
        dense_left, dense_weight = spline_coordinates(dense_radius, knots)
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
        component_names = ("Re C_m(r)", "Im C_m(r)", "|C_m(r)|", "phase C_m(r) [градусы]")
        for branch_index, (signal_delay, envelope_delay) in enumerate(branches):
            c = dpd_coefficients[branch_index]
            interpolated = c[dense_left] + dense_weight * (c[dense_left + 1] - c[dense_left])
            knot_values = (c.real, c.imag, np.abs(c), np.degrees(np.angle(c)))
            dense_values = (interpolated.real, interpolated.imag, np.abs(interpolated),
                            np.degrees(np.angle(interpolated)))
            for column, (ylabel, dense_value, knot_value) in enumerate(
                zip(component_names, dense_values, knot_values)
            ):
                axis = axes[branch_index, column]
                axis.plot(dense_radius, dense_value, color=f"C{branch_index}")
                axis.plot(knots, knot_value, "o", ms=3.5, color=f"C{branch_index}")
                axis.set_ylabel(ylabel)
                axis.set_title(f"C{branch_index}: x[n-{signal_delay}], |x[n-{envelope_delay}]|")
                if branch_index == len(branches) - 1:
                    axis.set_xlabel("нормированная амплитуда r")
        fig.suptitle("Комплексные коэффициенты ветвей: узлы и линейная интерполяция", y=1.01)
        plt.tight_layout()
        plt.show()
        ''',
    ),
    code(
        r'''
        # Один отсчёт: разложение z[n] на три комплексных вклада.
        example_n = CASCADE_WARMUP + 137
        contributions = []
        example_values = []
        for branch_index, (signal_delay, envelope_delay) in enumerate(branches):
            radius = abs(val_x[example_n - envelope_delay])
            left, weight = spline_coordinates(np.array([radius]), knots)
            c_value = (dpd_coefficients[branch_index, left[0]]
                       + weight[0] * (dpd_coefficients[branch_index, left[0] + 1]
                                      - dpd_coefficients[branch_index, left[0]]))
            contribution = val_x[example_n - signal_delay] * c_value
            contributions.append(contribution)
            example_values.append((radius, c_value))
        z_example = sum(contributions)
        assert abs(z_example - predistorted_val[example_n]) < 1e-12

        print(f"n = {example_n}")
        for index, ((radius, c_value), contribution) in enumerate(zip(example_values, contributions)):
            print(f"u{index}: r={radius:.6f}, C{index}(r)={c_value:.6f}, u{index}={contribution:.6f}")
        print(f"z[n] = u0+u1+u2 = {z_example:.6f}")

        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        origin = 0j
        for index, contribution in enumerate(contributions):
            ax.quiver(origin.real, origin.imag, contribution.real, contribution.imag,
                      angles="xy", scale_units="xy", scale=1, width=0.008,
                      color=f"C{index}", label=f"u{index}")
            origin += contribution
        ax.scatter([z_example.real], [z_example.imag], color="black", s=45, label="z[n]")
        extent = 1.2 * max(abs(z_example), *(abs(value) for value in contributions), 1e-3)
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="вещественная часть",
               ylabel="мнимая часть", title="Сумма ветвей для одного отсчёта")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        plt.tight_layout()
        plt.show()
        ''',
    ),
    code(
        r'''
        # A 600-sample view is wide enough to show a representative fragment
        # without compressing all fast waveform variations into one band.
        fragment_length = 600
        window = slice(CASCADE_WARMUP, CASCADE_WARMUP + fragment_length)
        sample_axis = np.arange(fragment_length) + CASCADE_WARMUP
        fig, axes = plt.subplots(2, 1, figsize=(15.5, 7.8), sharex=True)
        axes[0].plot(sample_axis, ideal_val[window].real, color="black", lw=1.15, label="ideal g·x")
        axes[0].plot(sample_axis, pa_val_hat[window].real, color="tab:red", alpha=0.68, lw=0.75, label="PA без DPD")
        axes[0].plot(sample_axis, cascade_dpd[window].real, color="tab:blue", alpha=0.78, lw=0.75, label="PA с DPD")
        axes[0].set_ylabel("Re{output}")
        axes[0].set_title("Фрагмент выходного сигнала: 600 отсчётов")
        axes[0].legend(loc="upper right", ncols=3)

        error_no_dpd = (pa_val_hat - ideal_val)[window]
        error_with_dpd = (cascade_dpd - ideal_val)[window]

        def local_rms(signal: np.ndarray, width=25) -> np.ndarray:
            """Return a centered moving RMS envelope for plot readability."""
            kernel = np.ones(width, dtype=float) / width
            return np.sqrt(np.convolve(np.abs(signal) ** 2, kernel, mode="same"))

        # Magnitude avoids the visually misleading up/down sign flips of Re{e};
        # faint samples preserve detail and thick curves show the local trend.
        axes[1].plot(sample_axis, np.abs(error_no_dpd), color="tab:red", lw=0.55, alpha=0.18)
        axes[1].plot(sample_axis, np.abs(error_with_dpd), color="tab:blue", lw=0.55, alpha=0.18)
        axes[1].plot(sample_axis, local_rms(error_no_dpd), color="tab:red", lw=1.8, label="локальный RMS без DPD")
        axes[1].plot(sample_axis, local_rms(error_with_dpd), color="tab:blue", lw=1.8, label="локальный RMS с DPD")
        axes[1].set(
            xlabel="номер отсчёта",
            ylabel="|error| / local RMS",
            title="Остаточная комплексная ошибка: мгновенный модуль и локальный RMS",
        )
        axes[1].set_ylim(bottom=0)
        axes[1].legend(loc="upper right", ncols=2)
        plt.tight_layout()
        plt.show()
        ''',
    ),
    markdown(
        r'''
        ## 6. Спектральная диагностика

        Мы считаем Welch-подобную PSD с `fs=1` только для визуального сравнения
        формы спектра. Ось — циклы на отсчёт. Без sample rate и заранее заданных
        RF-регионов нельзя переводить это в dBc/ACLR.

        Показана Welch-оценка после небольшого сглаживания в линейной мощности.
        Чёрный пунктир — спектр реального желаемого сигнала $g x$, а не плоская
        теоретическая маска.

        Сглаживание выполняется **в линейной мощности до перевода в dB**. Оно
        делает рисунок читаемым, но не меняет данные и не используется при
        расчёте NMSE или обучении модели.
        ''',
    ),
    code(
        r'''
        def welch_complex(signal: np.ndarray, nperseg=2048, noverlap=1536):
            """Estimate a normalized-frequency Welch PSD for a complex vector."""
            signal = np.asarray(signal, dtype=np.complex128)
            step = nperseg - noverlap
            window = np.hanning(nperseg)
            scale = np.sum(window ** 2)
            starts = range(0, signal.size - nperseg + 1, step)
            spectra = []
            for start in starts:
                segment = signal[start:start + nperseg]
                spectra.append(np.abs(np.fft.fftshift(np.fft.fft(segment * window))) ** 2 / scale)
            if not spectra:
                raise ValueError("signal is shorter than nperseg")
            power = np.mean(np.stack(spectra), axis=0)
            frequency = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0))
            return frequency, power


        def smooth_linear_power(power: np.ndarray, bins=17) -> np.ndarray:
            """Smooth PSD samples in linear power with an odd Hann kernel."""
            power = np.asarray(power, dtype=float)
            bins = int(bins)
            if bins < 3:
                return power.copy()
            if bins % 2 == 0:
                bins += 1
            kernel = np.hanning(bins)
            kernel /= np.sum(kernel)
            pad = bins // 2
            padded = np.pad(power, (pad, pad), mode="edge")
            return np.convolve(padded, kernel, mode="valid")


        # A shorter segment gives more Welch averages on the 23k-sample
        # validation set; the mild frequency smoothing is presentation-only.
        PSD_NPERSEG = 2048
        PSD_NOVERLAP = 1536
        PSD_SMOOTH_BINS = 17
        psd_signals = {
            "ideal": ideal_val[CASCADE_WARMUP:],
            "no DPD": pa_val_hat[CASCADE_WARMUP:],
            "with DPD": cascade_dpd[CASCADE_WARMUP:],
        }
        psd = {
            name: welch_complex(signal, PSD_NPERSEG, PSD_NOVERLAP)
            for name, signal in psd_signals.items()
        }
        psd_smooth = {
            name: (frequency, smooth_linear_power(power, PSD_SMOOTH_BINS))
            for name, (frequency, power) in psd.items()
        }
        reference_peak = np.max(psd_smooth["ideal"][1])

        colors = {"ideal": "black", "no DPD": "tab:red", "with DPD": "tab:blue"}
        labels = {
            "ideal": "ideal g·x (пунктир)",
            "no DPD": "PA без DPD",
            "with DPD": "PA с DPD",
        }
        fig, ax = plt.subplots(figsize=(12.5, 5.2))

        for name, (frequency, power) in psd_smooth.items():
            smooth_db = 10 * np.log10(np.maximum(power, 1e-30) / reference_peak)
            style = {
                "color": colors[name],
                "linestyle": "--" if name == "ideal" else "-",
                "linewidth": 2.2 if name == "ideal" else 1.8,
            }
            ax.plot(frequency, smooth_db, label=labels[name], **style)

        segment_step = PSD_NPERSEG - PSD_NOVERLAP
        segment_count = 1 + (len(psd_signals["ideal"]) - PSD_NPERSEG) // segment_step
        ax.set(
            xlabel="нормированная частота [циклы/отсчёт]",
            ylabel="PSD [dB относительно пика ideal]",
            title="Нормированная PSD: визуальная диагностика",
        )
        ax.set_ylim(-100, 5)
        ax.legend(loc="best")
        ax.text(
            0.01,
            0.02,
            f"Welch: {segment_count} сегм., N={PSD_NPERSEG}; сглаживание={PSD_SMOOTH_BINS} bins",
            transform=ax.transAxes,
            fontsize=9,
            color="0.35",
        )
        plt.tight_layout()
        plt.show()
        ''',
    ),
    markdown(
        r'''
        **Как читать график.** Цветные линии сравнивают выход PA до и после
        предыскажения с желаемым спектром. Провалы чёрного пунктира принадлежат структуре исходного
        сигнала (например, незаполненным частотным компонентам), поэтому
        идеальная линия здесь не обязана быть горизонтальной.

        Нелинейность PA перераспределяет энергию между частотами и создаёт
        спектральное разрастание. DPD обучается по временной комплексной ошибке
        NMSE, а не по спектральной маске, поэтому улучшение следует оценивать по
        совокупности PSD и NMSE, не по одному локальному провалу кривой.
        Численный ACLR не рассчитан: официальные границы основной и соседних
        полос, а также частота дискретизации для этого capture неизвестны.
        ''',
    ),
    markdown(
        r'''
        ## 7. Вычислительная сложность

        Ниже стоимость вычисляется из фактических `branches` и `K`. Конвенция
        проекта считает обычное комплексное умножение как 4 real MUL + 2 real
        ADD; интерполяция комплексного коэффициента требует ещё 2 real MUL.
        Вычисление модуля, сравнения/поиск интервала, LUT и состояние вынесены
        отдельно. Это аналитическая стоимость Python-equivalent алгоритма, а не
        измеренная задержка FPGA/ASIC и не подтверждённый PASS по ориентиру 1000.
        ''',
    ),
    code(
        r'''
        def spline_fast_path_complexity(branches, knot_count):
            """Count operations under the project's explicit scalar convention."""
            branch_count = len(branches)
            unique_envelope_delays = len({envelope_delay for _, envelope_delay in branches})
            maximum_delay = max(max(pair) for pair in branches)
            return {
                "real_multiplications": 6 * branch_count + 3 * unique_envelope_delays,
                "real_additions": 8 * branch_count - 2 + 2 * unique_envelope_delays,
                "magnitude_operations": unique_envelope_delays,
                "comparisons": int(np.ceil(np.log2(knot_count))) * unique_envelope_delays,
                "LUT_reads": 2 * branch_count,
                "stored_complex_coefficients": branch_count * knot_count,
                "stored_real_coefficients": 2 * branch_count * knot_count,
                "state_complex_values": maximum_delay,
                "state_real_values": 2 * maximum_delay,
            }


        complexity = spline_fast_path_complexity(branches, knot_count)
        print(json.dumps(complexity, ensure_ascii=False, indent=2))
        print("Параметры других моделей здесь не рисуются рядом с MUL/sample: это разные величины.")
        ''',
    ),
    markdown(
        r'''
        ## 8. Fixed-point статус

        Этот BlackBox notebook проверяет только float64-реализацию. Программные
        16/14/12-bit результаты проекта относятся к отдельным DPA/APA
        экспериментам и сюда не переносятся. Для BlackBox ещё требуется
        отдельный bit-accurate запуск с подсчётом насыщений, коллизий узлов,
        спектральной деградации и совпадения потоковой обработки.
        ''',
    ),
    markdown(
        r'''
        ## 9. Справочное сравнение только с OpenDPD

        Ниже считываются bundled machine-readable результаты OpenDPD для
        `APA_200MHz`. Это официальный для данного репозитория reference table:
        MP/GMP используют ILA, GRU и TRes-GRU — DLA через frozen TRes-GRU PA.

        ### Как читать этот блок

        Это **встроенный reference OpenDPD**, а не повторное обучение OpenDPD
        внутри данного ноутбука. Если рядом с notebook нет репозитория OpenDPD,
        используется сохранённая компактная копия этих четырёх строк. Если
        репозиторий найден, числа читаются из его JSON автоматически.

        | Поле | Смысл в OpenDPD |
        |---|---|
        | `NMSE` | repository-specific segment-wise normalized error |
        | `EVM` | repository-specific spectral EVM |
        | `ACLRavg` | среднее левой и правой соседних областей |
        | `parameters` | число параметров/комплексных степеней свободы в bundled benchmark |

        Во всех трёх dB-метриках более отрицательное число означает меньшую
        ошибку или утечку. График ниже показывает только внутреннее ранжирование
        OpenDPD; строка нашего BlackBox эксперимента в эти столбцы не подмешивается.

        Сравнение с BlackBox нельзя подписать как «наш метод лучше/хуже», потому
        что:

        1. BlackBox — другой capture, без sample rate/RF-mask metadata;
        2. OpenDPD — `APA_200MHz`, другой evaluator и другой split;
        3. OpenDPD NMSE — среднее segment-wise dB, а здесь pooled complex NMSE;
        4. OpenDPD EVM/ACLR имеют собственные definitions и недоступны честно для
           BlackBox без частотных границ.

        Поэтому следующий график сравнивает модели **внутри OpenDPD между собой**,
        а наши BlackBox-цифры остаются отдельной строкой контекста.
        ''',
    ),
    code(
        r'''
        OPENDPD_FALLBACK = {
            "source_commit": "3df35e081e6e41463fa46f21778c72a748823274",
            "dataset": "APA_200MHz",
            "models": {
                "MP": {"parameters": 1000, "nmse_db": -42.1896324, "evm_db": -48.1534769, "aclr_avg_db": -45.1859831},
                "GMP": {"parameters": 1000, "nmse_db": -38.5258979, "evm_db": -46.3523044, "aclr_avg_db": -43.5931687},
                "GRU-H16": {"parameters": 994, "nmse_db": -45.13145, "evm_db": -47.4274689, "aclr_avg_db": -51.0114043},
                "TRes-GRU-H15": {"parameters": 999, "nmse_db": -44.285923, "evm_db": -45.0972778, "aclr_avg_db": -53.4879386},
            },
        }


        opendpd_reference = OPENDPD_FALLBACK.copy()
        if OPENDPD_JSON.is_file():
            with OPENDPD_JSON.open(encoding="utf-8") as stream:
                report = json.load(stream)
            apa = report["datasets"]["APA_200MHz"]["dpd_models"]
            name_map = {"mp": "MP", "gmp": "GMP", "gru": "GRU-H16", "tres_gru": "TRes-GRU-H15"}
            opendpd_reference = {
                "source_commit": report["completion_context"]["git_commit"],
                "dataset": "APA_200MHz",
                "models": {
                    name_map[key]: {
                        "parameters": value["model"]["parameters"],
                        "nmse_db": value["metrics"]["test"]["nmse_db"],
                        "evm_db": value["metrics"]["test"]["evm_db"],
                        "aclr_avg_db": value["metrics"]["test"]["aclr_avg_db"],
                    }
                    for key, value in apa.items()
                },
            }
        OPENDPD_ORDER = ["MP", "GMP", "GRU-H16", "TRes-GRU-H15"]
        print("=" * 78)
        print("OpenDPD BUNDLED REFERENCE | APA_200MHz | TEST SPLIT")
        print("source commit:", opendpd_reference["source_commit"])
        print("not retrained by this notebook; metrics follow OpenDPD definitions")
        print("-" * 78)
        print(f"{'model':16s} {'parameters':>10s} {'NMSE [dB]':>12s} {'EVM [dB]':>12s} {'ACLRavg [dB]':>14s}")
        print("-" * 78)
        for name in OPENDPD_ORDER:
            values = opendpd_reference["models"][name]
            print(f"{name:16s} {values['parameters']:10d} {values['nmse_db']:12.3f} {values['evm_db']:12.3f} {values['aclr_avg_db']:14.3f}")
        print("=" * 78)
        ''',
    ),
    code(
        r'''
        names = ["MP", "GMP", "GRU-H16", "TRes-GRU-H15"]
        metrics = ["nmse_db", "evm_db", "aclr_avg_db"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for axis, metric in zip(axes, metrics):
            values = [opendpd_reference["models"][name][metric] for name in names]
            axis.bar(names, values, color=["tab:gray", "tab:purple", "tab:orange", "tab:red"])
            axis.set_title(metric)
            axis.set_ylabel("dB (more negative is better)")
            axis.tick_params(axis="x", rotation=35)
            axis.axhline(0, color="black", lw=0.7)
        fig.suptitle("OpenDPD bundled reference | APA_200MHz test split")
        plt.tight_layout()
        plt.show()
        ''',
    ),
    markdown(
        r'''
        ### Что можно заключить из сравнения

        Внутри OpenDPD TRes-GRU-H15 показывает сильный reference по ACLR, а
        GRU-H16 — лучший NMSE среди этих строк. Напечатанный выше BlackBox
        pooled NMSE нельзя сопоставлять с `−44…−45 dB` OpenDPD как
        единый рейтинг: это разные эксперименты и разные метрики.

        Корректная следующая ступень для apples-to-apples сравнения — экспортировать
        один и тот же BlackBox split в формат OpenDPD, обучить выбранный OpenDPD
        DPD через тот же frozen PA evaluator и заранее зафиксировать общий metric
        protocol. В ноутбуке это намеренно не подменяется неподтверждённой
        цифрой.
        ''',
    ),
    markdown(
        r'''
        ## 10. Встроенные тесты

        Эти проверки не являются формальным unit-test suite проекта, но ловят
        самые опасные ошибки исследовательского ноутбука: перепутанное направление,
        нарушение фазовой эквивариантности, потерю state между блоками и выход за
        обученный диапазон.
        ''',
    ),
    code(
        r'''
        # 1) Данные и модель конечны.
        assert np.all(np.isfinite(train_x)) and np.all(np.isfinite(val_x))
        assert np.all(np.isfinite(pa_coefficients)) and np.all(np.isfinite(dpd_coefficients))
        assert dpd_coefficients.shape == (len(branches), knot_count)

        # 2) Правильное deployment-направление: DPD получает val_x, а не val_y.
        replay_from_desired = predict_spline_memory(val_x, knots, branches, dpd_coefficients)
        assert np.allclose(replay_from_desired, predistorted_val)

        # 3) Фазовая эквивариантность D(e^{jφ}x)=e^{jφ}D(x).
        phi = 0.731
        rotated = np.exp(1j * phi) * val_x[:4000]
        lhs = predict_spline_memory(rotated, knots, branches, dpd_coefficients)
        rhs = np.exp(1j * phi) * predict_spline_memory(val_x[:4000], knots, branches, dpd_coefficients)
        assert np.max(np.abs(lhs - rhs)) < 1e-12

        # 4) Потоковая проверка: разбиваем validation на произвольные chunks.
        def predict_streaming(signal, chunk_sizes):
            """Run one independent record while carrying only causal history."""
            history_size = max(max(pair) for pair in branches)
            history = np.zeros(history_size, dtype=np.complex128)
            output = []
            start = 0
            for size in chunk_sizes:
                chunk = np.asarray(signal[start:start + size], dtype=np.complex128)
                padded = np.concatenate((history, chunk))
                piece = predict_spline_memory(padded, knots, branches, dpd_coefficients)[history.size:]
                output.append(piece)
                history = padded[-history_size:] if history_size else history
                start += size
            if start < signal.size:
                chunk = np.asarray(signal[start:], dtype=np.complex128)
                padded = np.concatenate((history, chunk))
                output.append(predict_spline_memory(padded, knots, branches, dpd_coefficients)[history_size:])
            return np.concatenate(output)


        streamed = predict_streaming(val_x, [1, 8, 64, 257, 1024])
        reference_stream = predict_spline_memory(val_x, knots, branches, dpd_coefficients)
        assert np.max(np.abs(streamed - reference_stream)) < 1e-12

        # 5) Reset на границе независимого record: history train не попадает в validation.
        record_a, record_b = val_x[:3000], val_x[3000:7000]
        separate_b = predict_streaming(record_b, [17, 131, 509])
        repeated_b = predict_streaming(record_b, [1, 29, 777])
        direct_b = predict_spline_memory(record_b, knots, branches, dpd_coefficients)
        assert np.max(np.abs(separate_b - direct_b)) < 1e-12
        assert np.max(np.abs(repeated_b - direct_b)) < 1e-12
        concatenated = predict_spline_memory(np.concatenate((record_a, record_b)), knots, branches, dpd_coefficients)
        # При отсутствии reset первые memory samples второго record обычно отличаются.
        assert np.max(np.abs(concatenated[record_a.size:record_a.size + DPD_WARMUP]
                             - direct_b[:DPD_WARMUP])) > 0

        # 6) Никакого незаявленного extrapolation: validation desired и drive внутри train support.
        assert np.max(np.abs(val_x)) <= knots[-1] * (1 + 1e-12)
        assert np.max(np.abs(predistorted_val)) <= np.max(np.abs(train_x)) * (1 + 1e-12)

        # 7) DPD действительно уменьшает frozen-surrogate error power.
        assert dpd_mse < no_dpd_mse
        print("ALL SELF-TESTS PASSED")
        print(f"phase equivariance max error: {np.max(np.abs(lhs-rhs)):.3e}")
        print(f"streaming max error: {np.max(np.abs(streamed-reference_stream)):.3e}")
        ''',
    ),
    code(
        r'''
        print("КРАТКИЙ ИТОГ BLACKBOX VALIDATION")
        print(f"PA evaluator fidelity: {pooled_nmse_db(pa_val_hat, val_y, PA_WARMUP):.3f} dB NMSE")
        print(f"без DPD: {surrogate_no_dpd['nmse_db']:.3f} dB NMSE")
        print(f"с DPD:   {dpd_metrics['nmse_db']:.3f} dB NMSE")
        print(f"улучшение: {improvement_db:.3f} dB")
        print(f"архитектура: {branches}; K={knot_count}; ridge={ridge:g}")
        print(f"fast path: {complexity['real_multiplications']} MUL, "
              f"{complexity['real_additions']} ADD, "
              f"{complexity['magnitude_operations']} magnitude, "
              f"{complexity['LUT_reads']} LUT reads / complex sample")
        print("fixed-point: не проверен на BlackBox")
        ''',
    ),
    markdown(
        r'''
        ## 11. Что доказано и что не доказано

        ### Подтверждено этим notebook

        - детерминированная загрузка и хронологический train/validation split;
        - train-only alignment, normalization и complex gain;
        - комплексная ridge-калибровка, фазовая эквивариантность и причинность;
        - выбор топологии/K по явно указанному validation-критерию;
        - совпадение цельной и chunked обработки при переносе состояния;
        - корректный reset между независимыми records;
        - аналитическая float fast-path стоимость.

        ### Подтверждено только frozen PA evaluator

        Все cascade NMSE, AM/AM, AM/PM и PSD-графики. Они показывают
        согласованное улучшение внутри программного BlackBox эксперимента, но
        могут быть ограничены ошибкой GMP evaluator.

        ### Не подтверждено

        Физический PA после DPD, официальные Huawei bands/thresholds, стандартная
        EVM, fixed-point именно на BlackBox, hardware timing, переносимость на
        другие captures, online adaptation и apples-to-apples превосходство над
        OpenDPD. Следующий наиболее ценный эксперимент — один waveform и одна
        выходная мощность для `{без DPD / spline DPD / OpenDPD}` на одном
        физическом PA с заранее зафиксированными спектральными областями.
        ''',
    ),
]


def execute_and_capture(notebook_cells: list[dict]) -> dict[str, object]:
    """Execute code cells with stdlib only and retain text/PNG outputs."""
    import contextlib
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None

    namespace: dict[str, object] = {}
    execution_count = 0
    for cell in notebook_cells:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec("".join(cell["source"]), namespace)
        except Exception as exc:
            cell["outputs"] = [{
                "output_type": "error",
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": [],
            }]
            raise
        outputs = []
        if stdout.getvalue():
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": stdout.getvalue().splitlines(keepends=True),
            })
        for fig_number in list(plt.get_fignums()):
            fig = plt.figure(fig_number)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
            outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "text/plain": ["<Figure>"]
                },
                "metadata": {},
            })
            plt.close(fig)
        cell["outputs"] = outputs
    return namespace


def build_embedded_payload() -> str:
    """Pack only the train/validation arrays needed by the portable notebook."""
    selection = Path(__file__).resolve().parents[1] / "data/private/blackbox_v3/selection"

    def read(path: Path) -> np.ndarray:
        raw = np.loadtxt(path, delimiter=",", skiprows=1)
        return np.asarray(raw[:, 0] + 1j * raw[:, 1], dtype=np.complex128)

    arrays = {
        "train_x_raw": read(selection / "train_input.csv"),
        "train_y_raw": read(selection / "train_output.csv"),
        "val_x_raw": read(selection / "val_input.csv"),
        "val_y_raw": read(selection / "val_output.csv"),
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def notebook_document(notebook_cells: list[dict]) -> dict:
    """Return a nbformat-compatible document for a prepared cell list."""
    for index, cell in enumerate(notebook_cells):
        # nbformat 5.1+ requires stable cell ids; deterministic ids keep diffs
        # reproducible without relying on a random UUID.
        cell.setdefault("id", f"dpd{index:05d}")
    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, notebook_cells: list[dict]) -> None:
    """Write one generated notebook, creating only its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(notebook_document(notebook_cells), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"wrote {path} with {len(notebook_cells)} cells")


def main() -> None:
    embedded_payload = build_embedded_payload()
    embedded_cells = copy.deepcopy(cells)
    for cell in embedded_cells:
        if cell["cell_type"] == "code":
            cell["source"] = [
                line.replace("__EMBEDDED_PAYLOAD__", embedded_payload)
                for line in cell["source"]
            ]
            if "EMBEDDED_DATA_B64" in "".join(cell["source"]):
                cell["metadata"]["collapsed"] = True
                cell["metadata"]["tags"] = ["embedded-data"]
    namespace = execute_and_capture(embedded_cells)
    summary_replacements = {
        "__PA_FIDELITY__": f"{namespace['pooled_nmse_db'](namespace['pa_val_hat'], namespace['val_y'], namespace['PA_WARMUP']):.3f}",
        "__BRANCHES__": str(namespace["branches"]),
        "__KNOT_COUNT__": str(namespace["knot_count"]),
        "__NO_DPD_NMSE__": f"{namespace['surrogate_no_dpd']['nmse_db']:.3f}",
        "__DPD_NMSE__": f"{namespace['dpd_metrics']['nmse_db']:.3f}",
        "__IMPROVEMENT__": f"{namespace['improvement_db']:.3f}",
        "__MUL__": str(namespace["complexity"]["real_multiplications"]),
        "__ADD__": str(namespace["complexity"]["real_additions"]),
        "__MAG__": str(namespace["complexity"]["magnitude_operations"]),
        "__LUT__": str(namespace["complexity"]["LUT_reads"]),
    }
    for cell in embedded_cells:
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            for placeholder, value in summary_replacements.items():
                source = source.replace(placeholder, value)
            cell["source"] = source.splitlines(keepends=True)

    output_directory = Path(__file__).resolve().parents[1] / "docs/notebooks"
    # The repository deliberately keeps only two notebooks: the historical
    # reproducible version and this detailed corrected version. The historical
    # file is restored from the preceding Git commit; this generator owns only
    # the detailed file and does not create extra copies.
    write_notebook(output_directory / "Spline_memory_DPD_detailed.ipynb", embedded_cells)


if __name__ == "__main__":
    main()
