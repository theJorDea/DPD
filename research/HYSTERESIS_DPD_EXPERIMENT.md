# Branch-specific hysteresis residual DPD: результат BlackBox ablation

## Гипотеза

Проверена вложенная модель

\[
z[n]=\sum_{m=0}^{2}x[n-m]\left[C_m(r_m[n])+g_m[n]H_m(r_m[n])\right],
\]

где

\[
r_m[n]=|x[n-m]|,
\quad
g_m[n]\in\{-1,0,+1\}
\]

задаётся знаком изменения той же задержанной огибающей с deadband. При
`H=0` модель точно сводится к Spline Memory DPD.

## Проверенный поиск

- baseline: 3 ветви, `K=24`, ridge `1e-4`;
- `alpha={0,0.1,0.25,0.5,1.0}`;
- `deadband=alpha*median(|delta r|)`;
- residual ridge `{1e-3,1e-2,1e-1,1,10}`;
- всего 25 residual-конфигураций;
- одинаковые BlackBox train/validation, normalization, gain и causal GMP PA;
- validation использовалась для выбора alpha/ridge;
- physical PA и sealed test не использовались.

## Результат

| Модель | Validation cascade NMSE |
|---|---:|
| No DPD | -16.00477 dB |
| Spline Memory DPD | **-17.19093 dB** |
| Лучший hysteresis residual | -17.19052 dB |

Изменение лучшего residual относительно baseline: `-0.00042 dB`, то есть
кандидат немного хуже и практически совпадает с baseline.

Лучший residual-кандидат:

- `alpha=0.5`;
- residual ridge `10`;
- `||H||/||C||=1.59e-5`;
- residual практически полностью прижат к нулю.

При слабом residual ridge модель ухудшается:

- ridge `1e-3`: примерно `1.9–2.0 dB` хуже baseline;
- ridge `1e-2`: примерно `0.57–0.62 dB` хуже;
- ridge `1e-1`: примерно `0.058–0.062 dB` хуже.

Это характерно для дополнительной variance/ILA gate mismatch, а не для
устойчивого полезного hysteresis-эффекта.

## Residual analysis

Максимальная разница условных средних комплексных ошибок attack/decay в восьми
амплитудных интервалах:

- ветвь `(0,0)`: `0.00286`;
- ветвь `(1,1)`: `0.00249`;
- ветвь `(2,2)`: `0.00443`.

Разница присутствует, но raw residual correlation не доказывает, что изменение
DPD улучшит выход PA. Полный cascade experiment показал отсутствие выигрыша.

## Стоимость лучшего кандидата

При offline формировании банков `C+H`, `C`, `C-H`:

- 27 real MUL/sample;
- 31 real ADD/sample;
- 3 amplitude operations;
- 18 comparisons;
- 6 LUT reads;
- 288 stored real coefficients;
- 6 state real values.

MUL почти не растёт, но память коэффициентов удваивается. Поскольку качество не
улучшилось, fixed-point sweep этого кандидата не запускается: это не добавит
информации к решению о выборе модели.

## Вывод

На текущем BlackBox capture гипотеза **не прошла acceptance gate**. Модель
реализована и протестирована, но не заменяет Spline Memory DPD baseline.

Вероятные причины:

1. слабый полезный hysteresis-компонент именно в этом capture;
2. несовпадение gate при ILA (`u=y/g`) и deployment (`x`);
3. дополнительная коррелированность residual-признаков;
4. найденное различие attack/decay не учитывает PA sensitivity.

Следующий наиболее информативный шаг — не расширять этот gate, а проверить
PA-sensitivity-aware residual selection или выполнить direct/shadow calibration
на независимом evaluator/физическом PA.

Машиночитаемый результат: `experiments/results/blackbox_hysteresis_v1.json`.
