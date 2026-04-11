# Эксперименты

В каталоге `results/` лежат 88 детерминированных запусков классического baseline для
полярности и восьми мультилейбл-эмоций. Каждый JSON получен командой runner и содержит
конфигурацию, размеры split, метрики модели и prior baseline, дельты, пороги,
per-emotion diagnostics и, где применимо, stress diagnostics.

## Покрытие

| Серия | Запусков | Что меняется |
|---|---:|---|
| `ablation` | 22 | `char`, `word` и `char_word`, несколько split seed |
| `unicode` | 22 | доля NFKC-эквивалентных full-width символов от 5% до 100% |
| `punct` | 22 | удаление и повтор знаков, эмодзи и смешанное искажение |
| `threshold` | 22 | 19 фиксированных порогов и 3 запуска с настройкой по validation |

Все запуски используют встроенный smoke-набор из 48 строк. При каждом seed он делится
на 28 train, 10 validation и 10 test строк. Поэтому результаты проверяют код,
устойчивость и направление сравнений, но не оценивают качество на реальном корпусе.

## Наблюдения

Средние test-метрики ablation-серии:

| Признаки | Запусков | Признаков, среднее | Polarity macro F1 | Emotion micro F1 | Emotion macro F1 |
|---|---:|---:|---:|---:|---:|
| `char` | 8 | 591.0 | 0.9611 | 0.9792 | 0.8483 |
| `word` | 7 | 43.9 | 0.9705 | 0.9399 | 0.7884 |
| `char_word` | 7 | 633.9 | 0.9722 | 0.9762 | 0.8266 |

На этих split объединённые признаки дали лучший средний polarity macro F1, а
символьные — лучший средний emotion micro и macro F1. Разница невелика и требует
проверки на независимом корпусе.

Во всех 22 Unicode-запусках NFKC вернула каждую из 220 test-строк к исходному тексту;
raw и normalized метрики совпали. В punctuation-серии даже 100% искажение при seed 42
сохранило polarity macro F1 = 1.0 и emotion micro F1 = 1.0. Это показывает отсутствие
регрессии на smoke-примерах, а не универсальную устойчивость модели.

В sweep фиксированных порогов test emotion micro F1 менялся от 0.0 до 1.0. Плато
0.35–0.55 дало micro F1 = 1.0; при 0.60–0.65 значение снизилось до 0.9655, а при
0.80–0.95 — до 0.0. Индивидуальная настройка дала micro F1 = 1.0 на seed 42, 43 и 44;
macro F1 составил 0.75, 1.0 и 1.0 соответственно. В последнем запуске каждый класс
получил F1 = 1.0 при test support от 1 до 3; prior baseline имел emotion micro F1 = 0.0
и polarity macro F1 = 0.1538.

## Запуск

Пример обычного сравнения:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
  .venv/bin/python -m japanese_emotion_sentiment.experiment \
  --output experiments/results/example.json \
  --feature-mode char_word --seed 42
```

Пример punctuation stress и фиксированного порога:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
  .venv/bin/python -m japanese_emotion_sentiment.experiment \
  --output experiments/results/example_stress.json \
  --stress-mode punctuation --punctuation-style mixed --stress-fraction 1.0 \
  --threshold-mode fixed --fixed-threshold 0.45 --seed 42
```

Runner записывает JSON с сортированными ключами и завершающим переводом строки.
