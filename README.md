# Классификация политических текстов

Проект Дмитрия: 20 Newsgroups, категории guns / mideast / misc. Результаты экспериментов и разбор ошибок — в [results/analysis.md](results/analysis.md).

## Запуск в PowerShell

Один раз, если окружения ещё нет:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Эксперименты и финальное обучение:

```powershell
.\.venv\Scripts\python.exe -X utf8 -u train.py
.\.venv\Scripts\python.exe -X utf8 -u train.py --final
```

Прогноз готовой моделью:

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --predict "The debate concerns gun ownership and firearm regulation."
```

Сравнение способов сжатия существующей модели:

```powershell
.\.venv\Scripts\python.exe -X utf8 -u train.py --optimize
```

Сжатие не меняет параметры и ответы модели. Режим --final тоже сохраняет сжатый файл. В Zed есть задачи запуска Python с автоматическим сохранением исходников.

## Файлы

- main.py — загрузка модели и ввод текста для прогноза.
- train.py — обучение, сравнение вариантов и оценка.
- data.py — загрузка и очистка.
- models.py — конфигурации экспериментов.
- storage.py — проверенное сохранение и сжатие.
- artifacts/model.joblib — готовая модель.
- results/experiments.csv — результаты шести вариантов.
- results/compression.json — замеры размера и проверки без потерь.
- results/<вариант>/ — метрики, матрицы ошибок и ошибочные тексты.
- results/analysis.md — первоначальный анализ качества.


Лучший validation macro-F1: 0.8667 (ComplementNB). Официальный test macro-F1: 0.7503. Тест не использовался для выбора параметров. Подробности и разбор ошибок — в results/analysis.md.

Старые пользовательские исходники и резервные копии удалены. .venv и бинарная модель не включаются в Git; артефакт воспроизводится командой --final. 

