# Практическое задание по дисциплине «Основы нейронных сетей»

Репозиторий содержит три части практического задания: ООП-реализацию многослойного перцептрона, CLI-утилиту подготовки данных и многопоточную/многопроцессорную аугментацию.

## Структура

- common/ — общие исключения для всех частей
- part1_mlp/ — ООП-нейросеть и консольное меню
- part2_prepare/ — CLI prepare и doctor
- part3_parallel/ — producer-consumer и multiprocessing
- data/input/ — исходные данные (CSV, изображения)
- data/output/ — артефакты: .npy, веса, performance_report.json
- logs/ — логи
- requirements.txt — зависимости
- .env.example — образец конфигурации
- .gitignore — исключения Git

## Установка

- Создайте виртуальное окружение: python -m venv .venv
- Активируйте: source .venv/bin/activate (Linux/macOS) или .venv\Scripts\activate (Windows PowerShell)
- Установите зависимости: python -m pip install -r requirements.txt

## Запуск

Часть 1. ООП-нейросеть: python -m part1_mlp.app.main

Часть 2. Подготовка данных: python -m part2_prepare.app.cli prepare --path data/input --ext .csv --output data/output/dataset.npy

Проверка окружения: python -m part2_prepare.app.cli doctor

Часть 3. Параллельная аугментация: python -m part3_parallel.app.producer_consumer --path data/input/images --ext .jpg,.png --output data/output/performance_report.json --log-level DEBUG

## Конфигурация

Настройки читаются по приоритету CLI → переменные окружения → значения по умолчанию. Скопируйте .env.example в .env и при необходимости измените пути и уровень логирования. Файл .env в репозиторий не попадает (см. .gitignore).

## Требования

- Python 3.10+ (используется синтаксис str | None)
- Зависимости: numpy, pillow, python-dotenv, matplotlib, flake8

## Проверка стиля

python -m flake8 part1_mlp part2_prepare part3_parallel common --max-line-length=88

## Данные

Для части 1 используется датасет Social Network Ads Purchase Prediction (data/input/Social_Network_Ads.csv). Для частей 2 и 3 — CSV и изображения в data/input/. Артефакты (.npy, веса, JSON-отчёт) создаются в data/output/ и не коммитятся.

## Лицензия

Учебный проект, распространяется свободно для образовательных целей.