# Часть 2. Подготовка данных и CLI-утилита препроцессинга

CLI-утилита с двумя командами: prepare (сборка датасета из файлов) и doctor (проверка окружения).

## Состав

app/cli.py — argparse, команды prepare и doctor.
app/config.py — чтение настроек из ENV/CLI/дефолтов.
app/scanner.py — рекурсивный обход каталога, чтение изображений и CSV, нормализация, сборка в один массив.
app/logging_config.py — настройка логгера.
app/exceptions.py — DataPreparationError, DatasetFileError.

## Запуск

Собрать датасет из CSV:

python -m part2_prepare.app.cli prepare --path data/input --ext .csv --output data/output/dataset.npy

Собрать датасет из изображений:

python -m part2_prepare.app.cli prepare --path data/input/images --ext .jpg,.png --output data/output/images.npy

Проверить окружение:

python -m part2_prepare.app.cli doctor

## Конфигурация

Приоритет: CLI → ENV → дефолт.

Путь к данным: CLI --path, ENV DATA_PATH, дефолт data/input.
Выходной файл: CLI --output, ENV OUTPUT_PATH, дефолт data/output/dataset.npy.
Уровень логов: CLI --log-level, ENV LOG_LEVEL, дефолт INFO.
Расширения: CLI --ext, ENV DEFAULT_EXTENSIONS, дефолт .jpg,.jpeg,.png.

Скопируйте .env.example в .env, чтобы задать значения локально.

## Что делает prepare

1. Рекурсивно обходит каталог через Path.rglob("*").
2. Отбирает файлы с указанными расширениями (регистр не важен).
3. Читает изображения через PIL, приводит к RGB, делит на 255.
4. Читает CSV как числовые таблицы через np.genfromtxt.
5. Склеивает всё в один массив через np.stack.
6. Сохраняет в .npy и логирует количество, форму, размер, время.

## Что делает doctor

Версия Python и платформа.
Текущая рабочая директория.
Наличие библиотек numpy, PIL, dotenv и их версии.
Доступность CUDA через subprocess.run(["nvidia-smi"]).