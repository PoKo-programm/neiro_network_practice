import argparse
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
from .config import (
    get_data_path,
    get_default_extensions,
    get_log_level,
    get_output_path,
)
from .exceptions import DataPreparationError
from .logging_config import setup_logging
from .scanner import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    """Собирает парсер с двумя подкомандами."""
    parser = argparse.ArgumentParser(
        prog="dataprep",
        description="Подготовка данных и проверка окружения.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Собрать датасет из файлов")
    prepare.add_argument("--path", help="Каталог с исходными файлами")
    prepare.add_argument(
        "--ext", help="Расширения через запятую, напр. .jpg,.png"
    )
    prepare.add_argument("--output", help="Путь к выходному .npy")
    prepare.add_argument("--log-level", help="DEBUG/INFO/WARNING/ERROR")

    doctor = sub.add_parser("doctor", help="Проверить окружение")
    doctor.add_argument("--log-level", help="DEBUG/INFO/WARNING/ERROR")
    return parser


def parse_extensions(raw: str | None) -> set[str]:
    """Разбирает строку расширений в множество с точкой в начале."""
    if not raw:
        return get_default_extensions()
    extensions = set()
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        extensions.add(part if part.startswith(".") else f".{part}")
    if not extensions:
        raise DataPreparationError("Пустой список расширений")
    return extensions


def run_prepare(args) -> int:
    """Команда prepare: обход, нормализация, сохранение в .npy."""
    logger = setup_logging(get_log_level(args.log_level))
    root = get_data_path(args.path)
    output = get_output_path(args.output)
    extensions = parse_extensions(args.ext)

    logger.info("Входной каталог: %s", root)
    logger.info("Расширения: %s", sorted(extensions))
    logger.info("Выходной файл: %s", output)

    dataset = prepare_dataset(root, extensions, logger)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, dataset)
    logger.info("Сохранено в %s", output)
    return 0


def run_doctor(args) -> int:
    """Команда doctor: версия Python, библиотеки, CUDA, cwd."""
    logger = setup_logging(get_log_level(args.log_level))

    logger.info("Python: %s", sys.version.split()[0])
    logger.info("Платформа: %s", platform.platform())
    logger.info("Рабочая директория: %s", Path.cwd())

    for name in ("numpy", "PIL", "dotenv"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "unknown")
            logger.info("%s: %s", name, version)
        except ImportError:
            logger.warning("%s: НЕ УСТАНОВЛЕН", name)

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            first_line = result.stdout.splitlines()[0]
            logger.info("CUDA: доступна (%s)", first_line)
        else:
            logger.warning("CUDA: nvidia-smi вернул код %s", result.returncode)
    except FileNotFoundError:
        logger.warning("CUDA: nvidia-smi не найден")
    except subprocess.TimeoutExpired:
        logger.warning("CUDA: nvidia-smi не ответил за 5 секунд")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            return run_prepare(args)
        if args.command == "doctor":
            return run_doctor(args)
    except DataPreparationError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
