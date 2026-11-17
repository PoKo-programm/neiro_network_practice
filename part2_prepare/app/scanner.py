import time
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from .exceptions import DatasetFileError


def collect_files(root: Path, extensions: set[str]) -> list[Path]:
    """Рекурсивно собирает файлы с указанными расширениями."""
    if not root.exists():
        raise DatasetFileError(f"Каталог не найден: {root}")
    if not root.is_dir():
        raise DatasetFileError(f"Это не каталог: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def load_image(path: Path) -> np.ndarray:
    """Читает изображение, приводит к RGB и нормализует в [0, 1]."""
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
    except (UnidentifiedImageError, OSError) as exc:
        raise DatasetFileError(f"Не удалось прочитать {path}: {exc}")
    return array / 255.0


def load_csv_table(path: Path) -> np.ndarray:
    """Читает CSV как числовую таблицу (без заголовков)."""
    try:
        table = np.genfromtxt(path, delimiter=",", skip_header=1)
    except (OSError, ValueError) as exc:
        raise DatasetFileError(f"Не удалось прочитать {path}: {exc}")
    if table.ndim != 2:
        raise DatasetFileError(f"CSV не двумерный: {path}")
    return table.astype(np.float32)


def prepare_dataset(root: Path, extensions: set[str], logger) -> np.ndarray:
    """Обходит каталог, загружает файлы и склеивает их в один массив."""
    started = time.perf_counter()
    files = collect_files(root, extensions)
    if not files:
        raise DatasetFileError(
            f"В {root} не найдено файлов с расширениями {extensions}"
        )

    arrays = []
    skipped = 0
    for path in files:
        try:
            if path.suffix.lower() == ".csv":
                arrays.append(load_csv_table(path))
            else:
                arrays.append(load_image(path))
        except DatasetFileError as exc:
            logger.warning("Пропущен %s: %s", path, exc)
            skipped += 1

    if not arrays:
        raise DatasetFileError("Ни один файл не удалось загрузить")

    shapes = {array.shape for array in arrays}
    if len(shapes) > 1:
        logger.warning("Найдены разные формы массивов: %s", shapes)
        min_shape = tuple(min(s[i] for s in shapes) for i in range(len(next(iter(shapes)))))
        arrays = [
            array[tuple(slice(0, dim) for dim in min_shape)]
            for array in arrays
        ]

    stacked = np.stack(arrays)
    elapsed = time.perf_counter() - started
    logger.info(
        "Обработано файлов: %d (пропущено %d), форма: %s, "
        "размер: %.2f МБ, время: %.2f с",
        len(arrays), skipped, stacked.shape,
        stacked.nbytes / (1024 * 1024), elapsed,
    )
    return stacked
