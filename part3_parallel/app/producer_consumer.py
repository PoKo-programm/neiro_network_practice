import argparse
import logging
import multiprocessing as mp
import queue
import sys
import threading
from pathlib import Path
from .exceptions import DatasetFileError
from part2_prepare.app.logging_config import setup_logging
from part2_prepare.app.scanner import collect_files, load_image
from .augment import augment_pair
from .performance import save_report, timed


class ProgressCounter:
    """Потокобезопасный счётчик обработанных файлов."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = 0

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


def producer(
    root: Path,
    extensions: set[str],
    file_queue: queue.Queue,
    readers_count: int,
    logger: logging.Logger,
) -> None:
    """Сканирует каталог и кладёт пути в очередь. Кладёт poison pill."""
    try:
        files = collect_files(root, extensions)
    except DatasetFileError as exc:
        logger.error("Ошибка сканирования: %s", exc)
        files = []
    logger.info("Найдено файлов: %d", len(files))
    for path in files:
        file_queue.put(path)
    for _ in range(readers_count):
        file_queue.put(None)  # poison pill — сигнал «файлов больше не будет»


def reader(
    file_queue: queue.Queue,
    array_queue: queue.Queue,
    counter: ProgressCounter,
    logger: logging.Logger,
) -> None:
    """Читает файлы из очереди путей и кладёт массивы в очередь массивов."""
    while True:
        path = file_queue.get()
        if path is None:
            array_queue.put(None)  # передаём poison pill дальше
            return
        try:
            array = load_image(path)
        except DatasetFileError as exc:
            logger.warning("Пропущен %s: %s", path, exc)
            continue
        processed = counter.increment()
        logger.debug("Поток прочитал %s (#%d)", path.name, processed)
        array_queue.put(array)


def sequential_augment(files: list[Path], logger: logging.Logger) -> int:
    """Последовательная версия: читает и аугментирует в одном потоке."""
    count = 0
    for path in files:
        try:
            image = load_image(path)
        except DatasetFileError as exc:
            logger.warning("Пропущен %s: %s", path, exc)
            continue
        augment_pair((image, count))
        count += 1
    return count


def parallel_augment(
    root: Path,
    extensions: set[str],
    workers: int,
    readers_count: int,
    logger: logging.Logger,
) -> int:
    """Параллельная версия: потоки читают, процессы аугментируют."""
    file_queue: queue.Queue = queue.Queue(maxsize=100)
    array_queue: queue.Queue = queue.Queue(maxsize=100)
    counter = ProgressCounter()

    producer_thread = threading.Thread(
        target=producer,
        args=(root, extensions, file_queue, readers_count, logger),
        name="producer",
    )
    reader_threads = [
        threading.Thread(
            target=reader,
            args=(file_queue, array_queue, counter, logger),
            name=f"reader-{i}",
        )
        for i in range(readers_count)
    ]

    producer_thread.start()
    for thread in reader_threads:
        thread.start()

    processed = 0
    with mp.Pool(processes=workers) as pool:
        # Лениво забираем массивы из очереди и отдаём в пул.
        while True:
            item = array_queue.get()
            if item is None:
                break
            pool.apply_async(
                augment_pair,
                args=((item, processed),),
                callback=lambda _: None,
            )
            processed += 1
        pool.close()
        pool.join()

    producer_thread.join()
    for thread in reader_threads:
        thread.join()

    logger.info(
        "Параллельно обработано: %d (счётчик потоков: %d)",
        processed, counter.value,
    )
    return processed


def build_parser() -> argparse.ArgumentParser:
    """Парсер аргументов для части 3."""
    parser = argparse.ArgumentParser(
        prog="parallel_augment",
        description="Сравнение последовательной и параллельной аугментации.",
    )
    parser.add_argument("--path", help="Каталог с изображениями")
    parser.add_argument("--ext", default=".jpg,.jpeg,.png",
                        help="Расширения через запятую")
    parser.add_argument("--output", default="performance_report.json",
                        help="Путь к JSON-отчёту")
    parser.add_argument("--workers", type=int, default=None,
                        help="Число процессов (по умолчанию cpu_count)")
    parser.add_argument("--readers", type=int, default=4,
                        help="Число потоков-читателей")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа: последовательный и параллельный прогоны + отчёт."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging(args.log_level)

    root = Path(args.path or "data/input/images")
    extensions = {
        ext.strip().lower() if ext.startswith(".")
        else f".{ext.strip().lower()}"
        for ext in args.ext.split(",") if ext.strip()
    }
    workers = args.workers or mp.cpu_count()
    output = Path(args.output)

    try:
        files = collect_files(root, extensions)
    except DatasetFileError as exc:
        logger.error("Ошибка: %s", exc)
        return 1
    if not files:
        logger.error("В %s нет файлов с расширениями %s", root, extensions)
        return 1

    logger.info("Файлов: %d, процессов: %d, потоков: %d",
                len(files), workers, args.readers)

    _, seq_time = timed(sequential_augment, files, logger)
    logger.info("Последовательная версия: %.3f с", seq_time)

    _, par_time = timed(
        parallel_augment, root, extensions, workers, args.readers, logger,
    )
    logger.info("Параллельная версия: %.3f с", par_time)

    report = save_report(output, seq_time, par_time, len(files), workers)
    logger.info("Отчёт сохранён в %s: %s", output, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
