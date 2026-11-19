import json
import time
from pathlib import Path


def timed(func, *args, **kwargs):
    """Выполняет func и возвращает (результат, время в секундах)."""
    started = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - started
    return result, elapsed


def save_report(
    output_path: Path,
    sequential_sec: float,
    parallel_sec: float,
    files_processed: int,
    workers: int,
) -> dict:
    """Сохраняет сравнение последовательной и параллельной версий."""
    report = {
        "sequential_sec": round(sequential_sec, 3),
        "parallel_sec": round(parallel_sec, 3),
        "speedup": round(sequential_sec / parallel_sec, 2)
        if parallel_sec > 0 else None,
        "files_processed": files_processed,
        "process_workers": workers,
        "cpu_count": __import__("multiprocessing").cpu_count(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)
    return report
