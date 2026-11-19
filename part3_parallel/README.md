# Часть 3. Многопоточная и многопроцессорная аугментация

Расширение части 2: потоки читают файлы, процессы применяют аугментацию. Сравнивается время последовательной и параллельной версий.

## Состав

- app/producer_consumer.py — producer, потоки-читатели, пул процессов, точка входа
- app/augment.py — функции аугментации (отражение, поворот, шум)
- app/performance.py — замеры времени и запись performance_report.json

## Архитектура

- Producer (один поток) сканирует каталог и кладёт пути к файлам в первую очередь queue.Queue
- Readers (несколько потоков) читают файлы и кладут массивы во вторую очередь queue.Queue
- Pool процессов извлекает массивы и применяет аугментацию
- ProgressCounter с threading.Lock защищает общий счётчик обработанных файлов
- Завершение через poison pill (None) в обеих очередях, затем pool.close(), pool.join(), thread.join()

## Запуск

python -m part3_parallel.app.producer_consumer --path data/input/images --ext .jpg,.png --output data/output/performance_report.json --workers 4 --readers 4 --log-level DEBUG

Параметры:

- --path: дефолт data/input/images, каталог с изображениями
- --ext: дефолт .jpg,.jpeg,.png, расширения через запятую
- --output: дефолт data/output/performance_report.json, путь к JSON-отчёту
- --workers: дефолт cpu_count(), число процессов
- --readers: дефолт 4, число потоков-читателей
- --log-level: дефолт INFO, уровень логирования

## Результат

Файл performance_report.json содержит поля sequential_sec, parallel_sec, speedup, files_processed, process_workers, cpu_count. Пример:

- sequential_sec: 0.412
- parallel_sec: 0.187
- speedup: 2.2
- files_processed: 100
- process_workers: 8
- cpu_count: 8

## Почему процессы для аугментации

Аугментация — CPU-bound задача. В CPython GIL не позволяет потокам параллельно исполнять байт-код, поэтому для вычислений нужны процессы. Потоки эффективны только на I/O-bound чтении файлов, где GIL освобождается во время системных вызовов.