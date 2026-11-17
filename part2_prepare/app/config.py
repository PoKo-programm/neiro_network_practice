import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_data_path(cli_value: str | None = None) -> Path:
    """Путь к входным данным: CLI > ENV > default."""
    raw = cli_value or os.getenv("DATA_PATH", "data/input")
    return Path(raw)


def get_output_path(cli_value: str | None = None) -> Path:
    """Путь к выходному файлу/каталогу: CLI > ENV > default."""
    raw = cli_value or os.getenv("OUTPUT_PATH", "data/output/dataset.npy")
    return Path(raw)


def get_log_level(cli_value: str | None = None) -> str:
    """Уровень логирования: CLI > ENV > INFO."""
    return (cli_value or os.getenv("LOG_LEVEL", "INFO")).upper()


def get_default_extensions() -> set[str]:
    """Расширения по умолчанию из ENV или захардкоженный минимум."""
    raw = os.getenv("DEFAULT_EXTENSIONS", ".jpg,.jpeg,.png")
    return {ext.strip().lower() for ext in raw.split(",") if ext.strip()}
