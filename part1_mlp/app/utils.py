import numpy as np
from .exceptions import MismatchedDataError


def parse_int_list(raw):
    try:
        values = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise MismatchedDataError(f"Ожидались целые числа: {exc}")
    if not values:
        raise MismatchedDataError("Список размеров пуст")
    return values


def parse_float(raw, name="значение"):
    try:
        return float(raw)
    except ValueError as exc:
        raise MismatchedDataError(f"{name}: ожидалось число ({exc})")


def parse_vector(raw):
    tokens = raw.replace(",", " ").split()
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise MismatchedDataError(f"Некорректный вектор: {exc}")
    return np.array(values, dtype=float).reshape(1, -1)
