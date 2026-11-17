class DataPreparationError(Exception):
    '''Базовое исключение модуля подготовки данных.'''


class DatasetFileError(DataPreparationError):
    '''Ошибка чтения файла или каталога.'''


class ConfigError(DataPreparationError):
    '''Некорректная конфигурация (пути, уровень логирования).'''
