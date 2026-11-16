class NeuralNetworkError(Exception):
    '''Базовое исключение для всех ошибок модуля.'''


class InvalidLayerSizeError(NeuralNetworkError):
    '''Некорректный размер слоя или число слоёв.'''


class MismatchedDataError(NeuralNetworkError):
    '''Несовместимость размерностей данных и сети.'''


class DatasetFileError(NeuralNetworkError):
    '''Ошибка чтения или записи файла датасета.'''
