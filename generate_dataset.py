import random
import string
import time
import multiprocessing as mp

from datetime import datetime
from typing import List, Union

import pandas as pd
import numpy as np


def random_string(chars: str, lengths: np.ndarray):
    """
    Функция для генерации случайных строк с буквами, цифрами и знаками препинания
    :param chars: строка со всеми символами для генерации
    :param lengths: список случайных длин строк
    :return: строка из рандомного кол-ва (lengths) символов из строки chars
    """
    return ''.join(random.choice(chars) for _ in range(random.choice(lengths)))


def generate_chunk_parallel(
        chunk_size: int, chunk_num: int,
        chunk_end_dates: List[Union[str, datetime]],
        chars: str, lengths: np.ndarray,
        missing_ratio: float, duplicate_ratio: float
):
    """
    Функция для распараллеленной генерации части датасета
    :param chunk_size: размер чанка
    :param chunk_num: номер чанка
    :param chunk_end_dates: список дат, с которых генерируется каждый чанк
    :param chars: строка со всеми символами для генерации
    :param lengths: список случайных длин строк
    :param missing_ratio: процентная доля nan значений
    :param duplicate_ratio: процентная доля дубликатов
    :return:
    """
    print(f"Начало Chunk {chunk_num + 1}")
    chunk_start_time = time.time()

    random_exp = np.random.exponential(scale=1.0, size=chunk_size)

    # Генерация исходных данных
    data = {
        'datetime': pd.date_range(end=chunk_end_dates[chunk_num], periods=chunk_size, freq='T'),
        # 'numeric': np.random.randint(1, 10380, size=chunk_size),
        'numeric': np.clip((random_exp * 25).astype(int), 1, 200),
        'string': [random_string(chars, lengths) for _ in range(chunk_size)]
    }
    df_chunk = pd.DataFrame(data)

    num_missing = int(len(df_chunk) * missing_ratio)
    # Выбор строк с nan значениями
    missing_indices_numeric = np.random.choice(df_chunk.index, num_missing, replace=False)
    missing_indices_string = np.random.choice(df_chunk.index, num_missing, replace=False)
    # Добавление nan
    df_chunk.loc[missing_indices_numeric, 'numeric'] = None
    df_chunk.loc[missing_indices_string, 'string'] = None

    # Добавление дублей
    num_duplicates = int(len(df_chunk) * duplicate_ratio)
    # Выборка строк для дублей и соединение
    duplicates = df_chunk.sample(num_duplicates)
    df_chunk = pd.concat([df_chunk, duplicates], ignore_index=True)
    # Перемешивание датасета
    df_chunk = df_chunk.sample(frac=1).reset_index(drop=True)

    print(f"Chunk {chunk_num + 1} завершён. Время: {time.time() - chunk_start_time:.2f} секунд.")
    return df_chunk


def generate_dataset_parallel(
        file_path: str, total_rows: int,
        chunk_size: int, n_cores: int,
        chars: str, lengths: np.ndarray,
        missing_ratio: float, duplicate_ratio: float
):
    """
    Основная функция для генерации датасета
    :param file_path: путь считывания файла
    :param total_rows: кол-во строк в генерируемом датасете
    :param chunk_size: размер одного чанка
    :param n_cores: кол-во потоков распараллеливания
    :param chars: строка со всеми символами для генерации
    :param lengths: список случайных длин строк
    :param missing_ratio: процентная доля nan значений
    :param duplicate_ratio: процентная доля дубликатов
    :return:
    """
    start_time = time.time()

    # Определяем количество чанков
    n_chunks = total_rows // chunk_size

    # Начальная дата для первого чанка
    end_date = pd.Timestamp(datetime.today())
    # Определяем даты для каждого чанка, которые будут переданы в параллельные процессы
    chunk_end_dates = [end_date - pd.DateOffset(minutes=chunk_size * i) for i in range(n_chunks)]

    with mp.Pool(n_cores) as pool:
        # Распараллеливание
        results = [
            pool.apply_async(generate_chunk_parallel, args=(
                chunk_size, chunk_num, chunk_end_dates,
                chars, lengths, missing_ratio, duplicate_ratio
            )) for chunk_num in range(n_chunks)
        ]
        chunks = [r.get() for r in results]

    # Объединяем в один DataFrame
    df_final = pd.concat(chunks, ignore_index=True)
    print(f"Объединение завершено. Время: {time.time() - start_time:.2f} секунд.")

    df_final.to_csv(file_path + '.csv', index=False, chunksize=chunk_size)
    # df_final.to_parquet(file_path + '.parquet', engine='pyarrow')
    print(f"Генерация завершена. Время: {time.time() - start_time:.2f} секунд.")


if __name__ == '__main__':
    total_rows = 100_000_000
    n_cores = mp.cpu_count()
    chunk_size = 6_250_000
    # chunk_size = total_rows // n_cores

    # Символы и длины для генерации строк
    chars = string.ascii_letters + string.digits + ' ,.?!'
    lengths = np.arange(5, 16)

    missing_ratio = 0.005
    duplicate_ratio = 0.1

    file_path = 'generated_dataset'
    generate_dataset_parallel(
        file_path, total_rows,
        chunk_size, n_cores,
        chars, lengths,
        missing_ratio, duplicate_ratio
    )
