import time
import multiprocessing as mp

import pandas as pd


def clean_data(df_chunk):
    """
    Функция для процессинга данных
    :param df_chunk: датафрейм одного чанка
    :return:
    """
    chunk_start_time = time.time()

    # Удаление пустых строк
    df_chunk = df_chunk.dropna(how='any')

    # Удаление дублей внутри chunk
    df_chunk = df_chunk.drop_duplicates()

    # Преобразование строк, не содержащих цифр, в пустые строки
    df_chunk['string'] = df_chunk['string'].apply(
        lambda x: None if not any(char.isdigit() for char in x) else x
    )

    # Удаление записей с 1 до 3 часов ночи
    df_chunk['datetime'] = pd.to_datetime(df_chunk['datetime'])
    df_chunk = df_chunk[(df_chunk['datetime'].dt.hour <= 1) | (df_chunk['datetime'].dt.hour >= 3)].copy()

    # Удаление пустых строк 2
    df_chunk = df_chunk.dropna(how='any')

    print(f"Chunk обработан. Время: {time.time() - chunk_start_time:.2f} секунд.")
    return df_chunk


def process_dataset_in_chunks(input_file, chunk_size, n_cores, output_file):
    """
    Считывание данных по частям для распараллеливания
    :param input_file: путь к исходному файлу
    :param chunk_size: размер одного чанка
    :param n_cores: кол-во потоков распараллеливания
    :param output_file: путь к результирующему файлу
    :return:
    """
    start_time = time.time()

    # Пул процессов
    with mp.Pool(n_cores) as pool:
        results = [
            pool.apply_async(clean_data, args=(chunk,))
            for chunk in pd.read_csv(input_file, chunksize=chunk_size)
        ]

        # Сбор результатов
        proccesed_chunks = [r.get() for r in results]

    # Объединение в один датасет
    proccesed_df = pd.concat(proccesed_chunks)
    print(f"Объединение завершено. Время: {time.time() - start_time:.2f} секунд.")

    # Удаление дублей
    proccesed_df = proccesed_df.drop_duplicates()
    print(f"Удаление дублей завершено. Время: {time.time() - start_time:.2f} секунд.")

    # Сохранение в csv
    proccesed_df.to_csv(output_file, index=False, chunksize=chunk_size)
    print(f"Сохранение завершено. Время: {time.time() - start_time:.2f} секунд.")


if __name__ == '__main__':
    input_file = 'generated_dataset.csv'
    output_file = 'processed_dataset.csv'
    chunk_size = 6_250_000
    n_cores = mp.cpu_count()
    process_dataset_in_chunks(input_file, chunk_size, n_cores, output_file)
