import pandas as pd
import random
import numpy as np

def generate_missing_data(data):
    random.seed(77)
    # посчитаем количество наблюдений
    n_samples = len(data['Raw_Data'])
    # вычислим 20 процентов от этого числа,
    # это будет количество пропусков
    how_many = int(0.01 * n_samples)
    # случайным образом выберем 20 процентов значений индекса
    mask_target = random.sample(list(data.index.values), how_many)
    # и заполним их значением NaN в столбце target
    data.loc[mask_target] = np.nan
    return data