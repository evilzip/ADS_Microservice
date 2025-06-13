import pandas
from .decompress_df import decompress_df


def extract_model_qlty(model_dict: dict, metric='MAPE') -> float:
    return model_dict[metric]


def extract_model_winner_dfs(models_dict: dict, metric='MAPE'):
    min_quality = 1
    min_pos = 0
    for pos, dict in enumerate(models_dict):
        current_quality_df = decompress_df(dict['quality_df'])
        current_min_quality = current_quality_df[metric].iloc[0]
        if current_min_quality < min_quality:
            min_quality = current_min_quality
            min_pos = pos
    winner_model_quality_df = decompress_df(models_dict[min_pos]['quality_df'])
    winner_model_df = decompress_df(models_dict[min_pos]['model_df'])
    winner_anomalies_df = decompress_df(models_dict[min_pos]['anomalies_df'])
    winner_model_name = models_dict[min_pos]['model_name']
    return winner_model_name, winner_model_df, winner_anomalies_df, winner_model_quality_df
