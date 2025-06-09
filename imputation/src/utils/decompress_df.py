import pickle
import pandas


def decompress_df(df_str: str, init_pos: int) -> pandas.DataFrame:
    return_df = pickle.loads(bytes.fromhex(df_str[init_pos:]))
    return return_df
