import pandas
import pickle


def compress_df(data: pandas.DataFrame) -> str:
    return_str = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL).hex()
    return return_str
