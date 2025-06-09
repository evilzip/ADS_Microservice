import pickle

import pandas as pd
import base64

d = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
}
# creating a Dataframe object
df = pd.DataFrame(d)
print(df)
#
pickeled_df = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
print(pickeled_df)
print(type(pickeled_df))

hex_str = pickeled_df.hex()

restored_df = pickle.loads(bytes.fromhex(hex_str))
print('restored', restored_df)



depickled_df = pickle.loads(pickeled_df)
print(depickled_df)
print(type(depickled_df))



# data_pickled = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
# pickled_b64 = base64.b64encode(data_pickled)
# print(pickled_b64)
#
# hug_pickled_str = pickled_b64.decode('utf-8')
# ss_df = pickle.loads(base64.b64decode(hug_pickled_str.encode()))
#
# print(ss_df)
# from io import StringIO
#
# df_json = df.to_json()
# print(df_json)
# print(type(df_json))
# df_from_json = pd.read_json(StringIO(df_json))
# print(df_from_json)
# print(type(df_from_json))
# -------------------
# Below section from sandbox
# imputation = Imputation(data)
# imputation.find_missing_data()
# imputation.impute_spline()
# Imputation_df должно прилетать из imputation fastapi
# data_pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
# pickled_b64 = base64.b64encode(data_pickled)
# pickled_string = pickled_b64.decode('utf-8')
data_test = {
    "name": "John Doe",
    "email": "pickled_b64"
}
# dataset = data.to_dict(orient='list')

# df_json = data.to_json()
# print(df_json)
# df_from_json = pd.read_json(StringIO(df_json))
d = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
}
# creating a Dataframe object
df = pd.DataFrame(d)