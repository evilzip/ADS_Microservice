import uvicorn
from fastapi import FastAPI, Body
import pandas as pd
import pickle

from imputation.Imputation import Imputation
from utils.compress_df import compress_df
from utils.decompress_df import decompress_df

# Initialize Imputation object


# Initialize FastAPI app
app = FastAPI(debug=True)


# Define FastAPI endpoints
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Imputation API!"}


@app.post("/imputation/")
async def imputation(df_in: str = Body(...)):
    # print(df_in)
    # df = pickle.loads(bytes.fromhex(df_in[6:]))
    # print(df)
    # print(type(df))
    # impt = Imputation(df)
    # impt.find_missing_data()
    # impt.impute_spline()
    # return_str = pickle.dumps(impt.imputation_df, protocol=pickle.HIGHEST_PROTOCOL).hex()
    # return {'impt_df' : return_str}
    income_df = decompress_df(df_str=df_in, init_pos=len('df_in='))
    print('-----------income df---------------')
    print(income_df)
    print('--------------------------------------')
    impt = Imputation(income_df)
    impt.find_missing_data()
    impt.impute_spline()
    impt_df_str = compress_df(impt.imputation_df)
    missing_df_str = compress_df(impt.missing_df)
    return {
        'impt_df': impt_df_str,
        'impt_missing_df': missing_df_str
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)


