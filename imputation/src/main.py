from fastapi import FastAPI, Body
import pandas as pd
from io import StringIO
import pickle
import base64
from imputation.Imputation import Imputation

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
    df = pickle.loads(bytes.fromhex(df_in[6:]))
    print(df)
    print(type(df))
    impt = Imputation(df)
    impt.find_missing_data()
    impt.impute_spline()
    return_str = pickle.dumps(impt.imputation_df, protocol=pickle.HIGHEST_PROTOCOL).hex()
    return {'impt_df' : return_str}


