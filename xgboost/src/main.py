import uvicorn
from fastapi import FastAPI, Body

from xgbregressor.xgbregressor import XGBoostRegressor3
from utils.compress_df import compress_df
from utils.decompress_df import decompress_df

# Initialize Imputation object


# Initialize FastAPI app
app = FastAPI(debug=True)


# Define FastAPI endpoints
@app.get("/")
async def read_root():
    return {"message": "Welcome to the XGBoost regressor API!"}


@app.post("/xgbregressor/")
async def xgbregressor(df_to_xgb: str = Body(...)):
    income_df = decompress_df(df_str=df_to_xgb, init_pos=len('df_to_xgb='))
    print('-----------income df---------------')
    print(income_df)
    print('--------------------------------------')
    xgb_model = XGBoostRegressor3()
    xgb_model.fit_predict(data=income_df)
    xgb_model.anomalies()
    model_df_str = compress_df(xgb_model.model_df)
    quality_df_str = compress_df(xgb_model.model_quality_df)
    anomalies_df_str = compress_df(xgb_model.anomalies_df)
    return {
        'xgb_model_df': model_df_str,
        'xgb_quality_df': quality_df_str,
        'xgb_anomalies_df': anomalies_df_str
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)

