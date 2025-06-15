import uvicorn
from fastapi import FastAPI, Body

from sarimags.sarimags import SARIMAX_GS_2
from utils.compress_df import compress_df
from utils.decompress_df import decompress_df

# Initialize Imputation object
MODEL_NAME = 'SARIMAGX'

# Initialize FastAPI app
app = FastAPI(debug=True)


# Define FastAPI endpoints
@app.get("/")
async def read_root():
    return {"message": "SARIMAX model API"}


@app.get("/health")
async def health_check():
    return {'model_name': MODEL_NAME,
            'status': 'OK'}


@app.post("/sarimags/")
async def sarimags(df_to_models: str = Body(...)):
    income_df = decompress_df(df_str=df_to_models, init_pos=len('df_to_models='))
    print('-----------income df---------------')
    print(income_df)
    print('--------------------------------------')
    model = SARIMAX_GS_2()
    model.fit_predict(data=income_df)
    model.anomalies()
    print('-----------model df---------------')
    print(model.model_df)
    print('--------------------------------------')
    print('-----------model.model_quality_df------')
    print(model.model_quality_df)
    print('--------------------------------------')
    print('-----------model.anomalies_df---------------')
    print(model.anomalies_df)
    print('--------------------------------------')
    model_df_str = compress_df(model.model_df)
    quality_df_str = compress_df(model.model_quality_df)
    anomalies_df_str = compress_df(model.anomalies_df)
    return {
        'model_name': MODEL_NAME,
        'model_df': model_df_str,
        'quality_df': quality_df_str,
        'anomalies_df': anomalies_df_str
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
