import streamlit as st
import requests
import asyncio
from model_requests.get_requests import fetch_all_health

st.header("Anomaly Detection Software", divider=True)
st.subheader('This is visual and functional prototype of project "ADSoft"')

XGBOOST_HEALTH_URL = 'http://xgboost:8000/health/'
LSTM_URL = 'http://lstm:8000/lstm/'
SARIMAGS_URL = 'http://sarimags:8000/sarimags/'
HW_HEALTH_URL = 'http://holtwinters:8000/health/'

health_urls = [
    XGBOOST_HEALTH_URL,
    LSTM_URL,
    SARIMAGS_URL,
    HW_HEALTH_URL
]


with st.container(border=True):
    st.write('Model health check')
    with st.spinner("APIs health check..."):
        result = asyncio.run(fetch_all_health(health_urls))
    st.success("APIs health ccheck completed")

    st.write('APIs status')

    container = st.container(border=True)
    if result[0] == 'OK':
        st.success(f'XGBoost API: {result[0]}')
    else:
        st.error(f'XGBoost API: {result[0]}')

    if result[1] == 'OK':
        st.success(f'LSTM API: {result[1]}')
    else:
        st.error(f'LSTM API: {result[1]}')

    if result[2] == 'OK':
        st.success(f'SARIMA API: {result[2]}')
    else:
        st.error(f'SARIMA API: {result[2]}')

    if result[3] == 'OK':
        st.success(f'HoltWinters API: {result[3]}')
    else:
        st.error(f'HoltWinters API: {result[3]}')
