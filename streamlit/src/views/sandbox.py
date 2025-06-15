import requests
import streamlit as st
from time import time
import asyncio
import aiohttp
from plots.PlotBuilder import PlotBuilder
from utils.compress_df import compress_df
from utils.decompress_df import decompress_df
from utils.extrct_model_qlty import extract_model_winner_dfs
from config import model_urls
from model_requests.post_requests import fetch_all

# global income_configured_data_frame
st.set_page_config(layout="wide")

XGBOOST_URL = 'http://89.104.65.117:8000/xgbregressor/'
# LSTM_URL = 'http://lstm:8000/lstm/'
LSTM_URL = 'http://87.228.101.34:8000/lstm/'
SARIMAGS_URL = 'http://89.104.66.194:8000/sarimags/'
HW_URL = 'http://89.104.66.247:8000/holtwinters/'

pb = PlotBuilder()
data = None

try:
    data = st.session_state.df_for_import
except Exception as e:
    st.error('Timeseries not configured or not found')

# data = pd.read_csv('Data/configured_csv.csv')
if data is not None:
    if len(data) > 300:
        urls = [XGBOOST_URL,
                LSTM_URL]
    else:
        urls = [HW_URL,
                LSTM_URL,
                SARIMAGS_URL,
                XGBOOST_URL]

    # Raw data
    st.subheader('Your Loaded TimeSeries', divider=True)
    fig = pb.plot_scatter(data, columns_list=['Raw_Data'])
    st.plotly_chart(fig, theme=None, use_container_width=True, key='raw')

    # 1. Imputation
    # все это должно делаться на стороне API
    payload = {"df_in": compress_df(data)}
    print('--------------------------before imput request----------------------------------')
    # print(payload)
    impt_request = requests.post('http://imputation:8000/imputation/', data=payload)
    print('--------------------------after imput request------------------------------------')
    # print(impt_request.text)
    impt_dict = impt_request.json()

    if 'imputation_df' not in st.session_state:
        st.session_state['imputation_df'] = decompress_df(impt_dict['impt_df'])
    else:
        if st.session_state.new_data_flag == True:
            st.session_state['imputation_df'] = decompress_df(impt_dict['impt_df'])

    if 'missing_df' not in st.session_state:
        st.session_state['missing_df'] = decompress_df(impt_dict['impt_missing_df'])
    else:
        if st.session_state.new_data_flag == True:
            st.session_state['missing_df'] = decompress_df(impt_dict['impt_missing_df'])

    # imputation_df = decompress_df(impt_dict['impt_df'])
    # missing_df = decompress_df(impt_dict['impt_missing_df'])
    imputation_df = st.session_state.imputation_df
    missing_df = st.session_state.missing_df

    print(imputation_df)

    st.subheader('Timeseries with filled missing data', divider=True)
    fig = pb.plot_scatter(imputation_df, columns_list=['Raw_Data'])
    fig = pb.plot_imputed_data(fig=fig, data=imputation_df['spline'])
    st.plotly_chart(fig, theme=None, use_container_width=True, key='Imputation')
    fig = pb.bar_missing(imputation_df)
    st.plotly_chart(fig, theme=None, use_container_width=True, key='missing_bar')

    # Main Calculation
    if st.session_state.new_data_flag == True:
        with st.spinner("Calculations in progress..."):
            payload = {"df_to_models": compress_df(imputation_df)}
            time_start = time()
            # run the async functions
            results = asyncio.run(fetch_all(urls=urls, payload=payload))
            # print('------------------results--------------------')
            # print(results)
            # print('------------------end results--------------------')
            st.session_state['model_name'], \
                st.session_state['model_df'], \
                st.session_state['anomalies_df'], \
                st.session_state['model_quality_df'] = extract_model_winner_dfs(results)

            time_end = time()
            st.session_state['time_fitting'] = time_end - time_start

    model_name = st.session_state['model_name']
    model_df = st.session_state['model_df']
    anomalies_df = st.session_state['anomalies_df']
    model_quality_df = st.session_state['model_quality_df']
    time_fitting = st.session_state['time_fitting']

    print('------------------best mape model--------------------')
    print(model_name)
    print(model_quality_df)
    print('------------------after best mape model--------------------')

    #
    st.subheader('Timeseries model data with marked anomalies', divider=True)
    # fig = pb.plot_model_scatter(data=sarimax_gs_smart_index.model_df_least_MAPE,
    #                             columns_list=['Raw_Data', 'Y_Predicted'])
    fig = pb.plot_model_scatter(data=model_df,
                                columns_list=['Raw_Data', 'Y_Predicted'])
    st.plotly_chart(fig, theme=None, use_container_width=True, key='model_scatter')
    fig = pb.bar_anomalies(data=model_df)
    st.plotly_chart(fig, theme=None, use_container_width=True, key='anomalies_heatmap')

    # Model quality section
    st.subheader("Model Quality", divider=True)
    # st.write(sarimax_gs_smart_index.model_quality_df)
    st.write(f'Model: {model_name}')
    st.write(f'Calculation time: {round(time_fitting, 2)} seconds')
    st.write(model_quality_df)

    # Anomalies section
    st.subheader("Anomalies", divider=True)
    col1, col2 = st.columns([0.5, 0.5], vertical_alignment='top')
    with col1:
        st.subheader("Anomalies table")
        # st.write(sarimax_gs_smart_index.anomalies())
        st.write(anomalies_df)
    with col2:
        st.subheader("Anomalies amount")
        # fig = pb.anomalies_pie(sarimax_gs_smart_index.model_df_least_MAPE)
        fig = pb.anomalies_pie(model_df)
        st.plotly_chart(fig, theme=None, use_container_width=True, key='anomalies_pie_chart')

    # Missing/Imputation section
    st.subheader('Missing data', divider=True)
    col1, col2 = st.columns([0.3, 0.7], vertical_alignment='top')
    with col1:
        st.subheader("Missing data table")
        st.write(missing_df)
    with col2:
        st.subheader("Missing data amount")
        fig = pb.missing_pie(imputation_df)
        st.plotly_chart(fig, theme=None, use_container_width=True, key='missing_pie_chart')

    # del lstm_2, lstm_1, xgb3, xgb2, hw, SARIMAX_GS
    st.session_state.new_data_flag = False
else:
    st.warning("Go to Upload page and configure timeseries for analysis")

# # #
# Moving Average
