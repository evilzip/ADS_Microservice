import pandas as pd
import requests
import pickle
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from time import time
from plots.PlotBuilder import PlotBuilder
from utils.compress_df import compress_df
from utils.decompress_df import decompress_df

# from Imputation import Imputation
# from Models.MovingAverage import MovingAverage
# from Models.ExpoSmooth import ExpoSmooth
# from Models.DoubleExpoSmooth import DoubleExpoSmooth
# from Models.TripleExpoSmooth import TripleExpoSmooth
# from Models.HoltWinters import HoltWinters
# from Models.HoltWinters2 import HoltWinters2
# from Models.SARIMAX import SARIMAX
# from Models.SARIMAX_GS_smart import SARIMAX_GS_smart
# from Models.SARIMAX_GS import SARIMAX_GS
# from Models.SARIMAX_GS_2 import SARIMAX_GS_2
# from Models.auto_sarima_pmdarima import AutoArima
# from Models.XGBoostRegressor import XGBoostRegressor
# from Models.XGBRegressor2 import XGBoostRegressor2
# from Models.XGBRegressor3 import XGBoostRegressor3
# from Models.LSTM_1 import LSTM_1
# from Models.LSTM_2 import LSTM_2


pb = PlotBuilder()
# moving_average = MovingAverage()
# expo_smooth = ExpoSmooth()
# dbl_expo_smooth = DoubleExpoSmooth()
# tes = TripleExpoSmooth()
# hw = HoltWinters()
# xgb = XGBoostRegressor()
# xgb2 = XGBoostRegressor2()
# xgb3 = XGBoostRegressor3()
# lstm_1 = LSTM_1()
# lstm_2 = LSTM_2()
# sarimax_gs = SARIMAX_GS()
# sarimax_gs_2 = SARIMAX_GS_2()
# auto_Arima = AutoArima()

data = st.session_state.df_for_import

# global income_configured_data_frame
st.set_page_config(layout="wide")
# data = pd.read_csv('Data/configured_csv.csv')
if data is not None:
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
    imputation_df = decompress_df(impt_dict['impt_df'])
    missing_df = decompress_df(impt_dict['impt_missing_df'])
    # imputation_df = pickle.loads(bytes.fromhex(impt_dict['impt_df']))
    print(imputation_df)

    st.subheader('Timeseries with filled missing data', divider=True)
    fig = pb.plot_scatter(imputation_df, columns_list=['Raw_Data'])
    fig = pb.plot_imputed_data(fig=fig, data=imputation_df['spline'])
    st.plotly_chart(fig, theme=None, use_container_width=True, key='Imputation')
    fig = pb.bar_missing(imputation_df)
    st.plotly_chart(fig, theme=None, use_container_width=True, key='missing_bar')

    # model = xgb3

    # Main Calculation
    with st.spinner("Calculations in progress..."):
        xgb_payload = {"df_to_xgb": compress_df(imputation_df)}
        time_start = time()
        print('--------------------------before xgb request----------------------------------')
        xgb_request = requests.post('http://xgboost:8000/xgbregressor/', data=xgb_payload)
        print('--------------------------after xgb request------------------------------------')
        time_end = time()
        time_fitting = time_end - time_start

        xgb_dict = xgb_request.json()
        model_df = decompress_df(xgb_dict['xgb_model_df'])
        model_quality_df = decompress_df(xgb_dict['xgb_quality_df'])
        anomalies_df = decompress_df(xgb_dict['xgb_anomalies_df'])
        print(model_df)

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
    st.write(f'Время расчета: {round(time_fitting,2)} seconds')
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
else:
    st.write("Data has not configured yet or not found")

# # #
# Moving Average
