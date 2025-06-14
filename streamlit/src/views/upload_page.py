import streamlit as st
import pandas as pd
from pandas._libs.tslibs.parsing import DateParseError
from utils.generate_missing_data import generate_missing_data


df_for_import = pd.DataFrame()

st.title("Upload and Configure timeseries")

#
st.subheader("1. Upload your data", divider=True)
uploaded_file = st.file_uploader("Upload dataset here - Only '*.csv' File", key="upload_1")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, thousands=',')
    except FileNotFoundError:
        print("File not found.")
    except pd.errors.EmptyDataError:
        print("Empty file - No data")
    except pd.errors.ParserError:
        print("Parse error")
    except Exception:
        print("Some other exception")

    st.subheader("Loaded data")
    st.write(df)
    # Sample data
    # Form to select columns
    st.subheader("2. Select Time and Value columns", divider=True)
    with st.form(key='configure_column'):
        income_configured_data_frame = pd.DataFrame(columns=['Time', 'Raw_Data'])
        col1, col2 = st.columns(2, vertical_alignment="bottom")

        with col1:  # Select time column
            time_column = tuple(df.columns.tolist() + ['Indexes'])
            # Dropdown menu for selecting a category
            selected_time_column = st.selectbox('Select a time column', time_column, key='time_column')
            # Display the selected category
            st.write('Time column:', selected_time_column)
        with col2:  # Select value column
            value_column = tuple(df.columns.tolist())
            selected_value_column = st.selectbox("select a value column", value_column, key='value_column')
            st.write('Value column:', selected_value_column)
        # тут нужно сделать обраточик ошибки

        income_configured_data_frame['Time'] = df[selected_time_column].values
        income_configured_data_frame['Raw_Data'] = df[selected_value_column].values
        # income_configured_data_frame['Raw_Data'] = income_configured_data_frame['Raw_Data'].astype('float')
        try:
            income_configured_data_frame['Time'] = pd.to_datetime(income_configured_data_frame['Time'])
            step_2_error = False
        except DateParseError as e:
            st.error(f'{e}')
            step_2_error = True

        income_configured_data_frame = income_configured_data_frame.set_index('Time')
        column_submit = st.form_submit_button("Submit columns selection")

    if not step_2_error:
        st.subheader("3. Select time range", divider=True)
        with st.form(key='range_configure'):
            options = income_configured_data_frame.index
            # print("options", options)
            min_value = options[0]
            max_value = options[-1]
            #Time range slider
            start_income_df, end_income_df = st.select_slider(
                "Select a range for analysis",
                options=options,
                value=(min_value, max_value),
                key='slider_time_limits'
            )
            #print("start_income_df, end_income_df", start_income_df, end_income_df)
            income_configured_data_frame = income_configured_data_frame.loc[start_income_df:end_income_df]
            range_submit = st.form_submit_button("Submit range selection")

        st.subheader("4. Review configured dataframe and submit for analysis", divider=True)
        st.write('Yours configured timeseries dataframe', income_configured_data_frame)
        if st.button('Submit for analysis', key='main_confirm_btn'):
            # import income_configured_data_frame to plot builder ?????
            # st.session_state - python dictionary
            # Add income_configured_data_frame as value with key 'df_for_import'
            # st.session_state = {'df_for_import' : income_configured_data_frame}
            income_configured_data_frame = generate_missing_data(income_configured_data_frame)
            if 'df_for_import' not in st.session_state:
                st.session_state['df_for_import'] = income_configured_data_frame
            else:
                st.session_state['df_for_import'] = income_configured_data_frame
            df_for_import = income_configured_data_frame
            # income_configured_data_frame.to_csv('Data/configured_csv.csv')
            st.subheader("5. Configured timeseries has been sent for analysis. Go to Result page", divider=True)
            # print("data frame for analisis", income_configured_data_frame)
            # print("INFO", type(income_configured_data_frame.index))
            # print("INFO", type(income_configured_data_frame.Raw_Data))
            st.session_state['new_data_flag'] = True

