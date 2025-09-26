import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px


st.set_page_config(
    page_title="Dynamic Forecasting App",
    page_icon="📈",
    layout="wide"
)


st.title("📈 Dynamic Time-Series Forecasting")
st.write("Upload your time-series data (CSV) and this app will generate a forecast.")


with st.sidebar:
    st.header("1. Configuration")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.header("2. Select Columns")
       
        date_column = st.selectbox("Select Date Column", options=df.columns)
        value_column = st.selectbox("Select Value Column to Forecast", options=df.columns)

        st.header("3. Set Forecast Parameters")
        
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Yearly": "AS"}
        frequency = st.selectbox("Select Data Frequency", options=list(freq_map.keys()))
        
        periods_to_forecast = st.slider("Periods to Forecast", min_value=1, max_value=365, value=90)
        
        generate_forecast_button = st.button("Generate Forecast", type="primary")

if uploaded_file is not None and 'generate_forecast_button' in locals() and generate_forecast_button:
    st.header("Forecast Results")

   
    df_processed = df[[date_column, value_column]].copy()
    df_processed.rename(columns={date_column: 'ds', value_column: 'y'}, inplace=True)
    
    
    try:
        df_processed['ds'] = pd.to_datetime(df_processed['ds'], dayfirst=True)
   
        df_processed = df_processed.groupby(pd.Grouper(key='ds', freq=freq_map[frequency])).sum().reset_index()
    except Exception as e:
        st.error(f"Error processing dates: {e}")
        st.stop()

    
    model = Prophet()
    model.fit(df_processed)
    
   
    future = model.make_future_dataframe(periods=periods_to_forecast, freq=freq_map[frequency])
    forecast = model.predict(future)

   
    st.write("### Forecast Plot")
   
    fig_forecast = plot_plotly(model, forecast)
    fig_forecast.update_layout(title="Forecasted Values with Confidence Interval",
                               xaxis_title="Date", yaxis_title=value_column)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.write("### Forecast Components")

    fig_components = plot_components_plotly(model, forecast)
    st.plotly_chart(fig_components, use_container_width=True)
    
    st.write("### Forecast Data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_to_forecast))

else:
    st.info("Please upload a CSV file and configure the settings in the sidebar to generate a forecast.")
