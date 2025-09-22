import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px

# --- App Configuration ---
st.set_page_config(
    page_title="Dynamic Forecasting App",
    page_icon="📈",
    layout="wide"
)

# --- Main App UI ---
st.title("📈 Dynamic Time-Series Forecasting")
st.write("Upload your time-series data (CSV) and this app will generate a forecast.")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("1. Configuration")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.header("2. Select Columns")
        # Let user select the date and value columns
        date_column = st.selectbox("Select Date Column", options=df.columns)
        value_column = st.selectbox("Select Value Column to Forecast", options=df.columns)

        st.header("3. Set Forecast Parameters")
        # Map user-friendly frequency names to pandas frequency strings
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Yearly": "AS"}
        frequency = st.selectbox("Select Data Frequency", options=list(freq_map.keys()))
        
        periods_to_forecast = st.slider("Periods to Forecast", min_value=1, max_value=365, value=90)
        
        generate_forecast_button = st.button("Generate Forecast", type="primary")

# --- Main Content Area for Output ---
if uploaded_file is not None and 'generate_forecast_button' in locals() and generate_forecast_button:
    st.header("Forecast Results")

    # 1. Data Processing
    # Create a copy to avoid modifying the original dataframe
    df_processed = df[[date_column, value_column]].copy()
    df_processed.rename(columns={date_column: 'ds', value_column: 'y'}, inplace=True)
    
    # Convert 'ds' to datetime and handle potential errors
    try:
        df_processed['ds'] = pd.to_datetime(df_processed['ds'], dayfirst=True)
        # Aggregate data to the chosen frequency, summing values for duplicate dates
        df_processed = df_processed.groupby(pd.Grouper(key='ds', freq=freq_map[frequency])).sum().reset_index()
    except Exception as e:
        st.error(f"Error processing dates: {e}")
        st.stop()

    # 2. Modeling with Prophet
    # Instantiate and fit the model
    model = Prophet()
    model.fit(df_processed)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods_to_forecast, freq=freq_map[frequency])
    forecast = model.predict(future)

    # 3. Visualization
    st.write("### Forecast Plot")
    # Use Prophet's interactive plotting with Plotly
    fig_forecast = plot_plotly(model, forecast)
    fig_forecast.update_layout(title="Forecasted Values with Confidence Interval",
                               xaxis_title="Date", yaxis_title=value_column)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.write("### Forecast Components")
    # Plot the components (trend, weekly/yearly seasonality)
    fig_components = plot_components_plotly(model, forecast)
    st.plotly_chart(fig_components, use_container_width=True)
    
    st.write("### Forecast Data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_to_forecast))

else:
    st.info("Please upload a CSV file and configure the settings in the sidebar to generate a forecast.")