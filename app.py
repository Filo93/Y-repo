import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from scipy.stats import poisson
import datetime

st.set_page_config(page_title="Contact Center Tool", layout="wide")
st.title("📊 Contact Center Planning Tool")

st.sidebar.title("Navigazione")
pagina = st.sidebar.selectbox("Scegli sezione", ["Forecasting", "Erlang Staffing"])

# === FORECASTING SECTION ===
if pagina == "Forecasting":
    st.markdown("""
    Questo tool consente di:
    1. Caricare un file CSV contenente dati storici (es. volume chiamate).
    2. Analizzare e selezionare la miglior granularità.
    3. Eseguire modelli di forecasting multipli (Prophet, ARIMA, Holt-Winters).
    4. Valutare gli errori e confrontare i modelli.
    5. Proiettare il miglior forecast su grafico.
    6. Scaricare il forecast in CSV.
    """)

    file = st.file_uploader("Carica il file CSV", type="csv")
    holidays_file = st.file_uploader("Carica file festività opzionale (colonna 'ds')", type="csv")

    if file:
        df = pd.read_csv(file, parse_dates=True)
        st.write("Anteprima dei dati:", df.head())

        all_cols = df.columns.tolist()
        date_col = st.selectbox("Seleziona la colonna temporale", all_cols)
        value_col = st.selectbox("Seleziona la colonna di volume (target)", all_cols)

        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        df = df[[date_col, value_col]].dropna()
        df = df.sort_values(by=date_col)
        df = df.rename(columns={date_col: "ds", value_col: "y"})

        df['diff'] = df['ds'].diff().dt.total_seconds()
        granularity_sec = df['diff'].mode()[0]
        if granularity_sec <= 3600:
            granularity = 'H'
        elif granularity_sec <= 86400:
            granularity = 'D'
        elif granularity_sec <= 604800:
            granularity = 'W'
        else:
            granularity = 'M'
        df = df.drop(columns=['diff'])
        st.success(f"Granularità rilevata: {granularity}")

        st.subheader("Visualizzazione della serie storica")
        fig = px.line(df, x='ds', y='y', title='Serie Temporale')
        st.plotly_chart(fig)

        st.subheader("Forecasting con modelli multipli")
        horizon = st.slider("Seleziona orizzonte di forecast (periodi)", 7, 90, 30)

        results = {}
        future_dates = pd.date_range(start=df['ds'].max(), periods=horizon + 1, freq=granularity)[1:]

        try:
            if holidays_file:
                holidays = pd.read_csv(holidays_file)
                holidays['ds'] = pd.to_datetime(holidays['ds'], dayfirst=True)
            else:
                years = df['ds'].dt.year.unique()
                holiday_dates = []
                for year in years:
                    holiday_dates.extend([
                        f"{year}-01-01", f"{year}-01-06", f"{year}-04-25", f"{year}-05-01",
                        f"{year}-06-02", f"{year}-08-15", f"{year}-11-01", f"{year}-12-08",
                        f"{year}-12-25", f"{year}-12-26",
                    ])
                holidays = pd.DataFrame({'ds': pd.to_datetime(holiday_dates), 'holiday': 'festività_italiane'})

            model_prophet = Prophet(yearly_seasonality=True, holidays=holidays)
            model_prophet.fit(df)
            future = model_prophet.make_future_dataframe(periods=horizon, freq=granularity)
            forecast_prophet = model_prophet.predict(future)
            prophet_pred = forecast_prophet.set_index('ds').loc[future_dates]['yhat']
            results['Prophet'] = prophet_pred.values
        except:
            results['Prophet'] = [np.nan] * horizon

        try:
            model_hw = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=7).fit()
            hw_forecast = model_hw.forecast(horizon)
            results['Holt-Winters'] = hw_forecast.values
        except:
            results['Holt-Winters'] = [np.nan] * horizon

        try:
            model_arima = ARIMA(df['y'], order=(1,1,1)).fit()
            arima_forecast = model_arima.forecast(steps=horizon)
            results['ARIMA'] = arima_forecast.values
        except:
            results['ARIMA'] = [np.nan] * horizon

        st.subheader("Valutazione degli errori")
        cutoff = len(df) - horizon
        train = df.iloc[:cutoff]
        test = df.iloc[cutoff:]
        y_true = test['y'].values

        error_table = []
        for model, preds in results.items():
            if len(preds) == len(y_true):
                mae = mean_absolute_error(y_true, preds)
                rmse = mean_squared_error(y_true, preds) ** 0.5
                mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
                error_table.append({"Modello": model, "MAE": mae, "RMSE": rmse, "MAPE": mape})

        df_errors = pd.DataFrame(error_table).sort_values(by="MAPE")
        st.dataframe(df_errors)

        if not df_errors.empty:
            best_model = df_errors.iloc[0]['Modello']
            st.success(f"Il modello più accurato è: {best_model}")
            future_df = pd.DataFrame({"ds": future_dates, "Forecast": results[best_model]})
            df_plot = pd.concat([df, future_df], axis=0)
            fig3 = px.line(df_plot, x="ds", y="y" if 'y' in df_plot else "Forecast", title="Forecast finale")
            st.plotly_chart(fig3)

            csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Scarica il forecast in CSV", data=csv, file_name='forecast_output.csv', mime='text/csv')

# === ERLANG STAFFING SECTION ===
elif pagina == "Erlang Staffing":
    st.markdown("""
    Questo modulo calcola il numero minimo di operatori richiesti per ogni periodo forecastato usando l'algoritmo di Erlang C.
    Inserisci:
    - un file CSV con i volumi forecastati (colonna `ds`, `Forecast`)
    - obiettivi di servizio
    """)
    uploaded = st.file_uploader("Carica il file di forecast (output precedente)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Anteprima del forecast:", df.head())

        aht = st.number_input("Durata media chiamata (AHT, in secondi)", value=300)
        sla_time = st.number_input("Tempo massimo per rispondere (secondi)", value=20)
        sla_target = st.slider("Target livello di servizio (es. 0.80 = 80%)", 0.5, 0.95, 0.80)
        shrinkage = st.slider("Shrinkage (assenze, pause, training... es. 0.30 = 30%)", 0.0, 0.5, 0.30)

        def erlang_c_formula(traffic_intensity, agents, target_time, aht):
            ErlangC = ((traffic_intensity ** agents) / np.math.factorial(agents)) / (
                sum((traffic_intensity ** n) / np.math.factorial(n) for n in range(agents)) +
                ((traffic_intensity ** agents) / np.math.factorial(agents)) * (1 - traffic_intensity / agents)
            )
            pw = ErlangC * np.exp(-(agents - traffic_intensity) * (target_time / aht))
            return pw

        def find_min_agents(lambda_calls, aht, sla_target, sla_time):
            traffic = lambda_calls * (aht / 3600)
            for n in range(1, 500):
                pw = erlang_c_formula(traffic, n, sla_time, aht)
                if 1 - pw >= sla_target:
                    return n
            return None

        df['Forecast'] = df['Forecast'].fillna(0)
        df['Traffic (erlang)'] = df['Forecast'] * (aht / 3600)
        df['Operatori (netti)'] = df['Forecast'].apply(lambda x: find_min_agents(x, aht, sla_target, sla_time))
        df['Operatori (lordi)'] = (df['Operatori (netti)'] / (1 - shrinkage)).apply(np.ceil)

        st.subheader("📊 Staffing suggerito")
        st.dataframe(df[['ds', 'Forecast', 'Operatori (lordi)']])

        csv_out = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica staffing CSV", data=csv_out, file_name="staffing_output.csv", mime="text/csv")
