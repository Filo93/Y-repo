import streamlit as st

st.set_page_config(page_title="Contact Center Suite", layout="wide")

st.sidebar.title("🧰 Consultant Toolkit")

menu = st.sidebar.radio("Seleziona modulo", [
    "Forecasting",
    "Sizing",
    "Planning",
    "WFM",
    "Commissioning",
    "Quality Assurance"
])

if menu == "Forecasting":
    from forecasting.experimental import run_experimental
    forecasting_submenu = st.sidebar.radio("Modelli Forecasting", [
        "Experimental", "Prophet", "Holt-Winters", "ARIMA", "SARIMA"
    ])

    if forecasting_submenu == "Experimental":
        run_experimental()
    else:
        st.info(f"Modulo '{forecasting_submenu}' da implementare.")
else:
    st.info(f"Modulo '{menu}' in sviluppo.")
