import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide", page_title="Consultant Toolkit")

with st.sidebar:
    selected_main = option_menu(
        "🧰 Consultant Toolkit", 
        ["Forecasting", "Sizing", "Planning", "WFM", "Commissioning", "Quality Assurance"],
        icons=["bar-chart", "grid-1x2", "calendar", "clock", "graph-up", "check2-square"],
        menu_icon="tools", 
        default_index=0,
        orientation="vertical",
    )

    if selected_main == "Forecasting":
        selected_sub = option_menu(
            None,
            ["Experimental", "Prophet", "Holt-Winters", "ARIMA", "SARIMA"],
            icons=["bezier", "graph-up", "activity", "diagram-3", "scatter"],
            menu_icon="cast", default_index=0
        )
    else:
        selected_sub = None

# Caricamento moduli
if selected_main == "Forecasting":
    if selected_sub == "Experimental":
        from forecasting.experimental import run_experimental
        run_experimental()
    else:
        st.info(f"Modulo '{selected_sub}' da implementare.")
else:
    st.info(f"Modulo '{selected_main}' in sviluppo.")
