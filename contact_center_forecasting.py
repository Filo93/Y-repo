import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

# Custom CSS: riduce font del 20%
st.markdown("""
    <style>
    .css-1d391kg, .css-1v3fvcr, .css-qcqlej, .css-16idsys {
        font-size: 80% !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Hello")

    selected = option_menu(
        menu_title=None,
        options=[
            "📈 Forecaster", 
            "🧮 Capacity Planner", 
            "👥 Workforce Management", 
            "✅ Quality Assurance"
        ],
        icons=["graph-up", "calculator", "people", "check2-square"],
        menu_icon="cast", 
        default_index=0
    )

    st.markdown("---")
    st.button("Logout")
    st.caption("Your session id: 250626_abcd12")

# --- Contenuto pagina ---
st.title(f"{selected}")
st.markdown(f"**Info finder** &nbsp;&nbsp; {selected} requests")
st.markdown("<hr style='margin-top: -10px;'>", unsafe_allow_html=True)
