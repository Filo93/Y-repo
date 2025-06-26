import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

# Custom CSS stile Druso senza icone
st.markdown("""
    <style>
    /* Font piccolo e grigio chiaro */
    .css-1d391kg, .css-1v3fvcr, .css-qcqlej, .css-16idsys, .nav-link {
        font-size: 70% !important;
        color: #d3d3d3 !important;
    }

    /* Elemento attivo */
    .nav-link.active {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 5px;
    }

    /* Hover */
    .nav-link:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #f0f0f0 !important;
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
        icons=[],  # ← Nessuna icona
        menu_icon="cast", 
        default_index=0
    )

    st.markdown("---")
    st.button("Logout")
    st.caption("Your session id: 250626_abcd12")

# --- Contenuto centrale ---
st.title(f"{selected}")
st.markdown(f"**Info finder** &nbsp;&nbsp; {selected} requests")
st.markdown("<hr style='margin-top: -10px;'>", unsafe_allow_html=True)
