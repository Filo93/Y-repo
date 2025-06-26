import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

# Sidebar stile Druso
with st.sidebar:
    st.markdown("### Hello")

    selected = option_menu(
        menu_title=None,
        options=[
            "Generative Excel", 
            "Deep Extractor", 
            "Batches Monitor",
            "My Assistants", 
            "Audio Transcriber", 
            "Doc Assistant", 
            "Settings & Recovery"
        ],
        icons=[
            "file-earmark-excel", 
            "cone-striped", 
            "list-task",
            "emoji-smile", 
            "mic", 
            "file-earmark-text", 
            "tools"
        ],
        menu_icon="cast", 
        default_index=1
    )

    st.markdown("---")
    st.button("Logout")
    st.caption("Your session id: 250626_2c5a0e")

# --- Contenuto principale ---
st.title(f"🔧 - {selected}")
st.markdown(f"**Info finder** &nbsp;&nbsp; {selected} requests")
st.markdown("<hr style='margin-top: -10px;'>", unsafe_allow_html=True)
