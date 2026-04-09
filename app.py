st.markdown("""
<style>

/* REMOVE BLUE BORDER (MULTISELECT) */
[data-baseweb="select"] > div {
    border: none !important;
    box-shadow: none !important;
}

/* REMOVE FOCUS BLUE OUTLINE */
[data-baseweb="select"] *:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* SELECTED TAG STYLE */
[data-baseweb="tag"] {
    background-color: rgba(0,255,255,0.2) !important;
    color: white !important;
    border-radius: 6px !important;
    border: 1px solid #00ffff !important;
}

/* TAG HOVER */
[data-baseweb="tag"]:hover {
    background-color: rgba(0,255,255,0.4) !important;
}

/* DROPDOWN */
[data-baseweb="menu"] {
    background-color: #1e1e1e !important;
    color: white !important;
}

/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #0f2027, #203a43);
}

</style>
""", unsafe_allow_html=True)
