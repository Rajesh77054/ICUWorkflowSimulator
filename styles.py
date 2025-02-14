import streamlit as st

def apply_custom_styles():
    st.markdown("""
        <style>
        .main {
            padding: 1rem;
        }
        .stNumberInput label, .stSelectbox label {
            font-weight: 500;
            color: #2c3e50;
        }
        .plot-container {
            background-color: white;
            border-radius: 5px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section-header {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def section_header(title, description=""):
    st.markdown(f"""
        <div class="section-header">
            <h3>{title}</h3>
            <p style="color: #666;">{description}</p>
        </div>
    """, unsafe_allow_html=True)
