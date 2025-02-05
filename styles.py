import streamlit as st

def apply_custom_styles():
    st.markdown("""
        <style>
        .main {
            padding: 1rem;
            animation: fadeIn 0.6s ease-out;
        }

        .stNumberInput label, .stSelectbox label {
            font-weight: 500;
            color: #2c3e50;
            transition: color 0.3s ease;
        }

        .stNumberInput label:hover, .stSelectbox label:hover {
            color: #0096c7;
        }

        .plot-container {
            background-color: white;
            border-radius: 5px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            animation: fadeIn 0.8s ease-out;
        }

        .plot-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .section-header {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            transition: all 0.3s ease;
            animation: fadeIn 0.6s ease-out;
        }

        .section-header:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            animation: fadeIn 0.8s ease-out;
        }

        .metric-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .stButton>button {
            width: 100%;
            border-radius: 5px;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .stProgress .st-bo {
            background-color: #0096c7;
        }

        /* Enhanced tooltips */
        [data-tooltip]:hover:before {
            content: attr(data-tooltip);
            position: absolute;
            background: #2c3e50;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 1000;
            animation: fadeIn 0.3s ease-out;
        }
        </style>
    """, unsafe_allow_html=True)

def section_header(title, description=""):
    """Create an animated section header with hover effects"""
    st.markdown(f"""
        <div class="section-header hover-transition">
            <h3>{title}</h3>
            <p style="color: #666;">{description}</p>
        </div>
    """, unsafe_allow_html=True)

def metric_container(title, value, description=""):
    """Create an animated metric container with hover effects"""
    st.markdown(f"""
        <div class="metric-container hover-transition">
            <h4>{title}</h4>
            <div style="font-size: 24px; font-weight: bold;">{value}</div>
            <p style="color: #666; font-size: 14px;">{description}</p>
        </div>
    """, unsafe_allow_html=True)