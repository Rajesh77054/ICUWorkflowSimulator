import streamlit as st

def load_animations():
    """Load CSS animations and transitions"""
    st.markdown("""
        <style>
        /* Fade-in animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Pulse animation for metrics */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        /* Smooth hover transition */
        .hover-transition {
            transition: all 0.3s ease;
        }
        
        /* Apply animations to elements */
        .animate-fade-in {
            animation: fadeIn 0.6s ease-out;
        }
        
        .metric-container {
            animation: fadeIn 0.8s ease-out;
            transition: all 0.3s ease;
        }
        
        .metric-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Chart container animations */
        .plot-container {
            animation: fadeIn 0.8s ease-out;
            transition: all 0.3s ease;
        }
        
        .plot-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Interactive buttons */
        .stButton>button {
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Smooth section transitions */
        .section-header {
            animation: fadeIn 0.6s ease-out;
            transition: all 0.3s ease;
        }
        
        /* Loading spinner animation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-spinner {
            width: 30px;
            height: 30px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #0096c7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        /* Expandable card animation */
        .expandable-card {
            transition: all 0.3s ease;
        }
        
        .expandable-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Progress bar animation */
        @keyframes progressBar {
            0% { width: 0; }
            100% { width: 100%; }
        }
        
        .progress-bar {
            animation: progressBar 2s ease-out;
        }
        </style>
    """, unsafe_allow_html=True)

def add_loading_animation():
    """Add a loading spinner animation"""
    return st.markdown("""
        <div class="loading-spinner"></div>
    """, unsafe_allow_html=True)

def wrap_with_animation(content, animation_class):
    """Wrap content with specified animation class"""
    return f'<div class="{animation_class}">{content}</div>'
