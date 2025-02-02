import streamlit as st
import numpy as np
from styles import apply_custom_styles, section_header
from utils import (calculate_interruptions, calculate_workload,
                  create_interruption_chart, create_time_allocation_pie,
                  create_workload_timeline)
from simulator import WorkflowSimulator

def main():
    st.set_page_config(
        page_title="ICU Workflow Dynamics Model",
        page_icon="🏥",
        layout="wide"
    )
    
    apply_custom_styles()
    
    st.title("ICU Workflow Dynamics Model")
    st.markdown("""
        This interactive tool helps analyze and visualize ICU workflow dynamics,
        considering various factors that impact provider efficiency and patient care.
    """)
    
    simulator = WorkflowSimulator()
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Interruptions", "Enter frequency of different types of interruptions")
        nursing_q = st.number_input("Nursing Questions (per hour)", 0.0, 20.0, 5.0, 0.5)
        exam_callbacks = st.number_input("Exam Callbacks (per hour)", 0.0, 20.0, 3.0, 0.5)
        peer_interrupts = st.number_input("Peer Interruptions (per hour)", 0.0, 20.0, 2.0, 0.5)
        
        section_header("Provider Information", "Enter staffing details")
        providers = st.number_input("Number of Providers", 1, 10, 2)
        
    with col2:
        section_header("Admissions & Transfers", "Enter patient flow information")
        admissions = st.number_input("New Admissions (per shift)", 0, 20, 3)
        consults = st.number_input("Floor Consults (per shift)", 0, 20, 4)
        transfers = st.number_input("Transfer Center Calls (per shift)", 0, 20, 2)
        
        section_header("Critical Events", "Enter frequency of critical events")
        critical_events = st.number_input("Critical Events (per week)", 0, 50, 5)
        
    # Calculate metrics
    interrupts_per_provider, time_lost = calculate_interruptions(
        nursing_q, exam_callbacks, peer_interrupts, providers
    )
    workload = calculate_workload(admissions, consults, transfers, critical_events/7, providers)
    efficiency = simulator.simulate_provider_efficiency(
        nursing_q + exam_callbacks + peer_interrupts,
        providers
    )
    burnout_risk = simulator.calculate_burnout_risk(workload, interrupts_per_provider)
    cognitive_load = simulator.calculate_cognitive_load(
        interrupts_per_provider,
        critical_events/7,
        admissions
    )
    
    # Display metrics
    st.markdown("### Key Metrics")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Interruptions per Provider", f"{interrupts_per_provider:.1f}/shift")
    with metric_col2:
        st.metric("Hours Lost to Interruptions", f"{time_lost:.1f}")
    with metric_col3:
        st.metric("Provider Efficiency", f"{efficiency:.1%}")
    with metric_col4:
        st.metric("Burnout Risk", f"{burnout_risk:.1%}")
    
    # Visualizations
    st.markdown("### Workflow Analysis")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.plotly_chart(
            create_interruption_chart(nursing_q, exam_callbacks, peer_interrupts),
            use_container_width=True
        )
        
    with viz_col2:
        st.plotly_chart(
            create_time_allocation_pie(time_lost),
            use_container_width=True
        )
    
    st.plotly_chart(
        create_workload_timeline(workload, providers),
        use_container_width=True
    )
    
    # Recommendations
    st.markdown("### Recommendations")
    if burnout_risk > 0.7:
        st.warning("⚠️ High burnout risk detected. Consider increasing provider coverage or implementing interruption reduction strategies.")
    if cognitive_load > 80:
        st.warning("⚠️ High cognitive load detected. Consider workflow optimization or additional support staff.")
    if efficiency < 0.7:
        st.warning("⚠️ Low efficiency detected. Review interruption patterns and implement protected time for critical tasks.")

if __name__ == "__main__":
    main()
