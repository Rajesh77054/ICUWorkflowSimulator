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
        This interactive tool helps analyze and visualize ICU dayshift workflow dynamics (8 AM - 8 PM),
        considering various factors that impact provider efficiency and patient care.
    """)

    simulator = WorkflowSimulator()

    # Add configuration section with an expander
    with st.expander("⚙️ Time Settings Configuration"):
        st.markdown("### Configure Time Estimates")
        st.markdown("Adjust the time estimates for various activities below:")

        col1, col2 = st.columns(2)

        with col1:
            section_header("Interruption Times", "Duration of different types of interruptions")
            nursing_time = st.number_input("Nursing Question Duration (minutes)", 1, 10, 2)
            exam_callback_time = st.number_input("Exam Callback Duration (minutes)", 1, 20, 8)
            peer_interrupt_time = st.number_input("Peer Interruption Duration (minutes)", 1, 20, 8)

        with col2:
            section_header("Admission & Critical Event Times", "Duration of patient care activities")
            simple_admission_time = st.number_input("Simple Admission Duration (minutes)", 30, 120, 60)
            complex_admission_time = st.number_input("Complex Admission Duration (minutes)", 60, 180, 90)
            critical_event_time = st.number_input("Critical Event Duration (minutes)", 60, 180, 105)

        # Update simulator settings
        simulator.update_time_settings({
            'interruption_times': {
                'nursing_question': nursing_time,
                'exam_callback': exam_callback_time,
                'peer_interrupt': peer_interrupt_time
            },
            'admission_times': {
                'simple': simple_admission_time,
                'complex': complex_admission_time,
                'consult': 45,  # keeping these fixed for now
                'transfer': 30
            },
            'critical_event_time': critical_event_time
        })

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        section_header("Interruptions", "Enter frequency of different types of interruptions during dayshift")
        nursing_q = st.number_input("Nursing Questions (per hour)", 0.0, 20.0, 5.0, 0.5)
        exam_callbacks = st.number_input("Exam Callbacks (per hour)", 0.0, 20.0, 3.0, 0.5)
        peer_interrupts = st.number_input("Peer Interruptions (per hour)", 0.0, 20.0, 2.0, 0.5)

        section_header("Provider Information", "Enter dayshift staffing details")
        providers = st.number_input("Number of Providers", 1, 10, 2)

    with col2:
        section_header("Admissions & Transfers", "Enter patient flow information for dayshift")
        admissions = st.number_input("New Admissions (per dayshift)", 0, 20, 3)
        consults = st.number_input("Floor Consults (per dayshift)", 0, 20, 4)
        transfers = st.number_input("Transfer Center Calls (per dayshift)", 0, 20, 2)

        section_header("Critical Events", "Enter frequency of critical events")
        critical_events = st.number_input("Critical Events (per week)", 0, 50, 5)

    # Calculate metrics
    interrupts_per_provider, time_lost = calculate_interruptions(
        nursing_q, exam_callbacks, peer_interrupts, providers
    )

    workload = calculate_workload(
        admissions, consults, transfers, 
        critical_events/7, providers
    )

    # Convert weekly critical events to daily average
    critical_events_per_day = critical_events / 7.0

    # Calculate time impacts
    interrupt_time, admission_time, critical_time = simulator.calculate_time_impact(
        nursing_q, exam_callbacks, peer_interrupts,
        admissions, consults, transfers, critical_events_per_day
    )

    efficiency = simulator.simulate_provider_efficiency(
        nursing_q + exam_callbacks + peer_interrupts,
        providers,
        workload
    )

    burnout_risk = simulator.calculate_burnout_risk(
        workload,
        interrupts_per_provider,
        critical_events_per_day
    )

    # Update the cognitive load calculation to use simulator settings
    cognitive_load = simulator.calculate_cognitive_load(
        interrupts_per_provider,
        critical_events_per_day,
        admissions,
        workload
    )

    # Display metrics
    st.markdown("### Key Metrics")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "Interruptions per Provider",
            f"{interrupts_per_provider:.1f}/shift",
            help=f"Average interruption duration: {sum(simulator.interruption_times.values())/len(simulator.interruption_times):.1f} min"
        )
    with metric_col2:
        st.metric("Hours Lost to Interruptions", f"{time_lost:.1f}")
    with metric_col3:
        st.metric("Provider Efficiency", f"{efficiency:.1%}")
    with metric_col4:
        st.metric(
            "Cognitive Load",
            f"{cognitive_load:.0f}/100",
            help="Based on interruptions, critical events, and workload"
        )

    # Time impact breakdown with tooltips
    st.markdown("### Time Impact Analysis (minutes per shift)")
    impact_col1, impact_col2, impact_col3 = st.columns(3)

    with impact_col1:
        st.metric(
            "Interruption Time",
            f"{interrupt_time:.0f}",
            help="Total minutes spent handling interruptions per shift"
        )
    with impact_col2:
        st.metric(
            "Admission/Transfer Time",
            f"{admission_time:.0f}",
            help=f"Simple admission: {simulator.admission_times['simple']} min\nComplex admission: {simulator.admission_times['complex']} min"
        )
    with impact_col3:
        st.metric(
            "Critical Event Time",
            f"{critical_time:.0f}",
            help=f"Based on {simulator.critical_event_time} minutes per critical event"
        )

    # Update the visualization section to properly reflect critical events impact
    # Visualizations section
    st.markdown("### Workflow Analysis")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.plotly_chart(
            create_interruption_chart(
                nursing_q, exam_callbacks, peer_interrupts, simulator
            ),
            use_container_width=True
        )

    with viz_col2:
        st.plotly_chart(
            create_time_allocation_pie(time_lost),
            use_container_width=True
        )

    st.plotly_chart(
        create_workload_timeline(
            workload, providers, critical_events_per_day, simulator
        ),
        use_container_width=True
    )

    # Add a detailed breakdown of time impacts
    st.markdown("### Time Impact Details")
    st.markdown("""
        <style>
        .impact-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .impact-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    total_interrupts = nursing_q + exam_callbacks + peer_interrupts
    st.markdown(
        f"""
        <div class="impact-grid">
            <div class="impact-card">
                <h4>Interruption Impact</h4>
                <p>Total time: {interrupt_time:.0f} min/shift</p>
                <p>Avg duration: {(interrupt_time/max(1, total_interrupts)):.1f} min/interruption</p>
            </div>
            <div class="impact-card">
                <h4>Critical Events Impact</h4>
                <p>Time per event: {simulator.critical_event_time} min</p>
                <p>Daily impact: {critical_time:.0f} min/shift</p>
            </div>
            <div class="impact-card">
                <h4>Admission/Transfer Load</h4>
                <p>Total time: {admission_time:.0f} min/shift</p>
                <p>Per provider: {(admission_time/providers):.0f} min/provider</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Recommendations
    st.markdown("### Recommendations")
    if burnout_risk > 0.7:
        st.warning("⚠️ High burnout risk detected. Consider increasing provider coverage or implementing interruption reduction strategies.")
    if cognitive_load > 80:
        st.warning("⚠️ High cognitive load detected. Consider workflow optimization or additional support staff.")
    if efficiency < 0.7:
        st.warning("⚠️ Low efficiency detected. Review interruption patterns and implement protected time for critical tasks.")

    total_time = interrupt_time + admission_time + critical_time
    if total_time > 720:  # 12 hours in minutes
        st.error("⚠️ Total task time exceeds shift duration. Current workload may not be sustainable.")

if __name__ == "__main__":
    main()