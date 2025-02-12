import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from styles import apply_custom_styles, section_header
from utils import (calculate_interruptions, calculate_workload,
                  create_interruption_chart, create_time_allocation_pie,
                  create_workload_timeline, create_burnout_gauge,
                  create_burnout_radar_chart, create_prediction_trend_chart,
                  generate_report_data, format_recommendations)
from simulator import WorkflowSimulator
from models import get_db, save_workflow_record, get_historical_records
from ml_predictor import MLPredictor

def main():
    st.set_page_config(
        page_title="ICU Workflow Dynamics Model",
        page_icon="🏥",
        layout="wide"
    )

    apply_custom_styles()
    st.title("ICU Workflow Dynamics Model")

    # Initialize simulator in session state if not present
    if 'simulator' not in st.session_state:
        st.session_state.simulator = WorkflowSimulator()

    # Initialize ML predictor and training status
    if 'predictor' not in st.session_state:
        st.session_state.predictor = MLPredictor()
        st.session_state.model_trained = False

    # User Type Selection
    user_type = st.radio(
        "Select Your Role",
        ["Provider", "Administrator"],
        horizontal=True,
        help="Choose your role to see relevant metrics and insights"
    )

    # Workflow Configuration Section
    with st.expander("⚙️ Workflow Configuration", expanded=True):
        col1, col2 = st.columns(2)

        # Base Workload Components (Left Column)
        with col1:
            st.markdown("### Primary Workload")

            # ICU Census
            adc = st.number_input(
                "ICU Census (ADC)", 
                0, 16, 8, 1,
                help="Average Daily Census - Primary ICU workload driver"
            )
            st.caption("ADC determines base workload and scales interruption frequencies")

            # Floor Consults
            consults = st.number_input(
                "Floor Consults (per shift)", 
                0, 20, 4,
                help="Additional workload from non-ICU consultations"
            )

            # Providers
            providers = st.number_input(
                "Number of Providers", 
                1, 10, 2,
                help="Available providers working in parallel"
            )

        # Interruption Configuration (Right Column)
        with col2:
            st.markdown("### Interruption Factors")
            st.caption("*Note: Frequencies auto-scale with ICU Census*")

            # Nursing Questions Container
            st.markdown("#### Nursing Questions")
            nq_col1, nq_col2 = st.columns(2)
            with nq_col1:
                nursing_scale = st.number_input(
                    "Rate (per patient per hour)", 
                    0.0, 2.0,
                    value=st.session_state.simulator.interruption_scales['nursing_question'],
                    step=0.01, 
                    format="%.2f"
                )
                nursing_q = adc * nursing_scale
                st.metric("Current Rate", f"{nursing_q:.1f}/hour")
            with nq_col2:
                nursing_time = st.slider("Duration (minutes)", 1, 10, 2, key="nursing_duration")

            # Exam Callbacks Container
            st.markdown("#### Exam Callbacks")
            ec_col1, ec_col2 = st.columns(2)
            with ec_col1:
                callback_scale = st.number_input(
                    "Rate (per patient per hour)", 
                    0.0, 2.0,
                    value=st.session_state.simulator.interruption_scales['exam_callback'],
                    step=0.01, 
                    format="%.2f"
                )
                exam_callbacks = adc * callback_scale
                st.metric("Current Rate", f"{exam_callbacks:.1f}/hour")
            with ec_col2:
                callback_time = st.slider("Duration (minutes)", 1, 20, 8, key="callback_duration")

            # Peer Interruptions Container
            st.markdown("#### Peer Interruptions")
            pi_col1, pi_col2 = st.columns(2)
            with pi_col1:
                peer_scale = st.number_input(
                    "Rate (per patient per hour)", 
                    0.0, 2.0,
                    value=st.session_state.simulator.interruption_scales['peer_interrupt'],
                    step=0.01, 
                    format="%.2f"
                )
                peer_interrupts = adc * peer_scale
                st.metric("Current Rate", f"{peer_interrupts:.1f}/hour")
            with pi_col2:
                peer_time = st.slider("Duration (minutes)", 1, 20, 8, key="peer_duration")

        # Critical Events Configuration (Bottom Section)
        st.markdown("### Critical Events")
        ce_col1, ce_col2, ce_col3 = st.columns(3)

        with ce_col1:
            admissions = st.number_input(
                "New Admissions (per shift)", 
                0, 20, 3,
                help="Expected new ICU admissions"
            )
            simple_admission_time = st.number_input(
                "Simple Admission Duration", 
                30, 120, 60,
                help="Minutes required for straightforward admissions"
            )

        with ce_col2:
            transfers = st.number_input(
                "Transfer Calls (per shift)", 
                0, 20, 2,
                help="Expected transfer requests"
            )
            complex_admission_time = st.number_input(
                "Complex Admission Duration", 
                45, 180, 90,
                help="Minutes required for complex admissions"
            )

        with ce_col3:
            critical_events = st.number_input(
                "Critical Events (per week)", 
                0, 50, 5,
                help="Expected critical events requiring immediate attention"
            )
            critical_event_time = st.number_input(
                "Critical Event Duration", 
                60, 180, 105,
                help="Average minutes per critical event"
            )

    try:
        # Update simulator settings
        st.session_state.simulator.update_time_settings({
            'interruption_times': {
                'nursing_question': nursing_time,
                'exam_callback': callback_time,
                'peer_interrupt': peer_time
            },
            'admission_times': {
                'simple': simple_admission_time,
                'complex': complex_admission_time,
                'transfer': 30  # Default transfer time
            },
            'critical_event_time': critical_event_time
        })

        # Calculate metrics
        interrupts_per_provider, time_lost = calculate_interruptions(
            nursing_q, exam_callbacks, peer_interrupts,
            providers, st.session_state.simulator
        )

        workload = calculate_workload(
            adc, admissions, consults, transfers,
            critical_events/7, providers, st.session_state.simulator
        )

        critical_events_per_day = critical_events / 7.0

        interrupt_time, admission_time, critical_time = st.session_state.simulator.calculate_time_impact(
            nursing_q, exam_callbacks, peer_interrupts,
            admissions, consults, transfers, critical_events_per_day,
            providers
        )

        efficiency = st.session_state.simulator.simulate_provider_efficiency(
            nursing_q + exam_callbacks + peer_interrupts,
            providers, workload, critical_events_per_day, admissions, adc
        )

        burnout_risk = st.session_state.simulator.calculate_burnout_risk(
            workload,
            interrupts_per_provider,
            critical_events_per_day
        )

        cognitive_load = st.session_state.simulator.calculate_cognitive_load(
            interrupts_per_provider,
            critical_events_per_day,
            admissions,
            workload
        )

        if user_type == "Provider":
            # Provider View
            st.markdown("### Current Shift Overview")

            # Core Workflow Metrics Section
            st.markdown(section_header("Core Workflow Metrics"), unsafe_allow_html=True)

            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric(
                    "Interruptions/Provider",
                    f"{interrupts_per_provider:.1f}/shift",
                    help="Direct measure of workflow disruptions per provider"
                )
            with metrics_cols[1]:
                st.metric(
                    "Time Lost to Interruptions",
                    f"{time_lost:.1f} min" if time_lost is not None else "0.0 min",
                    help="Total organizational time lost to interruptions"
                )
            with metrics_cols[2]:
                st.metric(
                    "Provider Efficiency",
                    f"{efficiency:.0%}",
                    help="Current workflow efficiency"
                )
            with metrics_cols[3]:
                st.metric(
                    "Cognitive Load",
                    f"{cognitive_load:.0f}%",
                    help="Mental workload based on current conditions"
                )

            # Visual Timeline
            st.plotly_chart(
                create_workload_timeline(
                    workload, providers, critical_events_per_day, admissions, st.session_state.simulator
                ),
                use_container_width=True
            )

            # Time Distribution
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_interruption_chart(
                        nursing_q, exam_callbacks, peer_interrupts, st.session_state.simulator
                    ),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    create_time_allocation_pie(time_lost, providers),
                    use_container_width=True
                )

            # Recommendations for Providers
            with st.expander("📋 Recommendations"):
                recommendations = format_recommendations(
                    efficiency, cognitive_load, burnout_risk,
                    time_lost
                )
                for rec in recommendations:
                    st.markdown(f"• {rec}")

        else:
            # Administrator View
            st.markdown("### Administrative Dashboard")

            # Historical Analysis
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_burnout_gauge(
                        burnout_risk,
                        st.session_state.simulator.burnout_thresholds
                    ),
                    use_container_width=True
                )

            with col2:
                st.plotly_chart(
                    create_burnout_radar_chart({
                        "Workload": workload,
                        "Interruptions": interrupts_per_provider/50,  # Normalized
                        "Critical Events": critical_events_per_day/5,  # Normalized
                        "Cognitive Load": cognitive_load/100,
                        "Efficiency Loss": 1-efficiency
                    }),
                    use_container_width=True
                )

            # Predictive Analytics
            st.markdown("### Predictive Insights")

            try:
                current_features = np.array([
                    nursing_q, exam_callbacks, peer_interrupts,
                    providers, admissions, consults, transfers,
                    critical_events
                ])

                if not st.session_state.model_trained:
                    with st.spinner("Training prediction models..."):
                        st.session_state.predictor.train_initial_model(current_features)
                        st.session_state.model_trained = True

                predictions = st.session_state.predictor.predict(current_features.reshape(1, -1))
                trend_predictions = st.session_state.predictor.predict_next_week(current_features)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Predicted Workload Risk",
                        f"{predictions['predicted_workload']:.1%}"
                    )
                with col2:
                    st.metric(
                        "Predicted Burnout Risk",
                        f"{predictions['predicted_burnout']:.1%}"
                    )

                st.plotly_chart(
                    create_prediction_trend_chart(trend_predictions),
                    use_container_width=True
                )

                # Historical Data Analysis
                st.markdown("### Historical Trends")
                db = next(get_db())
                historical_records = get_historical_records(db)

                if historical_records:
                    hist_df = pd.DataFrame([{
                        'Timestamp': record.timestamp,
                        'Efficiency': record.efficiency,
                        'Cognitive Load': record.cognitive_load,
                        'Burnout Risk': record.burnout_risk,
                        'Interruptions/Provider': record.interrupts_per_provider
                    } for record in historical_records])

                    st.line_chart(hist_df.set_index('Timestamp')[
                        ['Efficiency', 'Cognitive Load', 'Burnout Risk']
                    ])

            except Exception as e:
                st.error(f"Error in predictive analytics: {str(e)}")

            # Export Options
            with st.expander("📊 Export Reports"):
                report_type = st.selectbox(
                    "Select Report Type",
                    ["Current State Analysis", "Historical Trends", "Predictive Analysis"]
                )

                if st.button("Generate Report"):
                    report_data = generate_report_data(
                        interrupts_per_provider, time_lost, efficiency,
                        cognitive_load, workload, burnout_risk,
                        interrupt_time, admission_time, critical_time, providers
                    )

                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="Download Report (CSV)",
                        data=pd.DataFrame(report_data).to_csv().encode('utf-8'),
                        file_name=f'workflow_analysis_{current_time}.csv',
                        mime='text/csv'
                    )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()