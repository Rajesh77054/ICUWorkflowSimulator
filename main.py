import streamlit as st
import numpy as np
import pandas as pd
import json
from datetime import datetime
from styles import apply_custom_styles, section_header
from utils import (calculate_interruptions, calculate_workload,
                  create_interruption_chart, create_time_allocation_pie,
                  create_workload_timeline, generate_report_data,
                  format_recommendations, create_burnout_gauge,
                  create_burnout_radar_chart, create_burnout_trend_chart,
                  format_burnout_recommendations, create_prediction_trend_chart,
                  create_feature_importance_chart)
from simulator import WorkflowSimulator
from ml_predictor import WorkflowPredictor
from models import get_db, save_workflow_record, get_historical_records

# Initialize predictor in session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = WorkflowPredictor()
    st.session_state.model_trained = False

def main():
    try:
        st.set_page_config(
            page_title="ICU Workflow Dynamics Model",
            page_icon="🏥",
            layout="wide"
        )

        apply_custom_styles()

        st.title("ICU Workflow Dynamics Model")

        # User Type Selection
        user_type = st.radio(
            "Select Your Role",
            ["Provider", "Administrator"],
            horizontal=True,
            help="Choose your role to see relevant metrics and insights"
        )

        simulator = WorkflowSimulator()

        # Common Configuration Section (collapsible)
        with st.expander("⚙️ Workflow Configuration"):
            col1, col2 = st.columns(2)

            with col1:
                nursing_q = max(0.0, st.number_input("Nursing Questions (per hour)", 0.0, 20.0, 5.0, 0.5))
                exam_callbacks = max(0.0, st.number_input("Exam Callbacks (per hour)", 0.0, 20.0, 3.0, 0.5))
                peer_interrupts = max(0.0, st.number_input("Peer Interruptions (per hour)", 0.0, 20.0, 2.0, 0.5))
                providers = max(1, st.number_input("Number of Providers", 1, 10, 2))

            with col2:
                admissions = max(0, st.number_input("New Admissions (per dayshift)", 0, 20, 3))
                consults = max(0, st.number_input("Floor Consults (per dayshift)", 0, 20, 4))
                transfers = max(0, st.number_input("Transfer Center Calls (per dayshift)", 0, 20, 2))
                critical_events = max(0, st.number_input("Critical Events (per week)", 0, 50, 5))

        # Calculate core metrics
        interrupts_per_provider, time_lost = calculate_interruptions(
            nursing_q, exam_callbacks, peer_interrupts, providers
        )

        workload = calculate_workload(
            admissions, consults, transfers,
            critical_events/7, providers, simulator
        )

        critical_events_per_day = critical_events / 7.0

        interrupt_time, admission_time, critical_time = simulator.calculate_time_impact(
            nursing_q, exam_callbacks, peer_interrupts,
            admissions, consults, transfers, critical_events_per_day,
            providers
        )


        efficiency = simulator.simulate_provider_efficiency(
            nursing_q + exam_callbacks + peer_interrupts,
            providers, workload, critical_events_per_day
        )

        burnout_risk = simulator.calculate_burnout_risk(
            workload,
            interrupts_per_provider,
            critical_events_per_day
        )

        cognitive_load = simulator.calculate_cognitive_load(
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
                    workload, providers, critical_events_per_day, simulator
                ),
                use_container_width=True
            )

            # Time Distribution
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_interruption_chart(
                        nursing_q, exam_callbacks, peer_interrupts, simulator
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
                        simulator.burnout_thresholds
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
        st.error(f"An unexpected error occurred: {str(e)}")
        return

if __name__ == "__main__":
    main()