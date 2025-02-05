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
        st.markdown("""
            This interactive tool helps analyze and visualize ICU dayshift workflow dynamics (8 AM - 8 PM),
            considering various factors that impact provider efficiency and patient care.

            **Note:** All calculations are based on a 12-hour shift duration.
        """)

        simulator = WorkflowSimulator()

        # Add configuration section with an expander
        with st.expander("⚙️ Time Settings Configuration"):
            st.markdown("### Configure Time Estimates")
            st.markdown("Adjust the time estimates for various activities below:")

            col1, col2 = st.columns(2)

            with col1:
                section_header("Interruption Times", "Duration of different types of interruptions")
                nursing_time = max(1, st.number_input("Nursing Question Duration (minutes)", 1, 10, 2))
                exam_callback_time = max(1, st.number_input("Exam Callback Duration (minutes)", 1, 20, 8))
                peer_interrupt_time = max(1, st.number_input("Peer Interruption Duration (minutes)", 1, 20, 8))

            with col2:
                section_header("Admission & Critical Event Times", "Duration of patient care activities")
                simple_admission_time = max(30, st.number_input("Simple Admission Duration (minutes)", 30, 120, 60))
                complex_admission_time = max(60, st.number_input("Complex Admission Duration (minutes)", 60, 180, 90))
                critical_event_time = max(60, st.number_input("Critical Event Duration (minutes)", 60, 180, 105))

            # Rest of the time settings remain unchanged
            simulator.update_time_settings({
                'interruption_times': {
                    'nursing_question': nursing_time,
                    'exam_callback': exam_callback_time,
                    'peer_interrupt': peer_interrupt_time
                },
                'admission_times': {
                    'simple': simple_admission_time,
                    'complex': complex_admission_time,
                    'consult': 45,
                    'transfer': 30
                },
                'critical_event_time': critical_event_time
            })

        # Create two columns for inputs
        col1, col2 = st.columns(2)

        with col1:
            section_header("Interruptions", "Enter frequency of different types of interruptions during dayshift")
            nursing_q = max(0.0, st.number_input("Nursing Questions (per hour)", 0.0, 20.0, 5.0, 0.5))
            exam_callbacks = max(0.0, st.number_input("Exam Callbacks (per hour)", 0.0, 20.0, 3.0, 0.5))
            peer_interrupts = max(0.0, st.number_input("Peer Interruptions (per hour)", 0.0, 20.0, 2.0, 0.5))

            section_header("Provider Information", "Enter dayshift staffing details")
            providers = max(1, st.number_input("Number of Providers", 1, 10, 2))

        with col2:
            section_header("Admissions & Transfers", "Enter patient flow information for dayshift")
            admissions = max(0, st.number_input("New Admissions (per dayshift)", 0, 20, 3))
            consults = max(0, st.number_input("Floor Consults (per dayshift)", 0, 20, 4))
            transfers = max(0, st.number_input("Transfer Center Calls (per dayshift)", 0, 20, 2))

            section_header("Critical Events", "Enter frequency of critical events")
            critical_events = max(0, st.number_input("Critical Events (per week)", 0, 50, 5))

        # Calculate metrics
        interrupts_per_provider, time_lost = calculate_interruptions(
            nursing_q, exam_callbacks, peer_interrupts, providers
        )

        workload = calculate_workload(
            admissions, consults, transfers,
            critical_events/7, providers, simulator
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
            workload,
            critical_events_per_day
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
                use_container_width=True,
                key="interruption_chart"
            )

        with viz_col2:
            st.plotly_chart(
                create_time_allocation_pie(time_lost),
                use_container_width=True,
                key="time_allocation_chart"
            )

        st.plotly_chart(
            create_workload_timeline(
                workload, providers, critical_events_per_day, simulator
            ),
            use_container_width=True,
            key="workload_timeline_chart"
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

        # Calculate detailed burnout risk
        detailed_burnout = simulator.calculate_detailed_burnout_risk(
            workload,
            interrupts_per_provider,
            critical_events_per_day,
            efficiency,
            cognitive_load
        )

        # After the existing metrics section, add:
        st.markdown("### Detailed Burnout Risk Analysis")

        # Create three columns for the burnout visualizations
        brn_col1, brn_col2 = st.columns(2)

        with brn_col1:
            # Display burnout gauge
            st.plotly_chart(
                create_burnout_gauge(
                    detailed_burnout['total_risk'],
                    simulator.burnout_thresholds
                ),
                use_container_width=True,
                key="burnout_gauge_chart"
            )

            # Display risk category and score
            st.info(f"""
                **Risk Category:** {detailed_burnout['risk_category'].upper()}  
                **Risk Score:** {detailed_burnout['total_risk']*100:.1f}%
            """)

        with brn_col2:
            # Display radar chart of risk components
            st.plotly_chart(
                create_burnout_radar_chart(detailed_burnout['risk_components']),
                use_container_width=True,
                key="burnout_radar_chart"
            )

        # Display burnout trend chart
        st.plotly_chart(
            create_burnout_trend_chart(
                {
                    'total_risk': detailed_burnout['total_risk'],
                    'thresholds': simulator.burnout_thresholds
                }
            ),
            use_container_width=True,
            key="burnout_trend_chart"
        )

        # Display detailed recommendations
        st.markdown("### Detailed Recommendations")
        recommendations = format_burnout_recommendations({
            'risk_category': detailed_burnout['risk_category'],
            'risk_components': detailed_burnout['risk_components']
        })

        for rec in recommendations:
            st.markdown(rec)

        # Component breakdown
        st.markdown("### Risk Component Analysis")
        component_data = pd.DataFrame({
            'Component': list(detailed_burnout['risk_components'].keys()),
            'Risk Level': list(detailed_burnout['risk_components'].values()),
            'Weight': [detailed_burnout['component_weights'][k] for k in detailed_burnout['risk_components'].keys()]
        })

        st.dataframe(
            component_data.style.format({
                'Risk Level': '{:.1%}',
                'Weight': '{:.1%}'
            }),
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

        total_time = interrupt_time + admission_time + critical_time
        if total_time > 720:  # 12 hours in minutes
            st.error("⚠️ Total task time exceeds shift duration. Current workload may not be sustainable.")

        # ML Predictions section with error handling
        st.markdown("### Machine Learning Predictions")
        st.markdown("""
            This section uses machine learning to predict future workload and burnout risks
            based on current patterns and historical data.
        """)

        # Prepare current features for prediction with validation
        try:
            current_features = np.array([
                max(0, nursing_q), max(0, exam_callbacks), max(0, peer_interrupts),
                max(1, providers), max(0, admissions), max(0, consults), max(0, transfers),
                max(0, critical_events)
            ])

            # Train model if not trained
            if not st.session_state.model_trained:
                with st.spinner("Training prediction models..."):
                    try:
                        training_scores = st.session_state.predictor.train_initial_model(current_features)
                        st.session_state.model_trained = True
                    except Exception as e:
                        st.error(f"Error training the model: {str(e)}")
                        st.session_state.model_trained = False
                        return

            # Make predictions with error handling
            try:
                predictions = st.session_state.predictor.predict(current_features.reshape(1, -1))
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                return

            # Display current predictions
            pred_col1, pred_col2 = st.columns(2)

            with pred_col1:
                st.metric(
                    "Predicted Workload Risk",
                    f"{predictions['predicted_workload']:.1%}",
                    help="ML-based prediction of workload risk based on current patterns"
                )

            with pred_col2:
                st.metric(
                    "Predicted Burnout Risk",
                    f"{predictions['predicted_burnout']:.1%}",
                    help="ML-based prediction of burnout risk based on current patterns"
                )

            # Get and display trend predictions
            trend_predictions = st.session_state.predictor.predict_next_week(current_features)
            st.plotly_chart(
                create_prediction_trend_chart(trend_predictions),
                use_container_width=True,
                key="prediction_trend_chart"
            )

            # Display feature importance analysis
            st.markdown("### Feature Importance Analysis")
            importance_col1, importance_col2 = st.columns(2)

            with importance_col1:
                st.markdown("#### Workload Factors")
                st.plotly_chart(
                    create_feature_importance_chart(predictions['workload_importance']),
                    use_container_width=True,
                    key="workload_importance_chart"
                )

            with importance_col2:
                st.markdown("#### Burnout Factors")
                st.plotly_chart(
                    create_feature_importance_chart(predictions['burnout_importance']),
                    use_container_width=True,
                    key="burnout_importance_chart"
                )

            # Add model explanation
            with st.expander("About the Prediction Model"):
                st.markdown("""
                    The machine learning model uses a Random Forest algorithm to predict workload and burnout risks.
                    It considers:
                    - Current interruption patterns
                    - Staffing levels
                    - Patient flow metrics
                    - Critical event frequency

                    The model is trained on synthetic data generated from current patterns and domain knowledge.
                    Predictions are updated in real-time as you adjust the input parameters.
                """)

            # Add Export Section after recommendations
            st.markdown("### Export Report")
            st.markdown("Download the analysis in your preferred format:")

            report_data = generate_report_data(
                interrupts_per_provider, time_lost, efficiency,
                cognitive_load, workload, burnout_risk,
                interrupt_time, admission_time, critical_time,
                providers
            )

            # Add recommendations to report data
            report_data["recommendations"] = format_recommendations(
                efficiency, cognitive_load, burnout_risk,
                interrupt_time + admission_time + critical_time
            )

            # Prepare different export formats
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

            # CSV Export
            df = pd.DataFrame({
                "Metric": list(report_data["metrics"].keys()) + list(report_data["time_analysis"].keys()),
                "Value": list(report_data["metrics"].values()) + list(report_data["time_analysis"].values())
            })
            csv = df.to_csv(index=False).encode('utf-8')

            # JSON Export
            json_str = json.dumps(report_data, indent=2).encode('utf-8')

            # Create download buttons
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f'icu_workflow_analysis_{current_time}.csv',
                    mime='text/csv'
                )

            with col2:
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f'icu_workflow_analysis_{current_time}.json',
                    mime='application/json'
                )

            # Display export preview
            with st.expander("Preview Export Data"):
                st.dataframe(df, use_container_width=True)

                st.markdown("### Recommendations")
                for rec in report_data["recommendations"]:
                    st.markdown(f"- {rec}")

        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")
            return

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return

if __name__ == "__main__":
    main()