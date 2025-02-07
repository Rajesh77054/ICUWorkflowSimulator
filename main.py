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
# Added imports
from models import get_db, save_workflow_record, get_historical_records
from sqlalchemy.orm import Session
from io import BytesIO

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
            providers  # Add providers parameter
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

        cognitive_load = simulator.calculate_cognitive_load(
            interrupts_per_provider,
            critical_events_per_day,
            admissions,
            workload
        )

        st.markdown("### Core Workflow Metrics")
        st.markdown("Direct measurements of workflow activities and time allocation:")

        primary_col1, primary_col2, primary_col3, primary_col4 = st.columns(4)

        with primary_col1:
            st.metric(
                "Interruptions per Provider",
                f"{interrupts_per_provider:.1f}/shift",
                help="""
                Direct measure of workflow disruptions:
                • Total interruptions / Number of providers
                • Normal range: 20-40 per shift
                • Warning level: >50 per shift
                • Critical level: >70 per shift
                """
            )
        with primary_col2:
            st.metric(
                "Time Allocation (Interruptions)",
                f"{interrupt_time:.0f} min",
                help="""
                Total provider time on interruptions:
                • Per shift across all providers
                • Based on actual frequency × duration
                • <120 min: Manageable
                • >180 min: Significant impact
                """
            )
        with primary_col3:
            st.metric(
                "Admission/Transfer Time",
                f"{admission_time:.0f} min",
                help=f"""
                Time for patient movement activities:
                • Simple admission: {simulator.admission_times['simple']} min
                • Complex admission: {simulator.admission_times['complex']} min
                • Consult: {simulator.admission_times['consult']} min
                • Transfer: {simulator.admission_times['transfer']} min
                """
            )
        with primary_col4:
            st.metric(
                "Critical Event Time",
                f"{critical_time:.0f} min",
                help=f"""
                Time managing critical events:
                • Average duration: {simulator.critical_event_time} min
                • <60 min: Regular staffing adequate
                • 60-120 min: Consider backup
                • >120 min: Additional provider needed
                """
            )

        st.markdown("### Impact Analysis")
        st.markdown("Calculated metrics showing overall workflow impact:")
        impact_col1, impact_col2 = st.columns(2)

        with impact_col1:
            total_time = interrupt_time + admission_time + critical_time
            st.metric(
                "Total Time Impact",
                f"{total_time:.0f} min",
                help="""
                Total time allocation across activities:
                • Sum of all measured time components
                • Warning if exceeds shift duration (720 min)
                • Key indicator of workload sustainability
                """
            )
        with impact_col2:
            time_per_provider = total_time / providers
            st.metric(
                "Average Time per Provider",
                f"{time_per_provider:.0f} min",
                help="""
                Individual provider time allocation:
                • Total time divided by provider count
                • Indicates individual workload level
                • Warning if exceeds 360 min per shift
                """
            )


        # Prepare current features for prediction with validation
        try:
            current_features = np.array([
                max(0, nursing_q), max(0, exam_callbacks), max(0, peer_interrupts),
                max(1, providers), max(0, admissions), max(0, consults), max(0, transfers),
                max(0, critical_events)
            ])

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

            # Calculate detailed burnout risk
            detailed_burnout = simulator.calculate_detailed_burnout_risk(
                workload,
                interrupts_per_provider,
                critical_events_per_day,
                efficiency,
                cognitive_load
            )

            # Save to database
            db = next(get_db())
            metrics = {
                'interrupts_per_provider': interrupts_per_provider,
                'time_lost': time_lost,
                'efficiency': efficiency,
                'cognitive_load': cognitive_load,
                'burnout_risk': burnout_risk,
                'interrupt_time': interrupt_time,
                'admission_time': admission_time,
                'critical_time': critical_time,
                'recommendations': format_recommendations(
                    efficiency, cognitive_load, burnout_risk,
                    interrupt_time + admission_time + critical_time
                )
            }

            predictions['risk_components'] = detailed_burnout['risk_components']

            save_workflow_record(
                db,
                nursing_q, exam_callbacks, peer_interrupts,
                providers, admissions, consults, transfers,
                critical_events, metrics, predictions
            )

        except Exception as e:
            st.error(f"An error occurred while preparing features or saving data: {str(e)}")
            return


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
                create_time_allocation_pie(
                    time_lost,
                    providers=providers
                ),
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

        total_interrupts = (nursing_q + exam_callbacks + peer_interrupts) * 12  # Total interruptions per shift
        st.markdown(
            f"""
            <div class="impact-grid">
                <div class="impact-card">
                    <h4>Interruption Impact</h4>
                    <p>Organization time: {interrupt_time:.0f} min/shift</p>
                    <p>Per provider: {(interrupt_time/providers):.0f} min/provider</p>
                    <p>Avg duration: {(interrupt_time/max(1, total_interrupts)):.1f} min/interruption</p>
                </div>
                <div class="impact-card">
                    <h4>Critical Events Impact</h4>
                    <p>Time per event: {simulator.critical_event_time} min</p>
                    <p>Organization impact: {critical_time:.0f} min/shift</p>
                    <p>Per provider: {(critical_time/providers):.0f} min/provider</p>
                </div>
                <div class="impact-card">
                    <h4>Admission/Transfer Load</h4>
                    <p>Organization time: {admission_time:.0f} min/shift</p>
                    <p>Per provider: {(admission_time/providers):.0f} min/provider</p>
                    <p><i>Note: Times reflect parallel work across providers</i></p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        detailed_burnout = simulator.calculate_detailed_burnout_risk(
            workload,
            interrupts_per_provider,
            critical_events_per_day,
            efficiency,
            cognitive_load
        )

        st.markdown("### Detailed Burnout Risk Analysis")

        brn_col1, brn_col2 = st.columns(2)

        with brn_col1:
            st.plotly_chart(
                create_burnout_gauge(
                    detailed_burnout['total_risk'],
                    simulator.burnout_thresholds
                ),
                use_container_width=True,
                key="burnout_gauge_chart"
            )

            st.info(f"""
                **Risk Category:** {detailed_burnout['risk_category'].upper()}  
                **Risk Score:** {detailed_burnout['total_risk']*100:.1f}%
            """)

        with brn_col2:
            st.plotly_chart(
                create_burnout_radar_chart(detailed_burnout['risk_components']),
                use_container_width=True,
                key="burnout_radar_chart"
            )

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

        st.markdown("### Detailed Recommendations")
        recommendations = format_burnout_recommendations({
            'risk_category': detailed_burnout['risk_category'],
            'risk_components': detailed_burnout['risk_components']
        })

        for rec in recommendations:
            st.markdown(rec)

        st.markdown("### Risk Component Analysis")
        component_data = pd.DataFrame({
            'Component': list(detailed_burnout['risk_components'].keys()),
            'Risk Level': list(detailed_burnout['risk_components'].values()),
            'Weight': [detailed_burnout['component_weights'][k] for k in detailed_burnout['risk_components'].keys()]
        })

        # Add tooltips to the dataframe columns
        st.markdown("""
        <style>
        [data-testid="stMetricLabel"] {
            cursor: help;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        ℹ️ **Understanding Risk Components**

        This analysis breaks down various factors contributing to provider stress and burnout:

        • **Interruption Risk**: Impact of frequency and duration of workflow interruptions
        • **Workload Risk**: Assessment of patient care and administrative duties
        • **Critical Events Risk**: Stress from managing high-acuity situations
        • **Efficiency Risk**: Impact of workflow disruptions on productivity
        • **Cognitive Load Risk**: Mental burden from multiple simultaneous responsibilities

        Each component is weighted based on its relative impact on provider wellbeing.
        """)

        st.dataframe(
            component_data.style.format({
                'Risk Level': '{:.1%}',
                'Weight': '{:.1%}'
            }),
            use_container_width=True
        )

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

        st.markdown("### Historical Analysis")
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

            st.markdown("#### Trend Analysis")
            trends_col1, trends_col2 = st.columns(2)

            with trends_col1:
                avg_efficiency = hist_df['Efficiency'].mean()
                current_efficiency = hist_df['Efficiency'].iloc[0]
                efficiency_delta = current_efficiency - avg_efficiency

                st.metric(
                    "Efficiency Trend",
                    f"{current_efficiency:.1%}",
                    f"{efficiency_delta:+.1%}",
                    help="Comparison with historical average"
                )

            with trends_col2:
                avg_burnout = hist_df['Burnout Risk'].mean()
                current_burnout = hist_df['Burnout Risk'].iloc[0]
                burnout_delta = current_burnout - avg_burnout

                st.metric(
                    "Burnout Risk Trend",
                    f"{current_burnout:.1%}",
                    f"{burnout_delta:+.1%}",
                    help="Comparison with historical average"
                )
        else:
            st.info("No historical data available yet. Data will be collected as you use the application.")


        st.markdown("### Machine Learning Predictions")
        st.markdown("""
            This section uses machine learning to predict future workload and burnout risks
            based on current patterns and historical data.
        """)

        try:
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

            trend_predictions = st.session_state.predictor.predict_next_week(current_features)
            st.plotly_chart(
                create_prediction_trend_chart(trend_predictions),
                use_container_width=True,
                key="prediction_trend_chart"
            )

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

            st.markdown("### Reports")
            report_type = st.selectbox(
                "Select Report Type",
                ["Current Session Report", "Historical Analysis Report"],
                help="Choose the type of report you want to generate"
            )

            if report_type == "Current Session Report":
                st.markdown("#### Export Current Session Report")
                st.markdown("Download the current session analysis in your preferred format:")

                report_data = generate_report_data(
                    interrupts_per_provider, time_lost, efficiency,
                    cognitive_load, workload, burnout_risk,
                    interrupt_time, admission_time, critical_time,
                    providers
                )

                report_data["recommendations"] = format_recommendations(
                    efficiency, cognitive_load, burnout_risk,
                    interrupt_time + admission_time + critical_time
                )

                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

                df = pd.DataFrame({
                    "Metric": list(report_data["metrics"].keys()) + list(report_data["time_analysis"].keys()),
                    "Value": list(report_data["metrics"].values()) + list(report_data["time_analysis"].values())
                })
                csv = df.to_csv(index=False).encode('utf-8')
                json_str = json.dumps(report_data, indent=2).encode('utf-8')

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

                with st.expander("Preview Export Data"):
                    st.dataframe(df, use_container_width=True)
                    st.markdown("#### Recommendations")
                    for rec in report_data["recommendations"]:
                        st.markdown(f"- {rec}")

            else:  # Historical Analysis Report
                st.markdown("#### Historical Analysis Report")
                st.markdown("""
                    Select date range and metrics to create a custom historical analysis report.
                    You can visualize the data before exporting.
                """)

                # Date range selection
                date_col1, date_col2 = st.columns(2)
                with date_col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=pd.Timestamp.now().date() - pd.Timedelta(days=30),
                        help="Select start date for historical data"
                    )
                with date_col2:
                    end_date = st.date_input(
                        "End Date",
                        value=pd.Timestamp.now().date(),
                        help="Select end date for historical data"
                    )

                # Metric selection
                available_metrics = [
                    "Efficiency", "Cognitive Load", "Burnout Risk",
                    "Interruptions/Provider", "Time Lost",
                    "Admission Time", "Critical Time"
                ]
                selected_metrics = st.multiselect(
                    "Select Metrics to Include",
                    available_metrics,
                    default=["Efficiency", "Cognitive Load", "Burnout Risk"],
                    help="Choose which metrics to include in the export"
                )

                if st.button("Generate Report"):
                    # Query historical data with date filter
                    historical_records = get_historical_records(db)
                    if historical_records:
                        hist_df = pd.DataFrame([{
                            'Timestamp': record.timestamp,
                            'Efficiency': record.efficiency,
                            'Cognitive Load': record.cognitive_load,
                            'Burnout Risk': record.burnout_risk,
                            'Interruptions/Provider': record.interrupts_per_provider,
                            'Time Lost': record.time_lost,
                            'Admission Time': record.admission_time,
                            'Critical Time': record.critical_time
                        } for record in historical_records])

                        # Apply date filter
                        mask = (hist_df['Timestamp'].dt.date >= start_date) & (hist_df['Timestamp'].dt.date <= end_date)
                        filtered_df = hist_df.loc[mask]

                        if len(filtered_df) > 0:
                            st.markdown("#### Data Preview")
                            st.dataframe(
                                filtered_df[['Timestamp'] + selected_metrics],
                                use_container_width=True
                            )

                            # Visualization of selected metrics
                            st.markdown("#### Trend Analysis")
                            st.line_chart(
                                filtered_df.set_index('Timestamp')[selected_metrics]
                            )

                            # Statistical summary
                            st.markdown("#### Statistical Summary")
                            st.dataframe(
                                filtered_df[selected_metrics].describe(),
                                use_container_width=True
                            )

                            # Export options
                            st.markdown("#### Export Options")
                            export_col1, export_col2, export_col3 = st.columns(3)

                            # Prepare export data
                            export_df = filtered_df[['Timestamp'] + selected_metrics].copy()
                            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

                            with export_col1:
                                csv = export_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name=f'historical_analysis_{current_time}.csv',
                                    mime='text/csv'
                                )

                            with export_col2:
                                json_str = export_df.to_json(orient='records', date_format='iso').encode('utf-8')
                                st.download_button(
                                    label="Download JSON",
                                    data=json_str,
                                    file_name=f'historical_analysis_{current_time}.json',
                                    mime='application/json'
                                )

                            with export_col3:
                                # Excel export with formatting
                                excel_buffer = BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                    export_df.to_excel(writer, sheet_name='Historical Data', index=False)
                                    workbook = writer.book
                                    worksheet = writer.sheets['Historical Data']

                                    # Add formatting
                                    header_format = workbook.add_format({
                                        'bold': True,
                                        'fg_color': '#D7E4BC',
                                        'border': 1
                                    })

                                    for col_num, value in enumerate(export_df.columns.values):
                                        worksheet.write(0, col_num, value, header_format)
                                        worksheet.set_column(col_num, col_num, 15)

                                excel_buffer.seek(0)
                                st.download_button(
                                    label="Download Excel",
                                    data=excel_buffer,
                                    file_name=f'historical_analysis_{current_time}.xlsx',
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                )
                        else:
                            st.warning("No data available for the selected date range.")
                    else:
                        st.info("No historical data available yet. Data will be collected as you use the application.")

        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")
            return

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return

if __name__ == "__main__":
    main()