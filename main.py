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
from scenario_manager import ScenarioManager
from models import save_scenario, save_scenario_result, get_scenarios, get_scenario_results
import plotly.graph_objects as go
from scenario_advisor import ScenarioAdvisor # Added import


def main():
    st.set_page_config(
        page_title="ICU Workflow Dynamics Model",
        page_icon="🏥",
        layout="wide"
    )

    apply_custom_styles()
    st.title("ICU Workflow Dynamics Model")

    # Initialize simulator and other components in session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = WorkflowSimulator()
        st.session_state.scenario_manager = ScenarioManager(st.session_state.simulator)

    if 'predictor' not in st.session_state:
        st.session_state.predictor = MLPredictor()
        st.session_state.model_trained = False

    if 'scenario_advisor' not in st.session_state: # Added initialization
        st.session_state.scenario_advisor = ScenarioAdvisor()

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
            st.markdown("### Floor Consults (Physician Only)")
            st.caption("Note: Only the physician responds to floor consults (8 AM - 5 PM)")

            consult_col1, consult_col2 = st.columns(2)
            with consult_col1:
                consults = st.number_input(
                    "Floor Consults (per shift)",
                    0, 20, 4,
                    help="Number of floor consults per dayshift (distributed 8 AM - 5 PM)"
                )
                st.caption(f"Average: {(consults/9):.1f} consults per hour")

            with consult_col2:
                consult_duration = st.number_input(
                    "Consult Duration (minutes)",
                    30, 90,
                    value=st.session_state.simulator.admission_times['consult'],
                    help="Average duration of each floor consult"
                )
                total_consult_time = consults * consult_duration
                st.metric(
                    "Total Consult Time",
                    f"{total_consult_time} min",
                    help="Total time physician spends on floor consults"
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

            # Transfer Calls Container (New)
            st.markdown("#### Transfer Calls")
            tc_col1, tc_col2 = st.columns(2)
            with tc_col1:
                transfer_scale = st.number_input(
                    "Rate (per patient per hour)",
                    0.0, 2.0,
                    value=st.session_state.simulator.interruption_scales.get('transfer_call', 0.1),
                    step=0.01,
                    format="%.2f"
                )
                transfer_calls = adc * transfer_scale
                st.metric("Current Rate", f"{transfer_calls:.1f}/hour")
            with tc_col2:
                transfer_time = st.slider("Duration (minutes)", 1, 20, 8, key="transfer_duration")

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
            critical_events = st.number_input(
                "Critical Events (per week)",
                0, 50, 5,
                help="Expected critical events requiring immediate attention"
            )
            complex_admission_time = st.number_input(
                "Complex Admission Duration",
                45, 180, 90,
                help="Minutes required for complex admissions"
            )

        with ce_col3:
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
                'peer_interrupt': peer_time,
                'transfer_call': transfer_time
            },
            'admission_times': {
                'consult': consult_duration,
                'simple': simple_admission_time,
                'complex': complex_admission_time,
                'transfer': 30  # Default transfer time
            },
            'critical_event_time': critical_event_time
        })

        # Calculate metrics
        interrupts_per_provider, time_lost = calculate_interruptions(
            nursing_q, exam_callbacks, peer_interrupts, transfer_calls,
            providers, st.session_state.simulator
        )

        workload = calculate_workload(
            adc, consults, providers, st.session_state.simulator
        )

        critical_events_per_day = critical_events / 7.0

        interrupt_time, admission_time, critical_time = st.session_state.simulator.calculate_time_impact(
            nursing_q, exam_callbacks, peer_interrupts, transfer_calls,
            admissions, consults, critical_events_per_day, providers
        )

        efficiency = st.session_state.simulator.simulate_provider_efficiency(
            nursing_q + exam_callbacks + peer_interrupts + transfer_calls,
            providers, workload['combined'], critical_events_per_day, admissions, adc
        )

        burnout_risk = st.session_state.simulator.calculate_burnout_risk(
            workload['combined'],
            interrupts_per_provider,
            critical_events_per_day
        )

        cognitive_load = st.session_state.simulator.calculate_cognitive_load(
            interrupts_per_provider,
            critical_events_per_day,
            admissions,
            workload['combined']
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
                    workload['combined'], providers, critical_events_per_day, admissions, st.session_state.simulator
                ),
                use_container_width=True
            )

            # Time Distribution
            st.markdown("### Provider Time Allocation")
            st.caption("Showing separate allocations for Physician and APP roles")

            col1, col2 = st.columns(2)
            with col1:
                # Physician time allocation
                st.plotly_chart(
                    create_time_allocation_pie(
                        time_lost,
                        total_consult_time,
                        providers,
                        role='physician'
                    ),
                    use_container_width=True
                )
            with col2:
                # APP time allocation
                st.plotly_chart(
                    create_time_allocation_pie(
                        time_lost,
                        total_consult_time,
                        providers,
                        role='app'
                    ),
                    use_container_width=True
                )

            # Calculate role-specific metrics
            physician_efficiency = st.session_state.simulator.simulate_provider_efficiency(
                nursing_q + exam_callbacks + peer_interrupts + transfer_calls,
                providers, workload['physician'], critical_events_per_day,
                admissions, adc, role='physician'
            )

            app_efficiency = st.session_state.simulator.simulate_provider_efficiency(
                nursing_q + exam_callbacks + peer_interrupts,  # APPs don't handle transfer calls
                providers, workload['app'], critical_events_per_day,
                admissions, adc, role='app'
            )

            # Display role-specific metrics
            st.markdown("### Provider-Specific Metrics")
            metrics_cols = st.columns(2)

            with metrics_cols[0]:
                st.markdown("#### Physician Metrics")
                st.metric(
                    "Efficiency",
                    f"{physician_efficiency:.0%}",
                    help="Physician-specific workflow efficiency"
                )
                physician_burnout = st.session_state.simulator.calculate_burnout_risk(
                    workload['physician'],
                    interrupts_per_provider,
                    critical_events_per_day,
                    role='physician'
                )
                st.metric(
                    "Burnout Risk",
                    f"{physician_burnout:.0%}",
                    help="Physician-specific burnout risk"
                )

            with metrics_cols[1]:
                st.markdown("#### APP Metrics")
                st.metric(
                    "Efficiency",
                    f"{app_efficiency:.0%}",
                    help="APP-specific workflow efficiency"
                )
                app_burnout = st.session_state.simulator.calculate_burnout_risk(
                    workload['app'],
                    interrupts_per_provider,
                    critical_events_per_day,
                    role='app'
                )
                st.metric(
                    "Burnout Risk",
                    f"{app_burnout:.0%}",
                    help="APP-specific burnout risk"
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
            # Administrator View with new Scenario Management section
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
                        "Workload": workload['combined'],
                        "Interruptions": interrupts_per_provider/50,  # Normalized
                        "Critical Events": critical_events_per_day/5,  # Normalized
                        "Cognitive Load": cognitive_load/100,
                        "Efficiency Loss": 1-efficiency
                    }),
                    use_container_width=True
                )

            # New Scenario Management Section
            st.markdown("### Scenario Management")
            scenario_tab1, scenario_tab2, scenario_tab3 = st.tabs([
                "Create Scenario", "Compare Scenarios", "Historical Analysis"
            ])

            with scenario_tab1:
                st.markdown("#### Create New Scenario")
                scenario_name = st.text_input("Scenario Name")
                scenario_description = st.text_area("Description")

                st.markdown("##### Intervention Strategies")
                protected_time = st.checkbox("Enable Protected Time Blocks")
                if protected_time:
                    protected_start = st.slider("Protected Time Start (Hour)", 0, 23, 9)
                    protected_duration = st.slider("Duration (Hours)", 1, 8, 2)

                staff_distribution = st.checkbox("Optimize Staff Distribution")
                if staff_distribution:
                    physician_ratio = st.slider("Physician/APP Ratio", 0.0, 1.0, 0.5)

                task_bundling = st.checkbox("Enable Task Bundling")
                if task_bundling:
                    bundling_efficiency = st.slider("Expected Efficiency Gain", 0.0, 0.5, 0.2)

                with st.expander("🤖 AI Assistant Recommendations", expanded=True):
                    if st.button("Get AI Recommendations"):
                        current_metrics = {
                            'efficiency': efficiency,
                            'cognitive_load': cognitive_load,
                            'burnout_risk': burnout_risk,
                            'workload': workload['combined']
                        }

                        scenario_config = {
                            'base_config': {
                                'providers': providers,
                                'adc': adc,
                                'consults': consults,
                                'critical_events': critical_events
                            },
                            'interventions': {
                                'protected_time_blocks': [{
                                    'start_hour': protected_start,
                                    'end_hour': protected_start + protected_duration,
                                    'reduction_factor': 0.5
                                }] if protected_time else None,
                                'staff_distribution': {
                                    'physician_ratio': physician_ratio
                                } if staff_distribution else None,
                                'task_bundling': {
                                    'efficiency_factor': 1 - bundling_efficiency
                                } if task_bundling else None
                            }
                        }

                        with st.spinner("Getting AI recommendations..."):
                            advice = st.session_state.scenario_advisor.get_optimization_advice(
                                scenario_config, current_metrics
                            )

                            if advice['status'] == 'success':
                                st.markdown("### AI Recommendations")
                                for i, rec in enumerate(advice['recommendations'], 1):
                                    st.markdown(f"{i}. {rec}")
                                    st.markdown("---")

                                st.markdown("### Expected Impact")
                                impact_cols = st.columns(3)

                                with impact_cols[0]:
                                    st.metric(
                                        "Efficiency Change",
                                        f"{advice['impact_analysis']['efficiency']:+.1%}",
                                        help="Expected change in workflow efficiency"
                                    )

                                with impact_cols[1]:
                                    st.metric(
                                        "Cognitive Load Change",
                                        f"{advice['impact_analysis']['cognitive_load']:+.1%}",
                                        help="Expected change in cognitive load"
                                    )

                                with impact_cols[2]:
                                    st.metric(
                                        "Burnout Risk Change",
                                        f"{advice['impact_analysis']['burnout_risk']:+.1%}",
                                        help="Expected change in burnout risk"
                                    )

                                st.progress(
                                    advice['confidence'],
                                    text=f"AI Confidence Score: {advice['confidence']:.1%}"
                                )
                            else:
                                st.error(f"Unable to get AI recommendations: {advice['message']}")

                if st.button("Save Scenario"):
                    try:
                        # Create scenario configuration
                        base_config = {
                            'providers': providers,
                            'adc': adc,
                            'consults': consults,
                            'critical_events': critical_events,
                            'workload': workload['combined']
                        }

                        interventions = {
                            'protected_time_blocks': [{
                                'start_hour': protected_start,
                                'end_hour': protected_start + protected_duration,
                                'reduction_factor': 0.5
                            }] if protected_time else None,
                            'staff_distribution': {
                                'physician_ratio': physician_ratio
                            } if staff_distribution else None,
                            'task_bundling': {
                                'efficiency_factor': 1 - bundling_efficiency
                            } if task_bundling else None
                        }

                        # Save scenario to database
                        db = next(get_db())
                        scenario = save_scenario(
                            db, scenario_name, scenario_description,
                            base_config, interventions
                        )
                        st.success(f"Scenario '{scenario_name}' saved successfully!")

                    except Exception as e:
                        st.error(f"Error saving scenario: {str(e)}")

            with scenario_tab2:
                st.markdown("#### Compare Scenarios")
                db = next(get_db())
                scenarios = get_scenarios(db)

                if scenarios:
                    selected_scenarios = st.multiselect(
                        "Select Scenarios to Compare",
                        options=[s.name for s in scenarios],
                        max_selections=3
                    )

                    if selected_scenarios:
                        if st.button("Run Comparison"):
                            comparison_results = st.session_state.scenario_manager.compare_scenarios(
                                selected_scenarios
                            )

                            # Key Metrics Comparison
                            st.markdown("### Key Metrics Comparison")
                            metrics_fig = go.Figure()

                            for scenario in selected_scenarios:
                                scenario_data = comparison_results[
                                    comparison_results['scenario_name'] == scenario
                                ]
                                metrics_fig.add_trace(go.Bar(
                                    name=scenario,
                                    x=['Efficiency', 'Cognitive Load', 'Burnout Risk'],
                                    y=[
                                        scenario_data['metrics'].iloc[0]['efficiency'],
                                        scenario_data['metrics'].iloc[0]['cognitive_load'],
                                        scenario_data['metrics'].iloc[0]['burnout_risk']
                                    ],
                                    text=[
                                        f"{v:.1%}" for v in [
                                            scenario_data['metrics'].iloc[0]['efficiency'],
                                            scenario_data['metrics'].iloc[0]['cognitive_load'],
                                            scenario_data['metrics'].iloc[0]['burnout_risk']
                                        ]
                                    ],
                                    textposition='auto'
                                ))

                            metrics_fig.update_layout(
                                title="Scenario Comparison - Key Metrics",
                                barmode='group',
                                yaxis=dict(
                                    title="Score",
                                    tickformat=".0%",
                                    range=[0, 1]
                                ),
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            st.plotly_chart(metrics_fig, use_container_width=True)

                            # Time Allocation Comparison
                            st.markdown("### Time Allocation Comparison")
                            time_cols = st.columns(len(selected_scenarios))
                            for idx, scenario in enumerate(selected_scenarios):
                                scenario_data = comparison_results[
                                    comparison_results['scenario_name'] == scenario
                                ]
                                with time_cols[idx]:
                                    st.markdown(f"**{scenario}**")
                                    time_data = scenario_data['metrics'].iloc[0]
                                    fig = go.Figure(data=[go.Pie(
                                        labels=['Direct Care', 'Interruptions', 'Critical Events', 'Admin'],
                                        values=[
                                            time_data.get('direct_care_time', 25),
                                            time_data.get('interruption_time', 15),
                                            time_data.get('critical_time', 20),
                                            time_data.get('admin_time', 40)
                                        ],
                                        hole=.4
                                    )])
                                    fig.update_layout(title="Time Distribution")
                                    st.plotly_chart(fig, use_container_width=True)

                            # Intervention Effectiveness
                            st.markdown("### Intervention Effectiveness")
                            effectiveness_data = []
                            for scenario in selected_scenarios:
                                scenario_data = comparison_results[
                                    comparison_results['scenario_name'] == scenario
                                ]
                                intervention_data = scenario_data['metrics'].iloc[0].get('intervention_effectiveness', {})
                                effectiveness_data.append({
                                    'Scenario': scenario,
                                    'Protected Time': intervention_data.get('protected_time', 0),
                                    'Staff Distribution': intervention_data.get('staff_distribution', 0),
                                    'Task Bundling': intervention_data.get('task_bundling', 0)
                                })

                            effectiveness_df = pd.DataFrame(effectiveness_data)
                            effectiveness_fig = go.Figure()

                            for col in ['Protected Time', 'Staff Distribution', 'Task Bundling']:
                                effectiveness_fig.add_trace(go.Bar(
                                    name=col,
                                    x=effectiveness_df['Scenario'],
                                    y=effectiveness_df[col],
                                    text=[f"{v:.0%}" for v in effectiveness_df[col]],
                                    textposition='auto'
                                ))

                            effectiveness_fig.update_layout(
                                title="Intervention Effectiveness by Type",
                                barmode='group',
                                yaxis=dict(
                                    title="Effectiveness Score",
                                    tickformat=".0%",
                                    range=[0, 1]
                                )
                            )
                            st.plotly_chart(effectiveness_fig, use_container_width=True)

                            # Risk Assessment Radar Chart
                            st.markdown("### Risk Assessment")
                            risk_fig = go.Figure()

                            risk_categories = [
                                'Workflow Disruption',
                                'Provider Burnout',
                                'Patient Care Impact',
                                'Resource Utilization',
                                'Implementation Risk'
                            ]

                            for scenario in selected_scenarios:
                                scenario_data = comparison_results[
                                    comparison_results['scenario_name'] == scenario
                                ]
                                risk_data = scenario_data['metrics'].iloc[0].get('risk_assessment', {})

                                risk_values = [
                                    risk_data.get('workflow_disruption', 0.5),
                                    risk_data.get('provider_burnout', 0.5),
                                    risk_data.get('patient_care_impact', 0.5),
                                    risk_data.get('resource_utilization', 0.5),
                                    risk_data.get('implementation_risk', 0.5),
                                    risk_data.get('workflow_disruption', 0.5)  # Close the radar plot
                                ]

                                risk_fig.add_trace(go.Scatterpolar(
                                    r=risk_values,
                                    theta=risk_categories + [risk_categories[0]],
                                    name=scenario,
                                    fill='toself'
                                ))

                            risk_fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )
                                ),
                                showlegend=True
                            )
                            st.plotly_chart(risk_fig, use_container_width=True)
                else:
                    st.info("No scenarios available. Create scenarios to compare them.")

            with scenario_tab3:
                st.markdown("#### Historical Analysis")
                db = next(get_db())
                scenarios = get_scenarios(db)

                if scenarios:
                    selected_scenario = st.selectbox(
                        "Select Scenario",
                        options=[s.name for s in scenarios]
                    )

                    if selected_scenario:
                        scenario = next(s for s in scenarios if s.name == selected_scenario)
                        results = get_scenario_results(db, scenario.id)

                        if results:
                            # Create historical trend visualization
                            trend_data = pd.DataFrame([{
                                'timestamp': r.timestamp,
                                'efficiency': r.efficiency,
                                'cognitive_load': r.cognitive_load,
                                'burnout_risk': r.burnout_risk,
                                'roi': r.roi
                            } for r in results])

                            st.line_chart(trend_data.set_index('timestamp'))
                        else:
                            st.info("No historical data available for this scenario.")
                else:
                    st.info("No scenarios available for historical analysis.")

            # Predictive Analytics
            st.markdown("### Predictive Insights")

            try:
                current_features = np.array([
                    nursing_q, exam_callbacks, peer_interrupts, transfer_calls,
                    providers, admissions, consults, critical_events
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
                        cognitive_load, workload['combined'], burnout_risk,
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