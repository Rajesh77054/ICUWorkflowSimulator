import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from simulator import WorkflowSimulator

def calculate_interruptions(nursing_q, exam_callbacks, peer_interrupts, providers):
    """Calculate interruption metrics
    Returns:
    - interrupts_per_provider: number of interruptions per shift
    - time_lost: total minutes lost to interruptions per shift
    """
    simulator = WorkflowSimulator()

    # Calculate total interruptions per shift (12 hours)
    total_interrupts = (nursing_q + exam_callbacks + peer_interrupts) * 12
    per_provider = total_interrupts / providers

    # Use simulator's calculation method for consistency
    time_lost = simulator.calculate_interruption_time(
        nursing_q, exam_callbacks, peer_interrupts
    )

    return per_provider, time_lost

def calculate_workload(admissions, consults, transfers, critical_events, providers, simulator):
    # Calculate total time required for all tasks using current simulator settings
    admission_time = admissions * (0.7 * simulator.admission_times['simple'] + 
                                   0.3 * simulator.admission_times['complex'])
    consult_time = consults * simulator.admission_times['consult']
    transfer_time = transfers * simulator.admission_times['transfer']
    critical_time = critical_events * simulator.critical_event_time

    total_time = (admission_time + consult_time + transfer_time + critical_time) / 60  # Convert to hours
    workload_per_provider = total_time / providers / 12  # Normalize to 12-hour shift

    return workload_per_provider

def create_interruption_chart(nursing_q, exam_callbacks, peer_interrupts, simulator):
    # Calculate time impact per hour using current simulator settings
    nursing_time = nursing_q * simulator.interruption_times['nursing_question']
    exam_time = exam_callbacks * simulator.interruption_times['exam_callback']
    peer_time = peer_interrupts * simulator.interruption_times['peer_interrupt']

    categories = ['Nursing Questions', 'Exam Callbacks', 'Peer Interruptions']
    values = [nursing_time, exam_time, peer_time]

    # Include time per interruption for more detailed analysis
    hover_text = [
        f'Time per interruption: {simulator.interruption_times["nursing_question"]} min',
        f'Time per interruption: {simulator.interruption_times["exam_callback"]} min',
        f'Time per interruption: {simulator.interruption_times["peer_interrupt"]} min'
    ]

    # Create a more detailed bar chart
    fig = go.Figure()

    # Add bars with hover information
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        text=[f'{v:.1f} min/hr' for v in values],
        textposition='auto',
        marker_color=['#66c2a5', '#fc8d62', '#8da0cb'],
        hovertext=hover_text,
        hoverinfo='text'
    ))

    fig.update_layout(
        title='Time Impact of Interruptions (minutes per hour)',
        yaxis_title='Minutes per Hour',
        showlegend=False,
        plot_bgcolor='white'
    )

    return fig


def create_time_allocation_pie(time_lost, available_hours=12):
    # Convert time_lost to minutes for more precise representation
    time_lost_minutes = time_lost * 60
    available_minutes = available_hours * 60

    labels = ['Time Lost to Interruptions', 'Available Time']
    values = [time_lost_minutes, available_minutes - time_lost_minutes]

    fig = px.pie(
        values=values,
        names=labels,
        title='Time Allocation per Shift (minutes)',
        color_discrete_sequence=['#ff9999', '#66b3ff']
    )

    # Add total minutes annotation
    fig.add_annotation(
        text=f'Total Shift: {available_minutes} minutes',
        showarrow=False,
        x=0.5,
        y=-0.2
    )

    return fig

def create_workload_timeline(workload, providers, critical_events_per_day, simulator):
    # Create hours array for 12-hour shift (8 AM to 8 PM)
    hours = list(range(8, 21))

    # Base workload variation throughout the day
    base_variation = 0.2 * np.sin((np.array(hours) - 8) * np.pi / 12)

    # Add specific rounding inefficiency (9-11 AM)
    rounding_hours = np.array([(9 <= h < 11) for h in hours])
    data_aggregation_overhead = 0.8 * rounding_hours  # 80% overhead during rounds
    repeated_data_collection = 0.3 * rounding_hours   # 30% inefficiency from repeated static data collection
    
    # Smooth the transition at rounding boundaries with consistent ramping
    transition_start = np.array([(h == 8) for h in hours]) * 0.4  # Aligned with simulator
    transition_end = np.array([(h == 11) for h in hours]) * 0.3   # Aligned with simulator
    rounding_effect = rounding_hours + transition_start + transition_end
    
    # Add gradual buildup and cooldown
    pre_rounds = np.array([(h == 8) for h in hours]) * 0.4
    post_rounds = np.array([(h == 11) for h in hours]) * 0.3
    rounding_hours = rounding_hours + pre_rounds + post_rounds

    # Combine variations
    base_variation = base_variation + data_aggregation_overhead + repeated_data_collection

    # Calculate critical event impact with cascading effect
    first_hour_impact = min(60, simulator.critical_event_time) / 60  # Full impact in first hour
    remaining_impact = max(0, simulator.critical_event_time - 60) / 60  # Half impact thereafter

    # During first hour both providers are occupied (2x impact)
    critical_impact = (critical_events_per_day * (
        (first_hour_impact * 2) +  # Both providers unavailable
        (remaining_impact * 1)      # One provider unavailable
    )) / 12  # Normalize to shift duration

    scaled_critical_impact = critical_impact * (simulator.critical_event_time / 105)

    # Combine base workload with variations and critical impact
    workload_timeline = workload * (1 + base_variation + scaled_critical_impact)

    fig = go.Figure()

    # Add base workload area
    fig.add_trace(go.Scatter(
        x=hours,
        y=workload_timeline,
        fill='tozeroy',
        name='Workload',
        line=dict(color='#0096c7', width=2)
    ))

    # Add optimal workload reference line
    fig.add_trace(go.Scatter(
        x=hours,
        y=[1.0] * len(hours),
        name='Optimal Load',
        line=dict(color='#666666', dash='dash')
    ))

    fig.update_layout(
        title='Projected Dayshift Workload (8 AM - 8 PM)',
        xaxis_title='Hour of Day',
        yaxis_title='Relative Workload',
        xaxis=dict(
            ticktext=['8 AM', '10 AM', '12 PM', '2 PM', '4 PM', '6 PM', '8 PM'],
            tickvals=[8, 10, 12, 14, 16, 18, 20]
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

    return fig

def generate_report_data(
    interrupts_per_provider,
    time_lost,
    efficiency,
    cognitive_load,
    workload,
    burnout_risk,
    interrupt_time,
    admission_time,
    critical_time,
    providers
):
    """Generate a structured dictionary of report data"""
    return {
        "metrics": {
            "interruptions_per_provider": round(interrupts_per_provider, 2),
            "time_lost_hours": round(time_lost, 2),
            "provider_efficiency": round(efficiency * 100, 1),
            "cognitive_load": round(cognitive_load, 1),
            "workload_level": round(workload, 2),
            "burnout_risk": round(burnout_risk * 100, 1)
        },
        "time_analysis": {
            "interruption_time_minutes": round(interrupt_time, 1),
            "admission_time_minutes": round(admission_time, 1),
            "critical_time_minutes": round(critical_time, 1),
            "total_time_minutes": round(interrupt_time + admission_time + critical_time, 1),
            "time_per_provider_minutes": round((interrupt_time + admission_time + critical_time) / providers, 1)
        }
    }

def format_recommendations(efficiency, cognitive_load, burnout_risk, total_time):
    """Format recommendations based on metrics"""
    recommendations = []

    if burnout_risk > 0.7:
        recommendations.append(
            "High burnout risk detected. Consider increasing provider coverage or "
            "implementing interruption reduction strategies."
        )
    if cognitive_load > 80:
        recommendations.append(
            "High cognitive load detected. Consider workflow optimization or "
            "additional support staff."
        )
    if efficiency < 0.7:
        recommendations.append(
            "Low efficiency detected. Review interruption patterns and implement "
            "protected time for critical tasks."
        )
    if total_time > 720:  # 12 hours in minutes
        recommendations.append(
            "Total task time exceeds shift duration. Current workload may not be "
            "sustainable."
        )

    return recommendations

def create_burnout_radar_chart(risk_components):
    """Create a radar chart showing different burnout risk components"""
    categories = list(risk_components.keys())
    values = list(risk_components.values())

    # Add the first value again to close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 150, 199, 0.3)',
        line=dict(color='#0096c7', width=2)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title='Burnout Risk Components'
    )

    return fig

def create_burnout_gauge(total_risk, thresholds):
    """Create a gauge chart showing overall burnout risk"""
    # Create color steps based on thresholds
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = total_risk * 100,
        title = {'text': "Overall Burnout Risk"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkgray"},
            'steps': [
                {'range': [0, thresholds['moderate'] * 100], 'color': colors[0]},
                {'range': [thresholds['moderate'] * 100, thresholds['high'] * 100], 'color': colors[1]},
                {'range': [thresholds['high'] * 100, thresholds['severe'] * 100], 'color': colors[2]},
                {'range': [thresholds['severe'] * 100, 100], 'color': colors[3]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': total_risk * 100
            }
        }
    ))

    fig.update_layout(height=300)
    return fig

def create_burnout_trend_chart(risk_data, shift_hours=12):
    """Create a line chart showing burnout risk trend throughout the shift"""
    hours = list(range(shift_hours))
    # Simulate risk variation throughout the shift
    base_trend = np.array([risk_data['total_risk']] * shift_hours)

    # Add slight variations based on typical shift patterns
    variation = 0.1 * np.sin(np.pi * np.array(hours) / shift_hours)
    fatigue_factor = np.linspace(0, 0.15, shift_hours)  # Gradual increase in risk due to fatigue

    risk_trend = base_trend + variation + fatigue_factor
    risk_trend = np.clip(risk_trend, 0, 1)  # Ensure values stay between 0 and 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hours,
        y=risk_trend,
        mode='lines+markers',
        name='Risk Level',
        line=dict(color='#0096c7', width=2)
    ))

    # Add threshold lines
    for threshold_name, threshold_value in risk_data['thresholds'].items():
        fig.add_trace(go.Scatter(
            x=hours,
            y=[threshold_value] * len(hours),
            mode='lines',
            name=f'{threshold_name.capitalize()} Risk',
            line=dict(dash='dash', width=1)
        ))

    fig.update_layout(
        title='Burnout Risk Trend During Shift',
        xaxis_title='Hour of Shift',
        yaxis_title='Risk Level',
        yaxis=dict(range=[0, 1]),
        showlegend=True
    )

    return fig

def format_burnout_recommendations(risk_data):
    """Format detailed burnout risk recommendations"""
    recommendations = []

    # Base recommendations on risk category
    if risk_data['risk_category'] == 'severe':
        recommendations.extend([
            "URGENT: Immediate intervention required to address severe burnout risk:",
            "- Consider emergency staffing adjustments",
            "- Implement mandatory breaks and task redistribution",
            "- Schedule urgent review of workload distribution"
        ])
    elif risk_data['risk_category'] == 'high':
        recommendations.extend([
            "High burnout risk detected. Recommended actions:",
            "- Review and optimize provider scheduling",
            "- Implement additional support during peak hours",
            "- Consider workflow adjustments to reduce interruptions"
        ])
    elif risk_data['risk_category'] == 'moderate':
        recommendations.extend([
            "Moderate burnout risk present. Consider:",
            "- Monitoring workload distribution more closely",
            "- Implementing preventive measures",
            "- Reviewing interruption patterns"
        ])

    # Add specific recommendations based on risk components
    components = risk_data['risk_components']
    if components['interruption_risk'] > 0.7:
        recommendations.append("- Consider implementing protected time periods to reduce interruptions")
    if components['workload_risk'] > 0.7:
        recommendations.extend([
            "- Evaluate task distribution and consider additional support staff",
            "- Consider implementing a unified data visualization system to reduce rounding overhead",
            "- Implement persistent storage for static patient data to reduce redundant data collection"
        ])
    if components['critical_events_risk'] > 0.7:
        recommendations.append("- Review critical event response protocols and support systems")
    if components['efficiency_risk'] > 0.7:
        recommendations.append("- Assess workflow optimization opportunities")
    if components['cognitive_load_risk'] > 0.7:
        recommendations.append("- Implement cognitive load management strategies")

    return recommendations

def create_prediction_trend_chart(predictions):
    """Create a line chart showing predicted workload and burnout trends"""
    fig = go.Figure()

    # Add workload prediction line
    fig.add_trace(go.Scatter(
        x=[p['day'] for p in predictions],
        y=[p['workload'] for p in predictions],
        name='Predicted Workload',
        line=dict(color='#0096c7', width=2)
    ))

    # Add burnout prediction line
    fig.add_trace(go.Scatter(
        x=[p['day'] for p in predictions],
        y=[p['burnout'] for p in predictions],
        name='Predicted Burnout Risk',
        line=dict(color='#ef476f', width=2)
    ))

    fig.update_layout(
        title='Predicted Trends (Next 7 Days)',
        xaxis_title='Date',
        yaxis_title='Risk Level',
        yaxis=dict(range=[0, 1]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def create_feature_importance_chart(importance_dict):
    """Create a horizontal bar chart showing feature importance"""
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())

    # Sort by importance
    sorted_indices = np.argsort(importance)
    features = [features[i] for i in sorted_indices]
    importance = [importance[i] for i in sorted_indices]

    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color='#0096c7'
    ))

    fig.update_layout(
        title='Feature Importance Analysis',
        xaxis_title='Relative Importance',
        yaxis_title='Feature',
        showlegend=False
    )

    return fig