import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from simulator import WorkflowSimulator


def calculate_interruptions(nursing_q, exam_callbacks, peer_interrupts,
                            transfer_calls, providers, simulator):
    """Calculate interruption metrics using actual input values
    Input frequencies are per hour per provider
    Returns:
    - interrupts_per_provider: number of interruptions per provider per shift
    - time_lost: total organizational minutes lost to interruptions per shift
    """
    # Calculate total interruptions per provider per shift (12 hours)
    per_provider = (nursing_q + exam_callbacks + peer_interrupts +
                    transfer_calls) * 12

    # Calculate time lost per hour using provided simulator settings
    hourly_time = (
        nursing_q * simulator.interruption_times['nursing_question'] +
        exam_callbacks * simulator.interruption_times['exam_callback'] +
        peer_interrupts * simulator.interruption_times['peer_interrupt'] +
        transfer_calls * simulator.interruption_times['transfer_call'])

    # Calculate total time lost for all providers over full shift
    time_lost = hourly_time * 12 * providers

    return per_provider, time_lost


def calculate_workload(adc, consults, providers, simulator):
    """Calculate workload relative to total available shift minutes"""
    # Calculate base ICU workload (scales 0-1 for 0-16 patients)
    icu_workload = adc / 16.0

    # Calculate floor consults workload (physician only)
    consult_time = consults * simulator.admission_times['consult']

    # Calculate total available minutes
    available_minutes = providers * 12 * 60

    # Calculate physician-specific impact
    physician_minutes_lost = consult_time
    physician_impact = physician_minutes_lost / available_minutes if available_minutes > 0 else 0

    # Final workload considers base ICU load plus physician-specific impact
    physician_workload = icu_workload + physician_impact
    app_workload = icu_workload  # APP workload excludes consults

    # Always return dictionary structure, even with zero workload
    return {
        'physician': physician_workload,
        'app': app_workload,
        'combined': (physician_workload + app_workload) / 2
    }


def create_interruption_chart(nursing_q, exam_callbacks, peer_interrupts,
                              transfer_calls, simulator):
    # Calculate time impact per hour using current simulator settings
    nursing_time = nursing_q * simulator.interruption_times['nursing_question']
    exam_time = exam_callbacks * simulator.interruption_times['exam_callback']
    peer_time = peer_interrupts * simulator.interruption_times['peer_interrupt']
    transfer_time = transfer_calls * simulator.interruption_times[
        'transfer_call']

    categories = [
        'Nursing Questions', 'Exam Callbacks', 'Peer Interruptions',
        'Transfer Calls'
    ]
    values = [nursing_time, exam_time, peer_time, transfer_time]

    # Include time per interruption for more detailed analysis
    hover_text = [
        f'Time per interruption: {simulator.interruption_times["nursing_question"]} min',
        f'Time per interruption: {simulator.interruption_times["exam_callback"]} min',
        f'Time per interruption: {simulator.interruption_times["peer_interrupt"]} min',
        f'Time per interruption: {simulator.interruption_times["transfer_call"]} min'
    ]

    # Create a more detailed bar chart
    fig = go.Figure()

    # Add bars with hover information
    fig.add_trace(
        go.Bar(x=categories,
               y=values,
               text=[f'{v:.1f} min/hr' for v in values],
               textposition='auto',
               marker_color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'],
               hovertext=hover_text,
               hoverinfo='text'))

    fig.update_layout(title='Time Impact of Interruptions (minutes per hour)',
                      yaxis_title='Minutes per Hour',
                      showlegend=False,
                      plot_bgcolor='white')

    return fig


def create_time_allocation_pie(time_lost,
                               consult_time,
                               providers=1,
                               available_hours=12,
                               role='physician'):
    """Create pie chart showing time allocation during shift
    Args:
        time_lost: Total organizational time lost to interruptions (minutes)
        consult_time: Time spent on floor consults (minutes)
        providers: Number of providers
        available_hours: Shift duration in hours
        role: Provider role ('physician' or 'app')
    """
    available_minutes = available_hours * 60
    consult_time = float(consult_time) if consult_time is not None else 0.0

    if role == 'physician':
        # Physician time allocation includes consults
        time_lost_per_provider = time_lost / providers if providers > 0 else 0
        consult_time_per_provider = consult_time  # All consult time affects physician
        remaining_minutes = available_minutes - time_lost_per_provider - consult_time_per_provider

        # Ensure we don't show negative remaining time
        remaining_minutes = max(0, remaining_minutes)

        labels = [
            'Interrupted Time', 'Consult Time', 'Available Clinical Time'
        ]
        values = [
            time_lost_per_provider, consult_time_per_provider,
            remaining_minutes
        ]
        colors = ['#ff6b6b', '#ffd93d', '#4ecdc4']

    else:  # APP
        # APP time allocation excludes consults
        time_lost_per_provider = time_lost / providers if providers > 0 else 0
        remaining_minutes = available_minutes - time_lost_per_provider

        # Ensure we don't show negative remaining time
        remaining_minutes = max(0, remaining_minutes)

        labels = ['Interrupted Time', 'Available Clinical Time']
        values = [time_lost_per_provider, remaining_minutes]
        colors = ['#ff6b6b', '#4ecdc4']

    # Handle division by zero
    total_minutes = sum(values)
    if total_minutes == 0:
        percentages = [0] * len(values)
    else:
        percentages = [v / available_minutes * 100 for v in values]

    # Create custom hover text
    hover_text = []
    for i, (label, value, pct) in enumerate(zip(labels, values, percentages)):
        hover_text.append(f'{label}: {value:.0f} min ({pct:.0f}%)')

    fig = go.Figure(data=[
        go.Pie(labels=labels,
               values=values,
               textinfo='percent+value',
               hovertemplate="%{customdata}<extra></extra>",
               customdata=hover_text,
               marker=dict(colors=colors))
    ])

    role_title = "Physician" if role == "physician" else "APP"
    fig.update_layout(
        title=f'{role_title} Time Allocation (12-hour shift)',
        height=400,
        showlegend=True,
        margin=dict(t=100, b=30, l=40, r=20),
        annotations=[
            dict(
                text=
                f'Total Shift Duration: {available_minutes} minutes (12 hours)',
                showarrow=False,
                x=0.44,
                y=-0.0,
                font=dict(size=14))
        ])

    return fig


def create_workload_timeline(workload, providers, critical_events_per_day,
                             admissions, simulator):
    """Create timeline showing projected workload with tooltip explanation

    The workload timeline shows the relative workload throughout the day where:
    - 1.0 represents full utilization of all available provider time
    - Values > 1.0 indicate overload conditions
    - Base workload accounts for parallel provider work capacity
    """
    hours = list(range(8, 21))  # 8 AM to 8 PM
    shift_minutes = len(hours) * 60

    # Base workload variation throughout the day (±20% sinusoidal variation)
    base_variation = 0.2 * np.sin((np.array(hours) - 8) * np.pi / 12)

    # Add specific rounding inefficiency (9-11 AM)
    rounding_hours = np.array([(9 <= h < 11) for h in hours])
    data_aggregation_overhead = 0.8 * rounding_hours  # 80% overhead during rounds
    repeated_data_collection = 0.3 * rounding_hours  # 30% inefficiency from repeated data collection

    # Smooth the transition at rounding boundaries
    transition_start = np.array([(h == 8)
                                 for h in hours]) * 0.4  # Ramp up to rounds
    transition_end = np.array([(h == 11)
                               for h in hours]) * 0.3  # Ramp down from rounds
    rounding_effect = rounding_hours + transition_start + transition_end

    # Factor in provider count for rounding impact (overhead is distributed across providers)
    base_variation = (base_variation + data_aggregation_overhead +
                      repeated_data_collection) / providers

    # Calculate critical event impact on available provider time
    # During first hour, both providers are unavailable (reduced parallel capacity)
    first_hour_impact = (min(60, simulator.critical_event_time) / 60) * (
        2 / providers)  # Scale by provider ratio
    # After first hour, one provider remains on critical event
    remaining_impact = (max(0, simulator.critical_event_time - 60) / 60) * (
        1 / providers)  # Scale by provider ratio

    # Calculate total critical impact accounting for parallel provider availability
    critical_impact = (
        critical_events_per_day * (
            first_hour_impact
            +  # Initial impact when both providers are needed
            remaining_impact  # Remaining impact with one provider
        )) / 12  # Normalize to shift duration

    # Scale critical impact by actual duration vs default
    scaled_critical_impact = critical_impact * (simulator.critical_event_time /
                                                105)

    # Generate random event distributions
    admission_times = np.array(
        sorted(
            np.random.choice(
                shift_minutes, size=int(admissions), replace=False))) // 60
    critical_event_times = np.array(
        sorted(
            np.random.choice(shift_minutes,
                             size=int(critical_events_per_day),
                             replace=False))) // 60

    # Initialize workload timeline with base variation
    workload_timeline = np.ones(len(hours)) * workload * (1 + base_variation)

    # Add admission impacts
    for time in admission_times:
        if time < len(hours):
            # Increase workload for admission duration
            duration = 2  # Impact lasts ~2 hours
            end_time = min(time + duration, len(hours))
            workload_timeline[time:end_time] *= (
                providers / (providers - 1)) if providers > 1 else 2.0

    # Add critical event impacts
    for time in critical_event_times:
        if time < len(hours):
            # First hour - both providers unavailable
            first_hour = min(time + 1, len(hours))
            workload_timeline[time:first_hour] *= 2.0

            # Remaining time - one provider unavailable
            remaining_duration = 2  # ~2 hours total impact
            end_time = min(time + remaining_duration, len(hours))
            if first_hour < end_time:
                workload_timeline[first_hour:end_time] *= (
                    providers / (providers - 1)) if providers > 1 else 1.5

    fig = go.Figure()

    # Add workload area
    fig.add_trace(
        go.Scatter(x=hours,
                   y=workload_timeline,
                   fill='tozeroy',
                   name='Workload',
                   line=dict(color='#0096c7', width=2),
                   hovertemplate=
                   "Hour: %{x}<br>Relative Workload: %{y:.2f}<extra></extra>"))

    # Add capacity workload reference line (1.0 = full utilization of total provider capacity)
    fig.add_trace(
        go.Scatter(x=hours,
                   y=[1.0] * len(hours),
                   name='Capacity Load',
                   line=dict(color='#666666', dash='dash'),
                   hovertemplate="Target Load: 1.0<extra></extra>"))

    fig.update_layout(
        title='Projected Dayshift Workload (8 AM - 8 PM)',
        xaxis_title='Hour of Day',
        yaxis_title='Relative Workload (1.0 = Full Provider Capacity)',
        xaxis=dict(ticktext=[
            '8 AM', '10 AM', '12 PM', '2 PM', '4 PM', '6 PM', '8 PM'
        ],
                   tickvals=[8, 10, 12, 14, 16, 18, 20]),
        showlegend=True,
        legend=dict(orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1))

    return fig


def generate_report_data(interrupts_per_provider, time_lost, efficiency,
                         cognitive_load, workload, burnout_risk,
                         interrupt_time, admission_time, critical_time,
                         providers):
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
            "interruption_time_minutes":
            round(interrupt_time, 1),
            "admission_time_minutes":
            round(admission_time, 1),
            "critical_time_minutes":
            round(critical_time, 1),
            "total_time_minutes":
            round(interrupt_time + admission_time + critical_time, 1),
            "time_per_provider_minutes":
            round(
                (interrupt_time + admission_time + critical_time) / providers,
                1)
        }
    }


def format_recommendations(efficiency, cognitive_load, burnout_risk,
                           total_time):
    """Format recommendations based on metrics"""
    recommendations = []

    if burnout_risk > 0.7:
        recommendations.append(
            "High burnout risk detected. Consider increasing provider coverage or "
            "implementing interruption reduction strategies.")
    if cognitive_load > 80:
        recommendations.append(
            "High cognitive load detected. Consider workflow optimization or "
            "additional support staff.")
    if efficiency < 0.7:
        recommendations.append(
            "Low efficiency detected. Review interruption patterns and implement "
            "protected time for critical tasks.")
    if total_time > 720:  # 12 hours in minutes
        recommendations.append(
            "Total task time exceeds shift duration. Current workload may not be "
            "sustainable.")

    return recommendations


def create_burnout_radar_chart(risk_components):
    """Create a radar chart showing different burnout risk components"""
    categories = list(risk_components.keys())
    values = list(risk_components.values())

    # Add the first value again to close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(r=values,
                        theta=categories,
                        fill='toself',
                        fillcolor='rgba(0, 150, 199, 0.3)',
                        line=dict(color='#0096c7', width=2)))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      showlegend=False,
                      title='Burnout Risk Components')

    return fig


def create_burnout_gauge(total_risk, thresholds):
    """Create a gauge chart showing overall burnout risk"""
    # Create color steps based on thresholds
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=total_risk * 100,
            title={'text': "Overall Burnout Risk"},
            gauge={
                'axis': {
                    'range': [0, 100]
                },
                'bar': {
                    'color': "darkgray"
                },
                'steps': [{
                    'range': [0, thresholds['moderate'] * 100],
                    'color': colors[0]
                }, {
                    'range':
                    [thresholds['moderate'] * 100, thresholds['high'] * 100],
                    'color':
                    colors[1]
                }, {
                    'range':
                    [thresholds['high'] * 100, thresholds['severe'] * 100],
                    'color':
                    colors[2]
                }, {
                    'range': [thresholds['severe'] * 100, 100],
                    'color': colors[3]
                }],
                'threshold': {
                    'line': {
                        'color': "red",
                        'width': 4
                    },
                    'thickness': 0.75,
                    'value': total_risk * 100
                }
            }))

    fig.update_layout(height=300)
    return fig


def create_burnout_trend_chart(risk_data, shift_hours=12):
    """Create a line chart showing burnout risk trend throughout the shift"""
    hours = list(range(shift_hours))
    # Simulate risk variation throughout the shift
    base_trend = np.array([risk_data['total_risk']] * shift_hours)

    # Add slight variations based on typical shift patterns
    variation = 0.1 * np.sin(np.pi * np.array(hours) / shift_hours)
    fatigue_factor = np.linspace(
        0, 0.15, shift_hours)  # Gradual increase in risk due to fatigue

    risk_trend = base_trend + variation + fatigue_factor
    risk_trend = np.clip(risk_trend, 0,
                         1)  # Ensure values stay between 0 and 1

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=hours,
                   y=risk_trend,
                   mode='lines+markers',
                   name='Risk Level',
                   line=dict(color='#0096c7', width=2)))

    # Add threshold lines
    for threshold_name, threshold_value in risk_data['thresholds'].items():
        fig.add_trace(
            go.Scatter(x=hours,
                       y=[threshold_value] * len(hours),
                       mode='lines',
                       name=f'{threshold_name.capitalize()} Risk',
                       line=dict(dash='dash', width=1)))

    fig.update_layout(title='Burnout Risk Trend During Shift',
                      xaxis_title='Hour of Shift',
                      yaxis_title='Risk Level',
                      yaxis=dict(range=[0, 1]),
                      showlegend=True)

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
        recommendations.append(
            "- Consider implementing protected time periods to reduce interruptions"
        )
    if components['workload_risk'] > 0.7:
        recommendations.extend([
            "- Evaluate task distribution and consider additional support staff",
            "- Consider implementing a unified data visualization system to reduce rounding overhead",
            "- Implement persistent storage for static patient data to reduce redundant data collection"
        ])
    if components['critical_events_risk'] > 0.7:
        recommendations.append(
            "- Review critical event response protocols and support systems")
    if components['efficiency_risk'] > 0.7:
        recommendations.append("- Assess workflow optimization opportunities")
    if components['cognitive_load_risk'] > 0.7:
        recommendations.append(
            "- Implement cognitive load management strategies")

    return recommendations


def create_prediction_trend_chart(predictions):
    """Create a line chart showing predicted workload and burnout trends"""
    fig = go.Figure()

    # Add workload prediction line
    fig.add_trace(
        go.Scatter(x=[p['day'] for p in predictions],
                   y=[p['workload'] for p in predictions],
                   name='Predicted Workload',
                   line=dict(color='#0096c7', width=2)))

    # Add burnout prediction line
    fig.add_trace(
        go.Scatter(x=[p['day'] for p in predictions],
                   y=[p['burnout'] for p in predictions],
                   name='Predicted Burnout Risk',
                   line=dict(color='#ef476f', width=2)))

    fig.update_layout(title='Predicted Trends (Next 7 Days)',
                      xaxis_title='Date',
                      yaxis_title='Risk Level',
                      yaxis=dict(range=[0, 1]),
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1))

    return fig


def create_feature_importance_chart(importance_dict):
    """Create a horizontal bar chart showing feature importance"""
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())

    # Sort by importance
    sorted_indices = np.argsort(importance)
    features = [features[i] for i in sorted_indices]
    importance = [importance[i] for i in sorted_indices]

    fig = go.Figure(
        go.Bar(x=importance,
               y=features,
               orientation='h',
               marker_color='#0096c7'))

    fig.update_layout(title='Feature Importance Analysis',
                      xaxis_title='Relative Importance',
                      yaxis_title='Feature',
                      showlegend=False)

    return fig
