import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from simulator import WorkflowSimulator

def calculate_interruptions(nursing_q, exam_callbacks, peer_interrupts, providers):
    # Calculate for 12-hour dayshift
    total_interrupts = (nursing_q + exam_callbacks + peer_interrupts) * 12  # per 12-hour dayshift
    per_provider = total_interrupts / providers

    # Calculate time lost using simulator's time constants
    simulator = WorkflowSimulator()
    time_lost = (
        nursing_q * simulator.interruption_times['nursing_question'] + 
        exam_callbacks * simulator.interruption_times['exam_callback'] + 
        peer_interrupts * simulator.interruption_times['peer_interrupt']
    ) * 12 / 60  # Convert to hours

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

    # Adjust workload based on critical events
    # Critical events impact is proportional to their duration
    critical_impact = (critical_events_per_day * simulator.critical_event_time) / (12 * 60)  # Normalize to shift duration

    # Combine base workload with variations and critical impact
    workload_timeline = workload * (1 + base_variation + critical_impact)

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