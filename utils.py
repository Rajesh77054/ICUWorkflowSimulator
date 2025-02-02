import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

def calculate_workload(admissions, consults, transfers, critical_events, providers):
    simulator = WorkflowSimulator()

    # Calculate total time required for all tasks
    admission_time = admissions * (0.7 * simulator.admission_times['simple'] + 
                                 0.3 * simulator.admission_times['complex'])
    consult_time = consults * simulator.admission_times['consult']
    transfer_time = transfers * simulator.admission_times['transfer']
    critical_time = critical_events * simulator.critical_event_time

    total_time = (admission_time + consult_time + transfer_time + critical_time) / 60  # Convert to hours
    workload_per_provider = total_time / providers / 12  # Normalize to 12-hour shift

    return workload_per_provider

def create_interruption_chart(nursing_q, exam_callbacks, peer_interrupts):
    simulator = WorkflowSimulator()

    categories = ['Nursing Questions', 'Exam Callbacks', 'Peer Interruptions']
    values = [
        nursing_q * simulator.interruption_times['nursing_question'],
        exam_callbacks * simulator.interruption_times['exam_callback'],
        peer_interrupts * simulator.interruption_times['peer_interrupt']
    ]

    fig = px.bar(
        x=categories,
        y=values,
        title='Time Impact of Interruptions (minutes per hour)',
        color=values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False)
    return fig

def create_time_allocation_pie(time_lost, available_hours=12):
    labels = ['Time Lost to Interruptions', 'Available Time']
    values = [time_lost, available_hours - time_lost]

    fig = px.pie(
        values=values,
        names=labels,
        title='Time Allocation per Shift (hours)',
        color_discrete_sequence=['#ff9999', '#66b3ff']
    )
    return fig

def create_workload_timeline(workload_baseline, providers):
    # Create hours array from 8 to 20 (8 AM to 8 PM)
    hours = list(range(8, 21))
    # Simulate workload variation throughout the dayshift
    # Peak during mid-day (around 2 PM)
    workload = workload_baseline * (1 + np.sin((np.array(hours) - 8) * np.pi / 12) * 0.3)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=workload,
        mode='lines+markers',
        name='Workload',
        line=dict(color='#0096c7')
    ))

    fig.update_layout(
        title='Projected Dayshift Workload (8 AM - 8 PM)',
        xaxis_title='Hour of Day',
        yaxis_title='Relative Workload',
        xaxis=dict(
            ticktext=['8 AM', '10 AM', '12 PM', '2 PM', '4 PM', '6 PM', '8 PM'],
            tickvals=[8, 10, 12, 14, 16, 18, 20]
        )
    )
    return fig