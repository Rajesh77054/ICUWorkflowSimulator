import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def calculate_interruptions(nursing_q, exam_callbacks, peer_interrupts, providers):
    total_interrupts = (nursing_q + exam_callbacks + peer_interrupts) * 12  # per 12-hour shift
    per_provider = total_interrupts / providers
    time_lost = total_interrupts * 5 / 60  # assuming 5 minutes per interruption
    return per_provider, time_lost

def calculate_workload(admissions, consults, transfers, critical_events, providers):
    total_tasks = admissions * 1.5 + consults * 0.75 + transfers * 0.5 + critical_events * 2
    workload_per_provider = total_tasks / providers
    return workload_per_provider

def create_interruption_chart(nursing_q, exam_callbacks, peer_interrupts):
    categories = ['Nursing Questions', 'Exam Callbacks', 'Peer Interruptions']
    values = [nursing_q, exam_callbacks, peer_interrupts]
    
    fig = px.bar(
        x=categories,
        y=values,
        title='Interruptions Distribution (per hour)',
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
        title='Time Allocation per Shift',
        color_discrete_sequence=['#ff9999', '#66b3ff']
    )
    return fig

def create_workload_timeline(workload_baseline, providers):
    hours = list(range(24))
    # Simulate workload variation throughout the day
    workload = workload_baseline * (1 + np.sin(np.array(hours) * np.pi / 12) * 0.3)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=workload,
        mode='lines+markers',
        name='Workload',
        line=dict(color='#0096c7')
    ))
    
    fig.update_layout(
        title='Projected Workload Timeline',
        xaxis_title='Hour of Day',
        yaxis_title='Relative Workload'
    )
    return fig
