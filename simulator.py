import numpy as np

class WorkflowSimulator:
    def __init__(self):
        self.base_time_per_task = {
            'charting': 20,  # minutes
            'rounds': 15,
            'exams': 30,
            'critical_event': 120
        }

    def simulate_provider_efficiency(self, interruptions_per_hour, providers, workload, shift_hours=12):
        base_efficiency = 1.0
        interruption_impact = 0.05  # 5% efficiency loss per interruption
        workload_impact = 0.1  # 10% efficiency loss per unit of workload above baseline

        total_interruptions = interruptions_per_hour * shift_hours
        efficiency_loss = (total_interruptions * interruption_impact) + (max(0, workload - 1.0) * workload_impact)

        return max(0.3, base_efficiency - efficiency_loss)  # Minimum 30% efficiency

    def calculate_burnout_risk(self, workload_per_provider, interruptions_per_hour, critical_events_per_day):
        # Scale from 0-1, where >0.7 is high risk
        interruption_factor = interruptions_per_hour * 0.03  # 3% per interruption/hour
        workload_factor = workload_per_provider * 0.1  # 10% per unit of workload
        critical_factor = critical_events_per_day * 0.15  # 15% per critical event per day

        base_risk = min(1.0, interruption_factor + workload_factor + critical_factor)
        return base_risk

    def project_delays(self, tasks_per_shift, providers, efficiency):
        base_completion_time = sum(self.base_time_per_task.values())
        actual_completion_time = base_completion_time / efficiency

        delay = actual_completion_time - base_completion_time
        return max(0, delay)

    def calculate_cognitive_load(self, interruptions, critical_events_per_day, admissions, workload):
        # Scale from 0-100
        base_load = 30  # baseline cognitive load
        interrupt_factor = interruptions * 2
        critical_factor = critical_events_per_day * 15
        admission_factor = admissions * 5
        workload_factor = max(0, (workload - 1.0) * 20)  # Additional load for high workload

        total_load = base_load + interrupt_factor + critical_factor + admission_factor + workload_factor
        return min(100, total_load)