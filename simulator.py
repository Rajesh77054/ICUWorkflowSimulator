import numpy as np

class WorkflowSimulator:
    def __init__(self):
        # Time durations in minutes
        self.interruption_times = {
            'nursing_question': 2,  # average of 1-3 minutes
            'exam_callback': 7.5,   # average of 5-10 minutes
            'peer_interrupt': 7.5   # average of 5-10 minutes
        }

        self.admission_times = {
            'simple': 60,    # simple admission: 60 mins
            'complex': 90,   # complex admission: 90 mins
            'consult': 45,   # floor consult average time
            'transfer': 30   # transfer call average time
        }

        self.critical_event_time = 105  # average of 90-120 minutes

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

    def calculate_time_impact(self, nursing_q, exam_callbacks, peer_interrupts, 
                            admissions, consults, transfers, critical_events_per_day):
        # Calculate total time spent on interruptions
        interrupt_time = (
            nursing_q * self.interruption_times['nursing_question'] +
            exam_callbacks * self.interruption_times['exam_callback'] +
            peer_interrupts * self.interruption_times['peer_interrupt']
        )

        # Calculate time spent on admissions and transfers (assume 70% simple, 30% complex)
        admission_time = (
            admissions * (0.7 * self.admission_times['simple'] + 0.3 * self.admission_times['complex']) +
            consults * self.admission_times['consult'] +
            transfers * self.admission_times['transfer']
        )

        # Calculate time spent on critical events
        critical_time = critical_events_per_day * self.critical_event_time

        return interrupt_time, admission_time, critical_time

    def calculate_cognitive_load(self, interruptions, critical_events_per_day, admissions, workload):
        # Scale from 0-100
        base_load = 30  # baseline cognitive load
        interrupt_factor = interruptions * 2
        critical_factor = critical_events_per_day * 15
        admission_factor = admissions * 5
        workload_factor = max(0, (workload - 1.0) * 20)  # Additional load for high workload

        total_load = base_load + interrupt_factor + critical_factor + admission_factor + workload_factor
        return min(100, total_load)