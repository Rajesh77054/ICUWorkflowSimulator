import numpy as np

class WorkflowSimulator:
    def __init__(self):
        self.base_time_per_task = {
            'charting': 20,  # minutes
            'rounds': 15,
            'exams': 30,
            'critical_event': 120
        }
    
    def simulate_provider_efficiency(self, interruptions_per_hour, providers, shift_hours=12):
        base_efficiency = 1.0
        interruption_impact = 0.05  # 5% efficiency loss per interruption
        
        total_interruptions = interruptions_per_hour * shift_hours
        efficiency_loss = total_interruptions * interruption_impact
        
        return max(0, base_efficiency - efficiency_loss)
    
    def calculate_burnout_risk(self, workload_per_provider, interruptions_per_hour):
        # Scale from 0-1, where >0.7 is high risk
        base_risk = min(1.0, (workload_per_provider / 10) + (interruptions_per_hour / 20))
        return base_risk
    
    def project_delays(self, tasks_per_shift, providers, efficiency):
        base_completion_time = sum(self.base_time_per_task.values())
        actual_completion_time = base_completion_time / efficiency
        
        delay = actual_completion_time - base_completion_time
        return max(0, delay)
    
    def calculate_cognitive_load(self, interruptions, critical_events, admissions):
        # Scale from 0-100
        base_load = 30  # baseline cognitive load
        interrupt_factor = interruptions * 2
        critical_factor = critical_events * 10
        admission_factor = admissions * 5
        
        total_load = base_load + interrupt_factor + critical_factor + admission_factor
        return min(100, total_load)
