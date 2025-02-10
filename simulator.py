import numpy as np

class WorkflowSimulator:
    def __init__(self):
        # Default time durations in minutes
        self.interruption_times = {
            'nursing_question': 2,    # median of 1-3 minutes
            'exam_callback': 7.5,     # median of 5-10 minutes
            'peer_interrupt': 7.5     # median of 5-10 minutes
        }
        
        # Interruptions per hour per patient
        self.interruption_scales = {
            'nursing_question': 0.36,
            'exam_callback': 0.21,
            'peer_interrupt': 0.14
        }

        self.admission_times = {
            'simple': 60,     # simple admission: 60 mins
            'complex': 90,    # complex admission: 90 mins
            'consult': 45,    # floor consult average time
            'transfer': 30    # transfer call average time
        }

        self.critical_event_time = 105  # median of 90-120 minutes

        # Burnout risk thresholds
        self.burnout_thresholds = {
            'low': 0.3,
            'moderate': 0.5,
            'high': 0.7,
            'severe': 0.85
        }

    def update_time_settings(self, new_settings):
        """Update time duration settings"""
        if 'interruption_times' in new_settings:
            self.interruption_times.update(new_settings['interruption_times'])
        if 'admission_times' in new_settings:
            self.admission_times.update(new_settings['admission_times'])
        if 'critical_event_time' in new_settings:
            self.critical_event_time = new_settings['critical_event_time']

    def calculate_individual_interruption_time(self, nursing_q, exam_callbacks, peer_interrupts):
        """Calculate interruption time for a single provider during a 12-hour shift
        Input frequencies are per hour per provider
        Returns: Total minutes lost to interruptions for one provider
        """
        hourly_time = (
            nursing_q * self.interruption_times['nursing_question'] +
            exam_callbacks * self.interruption_times['exam_callback'] +
            peer_interrupts * self.interruption_times['peer_interrupt']
        )
        return hourly_time * 12  # Convert to full shift duration

    def calculate_total_interruption_time(self, nursing_q, exam_callbacks, peer_interrupts, providers):
        """Calculate total organizational interruption time during a 12-hour shift
        Input frequencies are per hour per provider
        Returns: Total minutes lost to interruptions across all providers
        """
        individual_time = self.calculate_individual_interruption_time(
            nursing_q, exam_callbacks, peer_interrupts
        )
        return individual_time * providers  # Total organizational impact

    def calculate_time_impact(self, nursing_q, exam_callbacks, peer_interrupts,
                            admissions, consults, transfers, critical_events_per_day, providers):
        """Calculate time impacts for different activities during a shift"""
        # Calculate total organizational interruption time
        interrupt_time = self.calculate_total_interruption_time(
            nursing_q, exam_callbacks, peer_interrupts, providers
        )

        # Calculate admission and transfer time
        admission_time = (
            admissions * (0.7 * self.admission_times['simple'] + 0.3 * self.admission_times['complex']) +
            consults * self.admission_times['consult'] +
            transfers * self.admission_times['transfer']
        )

        # Calculate critical event time
        critical_time = critical_events_per_day * self.critical_event_time

        return interrupt_time, admission_time, critical_time

    def simulate_provider_efficiency(self, interruptions_per_hour, providers, workload,
                                   critical_events_per_day, adc, shift_hours=12):
        """Calculate provider efficiency considering all factors and parallel work capacity"""
        base_efficiency = 1.0
        interruption_impact = 0.05
        workload_impact = 0.1
        
        # Adjust base efficiency based on ADC per provider ratio
        patients_per_provider = adc / providers
        base_efficiency = min(1.0, 1.2 - (patients_per_provider * 0.15))  # Decrease efficiency as patients/provider increases

        # Account for rounding inefficiency (9-11 AM)
        rounding_hours = 2  # 9-11 AM
        rounding_overhead = 0.8  # 80% overhead during rounds
        data_collection_inefficiency = 0.3  # 30% inefficiency from repeated data collection
        rounding_impact = (rounding_overhead + data_collection_inefficiency) * (rounding_hours / shift_hours)

        # Calculate critical event impact on efficiency, accounting for parallel provider work
        critical_first_hour = min(60, self.critical_event_time)
        critical_remaining = max(0, self.critical_event_time - 60)

        # During first hour, both providers are unavailable (full impact)
        total_unavailable_time = critical_first_hour * critical_events_per_day
        # After first hour, one provider remains on critical event (half impact due to parallel capacity)
        partial_unavailable_time = critical_remaining * critical_events_per_day

        # Calculate effective shift time reduction accounting for parallel provider capacity
        shift_minutes = shift_hours * 60
        total_provider_minutes = shift_minutes * providers
        efficiency_reduction = (
            (total_unavailable_time * 2) +  # Both providers unavailable
            partial_unavailable_time        # One provider unavailable
        ) / total_provider_minutes

        total_interruptions = interruptions_per_hour * shift_hours
        regular_efficiency_loss = (total_interruptions * interruption_impact) + (max(0, workload - 1.0) * workload_impact)

        return max(0.3, base_efficiency - regular_efficiency_loss - efficiency_reduction - rounding_impact)

    def calculate_burnout_risk(self, workload_per_provider, interruptions_per_hour, critical_events_per_day):
        """Calculate simple burnout risk metric"""
        # Calculate risk components consistently with detailed calculation
        interruption_factor = interruptions_per_hour * 0.03  # 3% per interruption/hour
        workload_factor = workload_per_provider * 0.1  # 10% per unit of workload
        critical_factor = critical_events_per_day * 0.15  # 15% per critical event per day

        # Add consistent rounding inefficiency impact
        rounding_overhead = 0.8  # 80% overhead during rounds
        data_collection_inefficiency = 0.3  # 30% inefficiency
        rounding_impact = (rounding_overhead + data_collection_inefficiency) * 0.25  # Scale factor

        # Use same weighting as detailed calculation
        base_risk = min(1.0, (interruption_factor * 0.2) + (workload_factor * 0.25) +
                       (critical_factor * 0.2) + (rounding_impact * 0.35))
        return base_risk

    def calculate_detailed_burnout_risk(self, workload_per_provider, interruptions_per_hour,
                                      critical_events_per_day, efficiency, cognitive_load):
        """Calculate detailed burnout risk metrics"""
        # Base factors from previous calculation
        interruption_factor = interruptions_per_hour * 0.03  # 3% per interruption/hour
        workload_factor = workload_per_provider * 0.1  # 10% per unit of workload
        critical_factor = critical_events_per_day * 0.15  # 15% per critical event per day

        # Additional factors
        efficiency_impact = (1 - efficiency) * 0.5  # Impact of reduced efficiency
        cognitive_impact = (cognitive_load / 100) * 0.4  # Impact of cognitive load

        # Calculate rounding inefficiency impact
        rounding_overhead = 0.8  # 80% overhead during rounds
        data_collection_inefficiency = 0.3  # 30% inefficiency
        rounding_impact = (rounding_overhead + data_collection_inefficiency) * 0.25  # Scale factor

        # Calculate individual risk components
        risk_components = {
            "interruption_risk": min(1.0, interruption_factor),
            "workload_risk": min(1.0, workload_factor + rounding_impact),
            "critical_events_risk": min(1.0, critical_factor),
            "efficiency_risk": min(1.0, efficiency_impact),
            "cognitive_load_risk": min(1.0, cognitive_impact + rounding_impact * 0.5)
        }

        # Calculate weighted total risk
        weights = {
            "interruption_risk": 0.2,
            "workload_risk": 0.25,
            "critical_events_risk": 0.2,
            "efficiency_risk": 0.15,
            "cognitive_load_risk": 0.2
        }

        total_risk = sum(risk * weights[factor] for factor, risk in risk_components.items())

        # Determine risk category
        risk_category = "low"
        for category, threshold in self.burnout_thresholds.items():
            if total_risk >= threshold:
                risk_category = category

        return {
            "total_risk": total_risk,
            "risk_category": risk_category,
            "risk_components": risk_components,
            "component_weights": weights
        }

    def calculate_cognitive_load(self, interruptions, critical_events_per_day, admissions, workload):
        """Calculate cognitive load score (0-100)"""
        base_load = 30  # baseline cognitive load

        # Factor in time impact of interruptions using actual duration settings
        avg_interrupt_time = sum(self.interruption_times.values()) / len(self.interruption_times)
        interrupt_factor = interruptions * (avg_interrupt_time / 60)  # Convert to hours

        # Factor in time impact of critical events using configured duration
        critical_factor = critical_events_per_day * (self.critical_event_time / 60)  # normalized by hour

        # Factor in admission complexity using configured durations
        avg_admission_time = (self.admission_times['simple'] + self.admission_times['complex']) / 2
        admission_factor = admissions * (avg_admission_time / 60)  # normalized by hour

        # Additional load for high workload
        workload_factor = max(0, (workload - 1.0) * 20)

        # Scale factors to maintain reasonable cognitive load range
        interrupt_scale = 5   # 5 points per hour of interruptions
        critical_scale = 10   # 10 points per hour of critical events
        admission_scale = 8   # 8 points per hour of admissions

        total_load = (
            base_load +
            (interrupt_factor * interrupt_scale) +
            (critical_factor * critical_scale) +
            (admission_factor * admission_scale) +
            workload_factor
        )

        return min(100, total_load)