import numpy as np

class WorkflowSimulator:
    def __init__(self):
        # Default time durations in minutes
        self.interruption_times = {
            'nursing_question': 2,    # median of 1-3 minutes
            'exam_callback': 7.5,     # median of 5-10 minutes
            'peer_interrupt': 7.5,    # median of 5-10 minutes
            'transfer_call': 8.0      # median transfer call duration
        }

        # Interruptions per hour per patient
        self.interruption_scales = {
            'nursing_question': 0.36,
            'exam_callback': 0.21,
            'peer_interrupt': 0.14,
            'transfer_call': 0.10     # default transfer call rate
        }

        self.admission_times = {
            'simple': 60,     # simple admission: 60 mins
            'complex': 90,    # complex admission: 90 mins
            'consult': 45,    # floor consult average time
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

    def calculate_individual_interruption_time(self, nursing_q, exam_callbacks, peer_interrupts, transfer_calls):
        """Calculate interruption time for a single provider during a 12-hour shift"""
        hourly_time = (
            nursing_q * self.interruption_times['nursing_question'] +
            exam_callbacks * self.interruption_times['exam_callback'] +
            peer_interrupts * self.interruption_times['peer_interrupt'] +
            transfer_calls * self.interruption_times['transfer_call']
        )
        return hourly_time * 12

    def calculate_role_specific_interruption_time(self, nursing_q, exam_callbacks, peer_interrupts, transfer_calls, role='physician'):
        """Calculate interruption time specific to provider role"""
        # Physicians handle all types of interruptions
        # APPs handle all except transfer calls (handled by physicians)
        if role == 'app':
            transfer_calls = 0  # APPs don't handle transfer calls

        return self.calculate_individual_interruption_time(
            nursing_q, exam_callbacks, peer_interrupts, transfer_calls
        )

    def calculate_total_interruption_time(self, nursing_q, exam_callbacks, peer_interrupts, transfer_calls, providers):
        """Calculate total organizational interruption time during a 12-hour shift"""
        individual_time = self.calculate_individual_interruption_time(
            nursing_q, exam_callbacks, peer_interrupts, transfer_calls
        )
        return individual_time * providers  # Total organizational impact

    def calculate_time_impact(self, nursing_q, exam_callbacks, peer_interrupts, transfer_calls,
                            admissions, consults, critical_events_per_day, providers):
        """Calculate time impacts for different activities during a shift"""
        interrupt_time = self.calculate_total_interruption_time(
            nursing_q, exam_callbacks, peer_interrupts, transfer_calls, providers
        )

        admission_time = (
            admissions * (0.7 * self.admission_times['simple'] + 0.3 * self.admission_times['complex']) +
            consults * self.admission_times['consult']
        )

        critical_time = critical_events_per_day * self.critical_event_time

        return interrupt_time, admission_time, critical_time

    def simulate_provider_efficiency(self, interruptions_per_hour, providers, workload,
                                   critical_events_per_day, admissions, adc, role='physician'):
        """Calculate provider efficiency considering role-specific duties"""
        np.random.seed(None)
        shift_minutes = 12 * 60

        # Distribute events across shift
        admission_times = sorted(np.random.choice(shift_minutes, size=admissions, replace=False))
        critical_times = sorted(np.random.choice(shift_minutes, size=int(critical_events_per_day), replace=False))

        # Track provider availability minute by minute
        available_providers = np.ones(shift_minutes)

        # Process role-specific events
        if role == 'physician':
            # Process consults (physician only, 8am-5pm)
            consult_window_start = 8 * 60
            consult_window_end = 17 * 60
            consult_impact = np.zeros(shift_minutes)
            consult_impact[consult_window_start:consult_window_end] = 0.8  # 80% impact during consult window
            available_providers *= (1 - consult_impact)

        # Process shared responsibilities
        for start_time in admission_times:
            end_time = min(start_time + self.admission_times['complex'], shift_minutes)
            available_providers[start_time:end_time] *= 0.5  # 50% availability during admissions

        # Process critical events (both providers initially, then one)
        for start_time in critical_times:
            first_hour_end = min(start_time + 60, shift_minutes)
            available_providers[start_time:first_hour_end] = 0  # Both unavailable

            second_phase_end = min(start_time + self.critical_event_time, shift_minutes)
            if first_hour_end < second_phase_end:
                available_providers[first_hour_end:second_phase_end] *= 0.5  # One provider returns

        # Calculate average availability
        avg_availability = np.mean(available_providers)

        # Base efficiency calculation with role-specific adjustments
        base_efficiency = min(1.0, 1.2 - (adc / providers * 0.15))

        # Role-specific interruption impact
        if role == 'physician':
            interruption_impact = interruptions_per_hour * 0.025  # 2.5% per interruption/hour for physicians
        else:  # APP
            interruption_impact = interruptions_per_hour * 0.02   # 2% per interruption/hour for APPs

        # Final efficiency calculation
        efficiency = base_efficiency * avg_availability * (1 - interruption_impact)

        return max(0.3, efficiency)  # Minimum efficiency of 30%

    def calculate_burnout_risk(self, workload_per_provider, interruptions_per_hour, critical_events_per_day, role='physician'):
        """Calculate role-specific burnout risk metric"""
        if workload_per_provider == 0 and interruptions_per_hour == 0 and critical_events_per_day == 0:
            return 0.0

        # Calculate risk components with role-specific weights
        interruption_factor = interruptions_per_hour * (0.035 if role == 'physician' else 0.03)
        workload_factor = workload_per_provider * 0.1
        critical_factor = critical_events_per_day * 0.15

        # Role-specific rounding impact
        rounding_impact = 0
        if workload_per_provider > 0:
            rounding_overhead = 0.8
            data_collection_inefficiency = 0.3
            rounding_impact = (rounding_overhead + data_collection_inefficiency) * (0.3 if role == 'physician' else 0.2)

        # Use role-specific weighting
        if role == 'physician':
            weights = {
                'interruption': 0.25,
                'workload': 0.3,
                'critical': 0.2,
                'rounding': 0.25
            }
        else:  # APP
            weights = {
                'interruption': 0.2,
                'workload': 0.35,
                'critical': 0.25,
                'rounding': 0.2
            }

        base_risk = min(1.0, 
            (interruption_factor * weights['interruption']) +
            (workload_factor * weights['workload']) +
            (critical_factor * weights['critical']) +
            (rounding_impact * weights['rounding'])
        )

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
        # If there's no work, cognitive load should be 0
        if workload == 0 and critical_events_per_day == 0 and admissions == 0 and interruptions == 0:
            return 0

        base_load = 30 if workload > 0 else 0  # baseline cognitive load only applies if there's work

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