import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from simulator import WorkflowSimulator
from models import get_db, get_scenarios, get_scenario_results

@dataclass
class ScenarioConfig:
    """Configuration for a workflow scenario"""
    name: str
    description: str
    base_config: Dict
    interventions: Dict
    created_at: datetime = datetime.now()

    # Intervention parameters
    protected_time_blocks: List[Dict] = None  # Time blocks where interruptions are reduced
    staff_distribution: Dict = None  # Provider distribution patterns
    task_bundling: Dict = None  # Task grouping strategies
    coverage_model: str = "standard"  # Type of coverage model

class ScenarioManager:
    def __init__(self, simulator: WorkflowSimulator):
        self.simulator = simulator
        self.scenarios: Dict[str, ScenarioConfig] = {}

    def create_scenario(self, name: str, description: str, base_config: Dict,
                       interventions: Optional[Dict] = None) -> ScenarioConfig:
        """Create a new scenario configuration"""
        if name in self.scenarios:
            raise ValueError(f"Scenario '{name}' already exists")

        scenario = ScenarioConfig(
            name=name,
            description=description,
            base_config=base_config,
            interventions=interventions or {}
        )
        self.scenarios[name] = scenario
        return scenario

    def run_scenario(self, scenario: ScenarioConfig) -> Dict:
        """Run a scenario and return the results"""
        # Store original simulator settings
        original_settings = {
            'interruption_times': self.simulator.interruption_times.copy(),
            'admission_times': self.simulator.admission_times.copy(),
            'critical_event_time': self.simulator.critical_event_time
        }

        try:
            # Apply scenario configurations
            self.simulator.update_time_settings(scenario.base_config)

            # Apply interventions if specified
            if scenario.interventions:
                self._apply_interventions(scenario.interventions)

            # Calculate metrics
            results = self._calculate_scenario_metrics(scenario)

            return {
                'scenario_name': scenario.name,
                'metrics': results,
                'timestamp': datetime.now(),
                'config': scenario.base_config,
                'interventions': scenario.interventions
            }

        finally:
            # Restore original settings
            self.simulator.update_time_settings(original_settings)

    def compare_scenarios(self, scenario_names: List[str]) -> pd.DataFrame:
        """Compare multiple scenarios and return analysis results"""
        results = []
        db = next(get_db())
        scenarios = get_scenarios(db)

        # Create a mapping of scenario names to their configurations
        scenario_map = {s.name: s for s in scenarios}

        for name in scenario_names:
            try:
                if name not in scenario_map:
                    raise ValueError(f"Scenario '{name}' not found in database")

                scenario_db = scenario_map[name]

                # Create ScenarioConfig from database record
                scenario = ScenarioConfig(
                    name=scenario_db.name,
                    description=scenario_db.description,
                    base_config=scenario_db.base_config,
                    interventions=scenario_db.interventions,
                    created_at=scenario_db.created_at
                )

                # Debug logging
                print(f"\nProcessing scenario: {name}")
                print(f"Protected time blocks: {scenario.interventions.get('protected_time_blocks', [])}")

                # Run scenario and get results
                scenario_result = self.run_scenario(scenario)

                # Debug logging for metrics
                print(f"Calculated metrics for {name}:")
                print(f"Efficiency: {scenario_result['metrics'].get('efficiency', 0):.3f}")
                print(f"Cognitive Load: {scenario_result['metrics'].get('cognitive_load', 0):.3f}")
                print(f"Burnout Risk: {scenario_result['metrics'].get('burnout_risk', 0):.3f}")
                print(f"Intervention Effectiveness: {scenario_result['metrics'].get('intervention_effectiveness', {})}")

                results.append(scenario_result)

            except Exception as e:
                print(f"Error processing scenario {name}: {str(e)}")
                results.append({
                    'scenario_name': name,
                    'metrics': {
                        'error': str(e),
                        'efficiency': 0,
                        'cognitive_load': 0,
                        'burnout_risk': 0,
                        'intervention_effectiveness': {
                            'protected_time': 0.0,
                            'staff_distribution': 0.0,
                            'task_bundling': 0.0
                        }
                    },
                    'timestamp': datetime.now(),
                    'config': {},
                    'interventions': {}
                })

        return pd.DataFrame(results)


    def _apply_interventions(self, interventions: Dict):
        """Apply intervention strategies to the simulator"""
        if 'protected_time_blocks' in interventions:
            self._apply_protected_time_blocks(interventions['protected_time_blocks'])

        if 'staff_distribution' in interventions:
            self._apply_staff_distribution(interventions['staff_distribution'])

        if 'task_bundling' in interventions:
            self._apply_task_bundling(interventions['task_bundling'])

    def _apply_protected_time_blocks(self, blocks: List[Dict]):
        """Apply protected time blocks to reduce interruptions"""
        if not blocks:
            return

        for block in blocks:
            if not block:  # Skip None blocks
                continue

            start_hour = block.get('start_hour', 0)
            end_hour = block.get('end_hour', 0)
            block_hours = end_hour - start_hour

            # Calculate the effectiveness based on time of day
            # Morning blocks (8-12) are most effective for reducing interruptions
            time_of_day_factor = 1.2 if 8 <= start_hour <= 12 else 1.0

            # Longer blocks are slightly less effective per hour
            duration_factor = 1.0 if block_hours <= 3 else 0.9

            reduction_factor = block.get('reduction_factor', 0.5) * time_of_day_factor * duration_factor

            # Only modify nursing questions and floor consults during protected time
            protected_interruptions = ['nursing_question']
            for key in protected_interruptions:
                if key in self.simulator.interruption_scales:
                    self.simulator.interruption_scales[key] *= reduction_factor

            # Log the changes
            print(f"Protected time block: {start_hour}:00 - {end_hour}:00")
            print(f"Reduction factor: {reduction_factor:.2f}")
            print("Updated interruption scales:", self.simulator.interruption_scales)

    def _apply_staff_distribution(self, distribution: Dict):
        """Apply staff distribution patterns"""
        if 'physician_ratio' in distribution:
            self.simulator.provider_ratios = {
                'physician': distribution['physician_ratio'],
                'app': 1 - distribution['physician_ratio']
            }

    def _apply_task_bundling(self, bundling: Dict):
        """Apply task bundling strategies"""
        if 'efficiency_factor' in bundling:
            factor = bundling['efficiency_factor']
            for key in self.simulator.admission_times:
                self.simulator.admission_times[key] *= factor

    def _calculate_scenario_metrics(self, scenario: ScenarioConfig) -> Dict:
        """Calculate comprehensive metrics for scenario analysis"""
        # Calculate base efficiency impact from protected time blocks
        protected_time_impact = self._calculate_protected_time_impact(scenario.interventions)

        # Calculate base efficiency without impact
        base_efficiency = self.simulator.simulate_provider_efficiency(
            sum(self.simulator.interruption_scales.values()),
            scenario.base_config.get('providers', 1),
            scenario.base_config.get('workload', 0.0),
            scenario.base_config.get('critical_events_per_day', 0),
            scenario.base_config.get('admissions', 0),
            scenario.base_config.get('adc', 0)
        )

        # Calculate base cognitive load without impact
        base_cognitive_load = self.simulator.calculate_cognitive_load(
            sum(self.simulator.interruption_scales.values()),
            scenario.base_config.get('critical_events_per_day', 0),
            scenario.base_config.get('admissions', 0),
            scenario.base_config.get('workload', 0.0)
        )

        # Calculate base burnout risk without impact
        base_burnout_risk = self.simulator.calculate_burnout_risk(
            scenario.base_config.get('workload', 0.0),
            sum(self.simulator.interruption_scales.values()),
            scenario.base_config.get('critical_events_per_day', 0)
        )

        # Apply protected time impacts
        base_metrics = {
            'efficiency': min(1.0, base_efficiency * (1 + protected_time_impact['efficiency_boost'])),
            'cognitive_load': max(0.0, min(1.0, base_cognitive_load * (1 - protected_time_impact['cognitive_reduction']))),
            'burnout_risk': max(0.0, min(1.0, base_burnout_risk * (1 - protected_time_impact['burnout_reduction'])))
        }

        # Calculate intervention effectiveness
        if scenario.interventions:
            start_hour = scenario.interventions.get('protected_time_blocks', [{}])[0].get('start_hour', 0)
            block_hours = scenario.interventions.get('protected_time_blocks', [{}])[0].get('end_hour', 0) - start_hour

            # Calculate effectiveness based on time of day
            if 8 <= start_hour <= 11:
                time_effectiveness = 0.8  # Most effective in morning
            elif 11 < start_hour <= 14:
                time_effectiveness = 0.6  # Moderately effective mid-day
            else:
                time_effectiveness = 0.4  # Less effective in afternoon

            # Apply block duration factor
            duration_factor = 1.0 if block_hours <= 3 else (3 / block_hours) ** 0.5

            intervention_effectiveness = {
                'protected_time': time_effectiveness * duration_factor,
                'staff_distribution': 0.0,  # Not implemented yet
                'task_bundling': 0.0  # Not implemented yet
            }

            base_metrics['intervention_effectiveness'] = intervention_effectiveness

        return base_metrics

    def _calculate_protected_time_impact(self, interventions: Dict) -> Dict:
        """Calculate the impact of protected time blocks based on time of day"""
        impact = {
            'efficiency_boost': 0.0,
            'cognitive_reduction': 0.0,
            'burnout_reduction': 0.0
        }

        if not interventions or 'protected_time_blocks' not in interventions:
            return impact

        blocks = interventions['protected_time_blocks']
        if not blocks:
            return impact

        for block in blocks:
            if not block:
                continue

            start_hour = block.get('start_hour', 0)
            end_hour = block.get('end_hour', 0)
            block_hours = end_hour - start_hour

            # Calculate time-of-day impact factors
            if 8 <= start_hour <= 11:  # Early morning (optimal)
                time_factor = 1.2
                cognitive_factor = 0.25
                burnout_factor = 0.3
            elif 11 < start_hour <= 14:  # Mid-day (moderate)
                time_factor = 1.0
                cognitive_factor = 0.2
                burnout_factor = 0.25
            else:  # Afternoon (less effective)
                time_factor = 0.8
                cognitive_factor = 0.15
                burnout_factor = 0.2

            # Scale impact based on proportion of interrupted activities being protected
            # Only nursing questions and floor consults are protected
            protected_activity_ratio = 0.3  # Estimate: 30% of interruptions are nursing questions and consults

            # Calculate impacts with scaled effectiveness
            efficiency_impact = min(0.3, block_hours * 0.05 * time_factor) * protected_activity_ratio
            cognitive_impact = min(0.25, block_hours * cognitive_factor) * protected_activity_ratio
            burnout_impact = min(0.3, block_hours * burnout_factor) * protected_activity_ratio

            # Update impact values
            impact['efficiency_boost'] = max(impact['efficiency_boost'], efficiency_impact)
            impact['cognitive_reduction'] = max(impact['cognitive_reduction'], cognitive_impact)
            impact['burnout_reduction'] = max(impact['burnout_reduction'], burnout_impact)

            # Log the impact calculations
            print(f"\nProtected time impact calculations for block {start_hour}:00 - {end_hour}:00")
            print(f"Time factor: {time_factor:.2f}")
            print(f"Efficiency impact: {efficiency_impact:.3f}")
            print(f"Cognitive impact: {cognitive_impact:.3f}")
            print(f"Burnout impact: {burnout_impact:.3f}")

        return impact

    def _calculate_intervention_metrics(self, scenario: ScenarioConfig) -> Dict:
        """Calculate metrics specific to interventions"""
        metrics = {
            'intervention_effectiveness': {
                'protected_time': 0.0,
                'staff_distribution': 0.0,
                'task_bundling': 0.0
            }
        }

        if scenario.interventions.get('protected_time_blocks'):
            blocks = scenario.interventions['protected_time_blocks']
            protected_time_effectiveness = 0.0

            for block in blocks:
                if block is None:
                    continue

                start_hour = block.get('start_hour', 0)
                end_hour = block.get('end_hour', 0)
                block_hours = end_hour - start_hour

                # Calculate base effectiveness
                base_effectiveness = min(0.8, block_hours * 0.1)

                # Time of day factors
                if 8 <= start_hour <= 11:  # Early morning is most effective
                    time_factor = 1.2
                elif 11 < start_hour <= 14:  # Mid-day is moderately effective
                    time_factor = 1.0
                else:  # Afternoon is less effective
                    time_factor = 0.8

                # Duration factors - longer blocks are less efficient per hour
                duration_factor = 1.0 if block_hours <= 3 else 0.9

                block_effectiveness = base_effectiveness * time_factor * duration_factor
                protected_time_effectiveness = max(protected_time_effectiveness, block_effectiveness)

            metrics['intervention_effectiveness']['protected_time'] = protected_time_effectiveness

        return metrics

    def _calculate_risk_assessment(self, scenario: ScenarioConfig, metrics: Dict) -> Dict:
        """Calculate risk assessment based on scenario configuration and metrics"""
        base_risk = 0.5

        if 'protected_time_blocks' in scenario.interventions:
            blocks = scenario.interventions['protected_time_blocks']
            total_impact = 0.0

            for block in blocks:
                if block is None:
                    continue

                start_hour = block.get('start_hour', 0)
                end_hour = block.get('end_hour', 0)
                block_hours = end_hour - start_hour

                # Calculate time-of-day impact on risk
                if 8 <= start_hour <= 11:
                    time_impact = 0.15  # Early morning has highest positive impact
                elif 11 < start_hour <= 14:
                    time_impact = 0.10  # Mid-day has moderate impact
                else:
                    time_impact = 0.05  # Afternoon has least impact

                total_impact += time_impact * (1.0 if block_hours <= 3 else 0.9)

            workflow_disruption = max(0.2, base_risk - total_impact)
            implementation_risk = min(0.8, base_risk + (total_impact * 0.5))
        else:
            workflow_disruption = base_risk
            implementation_risk = base_risk

        return {
            'workflow_disruption': workflow_disruption,
            'provider_burnout': max(0.2, metrics['burnout_risk'] - total_impact),
            'patient_care_impact': min(0.8, (1 - metrics['efficiency']) + 0.2),
            'resource_utilization': max(0.3, base_risk - (total_impact * 0.5)),
            'implementation_risk': implementation_risk
        }

    def export_scenario_analysis(self, scenario_names: List[str], format: str = 'csv') -> pd.DataFrame:
        """Export scenario analysis results"""
        comparison_results = self.compare_scenarios(scenario_names)

        if format == 'csv':
            return comparison_results
        # Add support for other export formats as needed

        return comparison_results