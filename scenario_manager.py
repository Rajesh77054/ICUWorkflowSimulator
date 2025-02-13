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
            'interruption_scales': self.simulator.interruption_scales.copy(),
            'admission_times': self.simulator.admission_times.copy(),
            'critical_event_time': self.simulator.critical_event_time
        }

        try:
            # Calculate baseline metrics before applying interventions
            baseline_metrics = self._calculate_baseline_metrics(scenario.base_config)

            # Apply interventions if specified
            if scenario.interventions:
                self._apply_interventions(scenario.interventions)

            # Calculate post-intervention metrics
            intervention_metrics = self._calculate_intervention_metrics(scenario)

            # Calculate time distributions
            time_distribution = self._calculate_time_distribution(scenario, baseline_metrics)

            # Combine all metrics
            combined_metrics = {
                **baseline_metrics,
                **intervention_metrics,
                **time_distribution
            }

            return {
                'scenario_name': scenario.name,
                'metrics': combined_metrics,
                'timestamp': datetime.now(),
                'config': scenario.base_config,
                'interventions': scenario.interventions
            }

        finally:
            # Restore original settings
            self.simulator.interruption_scales = original_settings['interruption_scales']
            self.simulator.admission_times = original_settings['admission_times']
            self.simulator.critical_event_time = original_settings['critical_event_time']

    def _calculate_baseline_metrics(self, base_config: Dict) -> Dict:
        """Calculate baseline metrics before interventions"""
        total_interruptions = sum(self.simulator.interruption_scales.values())
        protected_interruptions = self.simulator.interruption_scales.get('nursing_question', 0)
        unprotected_interruptions = total_interruptions - protected_interruptions

        return {
            'baseline_efficiency': self.simulator.simulate_provider_efficiency(
                total_interruptions,
                base_config.get('providers', 1),
                base_config.get('workload', 0.0),
                base_config.get('critical_events_per_day', 0),
                base_config.get('admissions', 0),
                base_config.get('adc', 0)
            ),
            'baseline_protected_interruptions': protected_interruptions,
            'baseline_unprotected_interruptions': unprotected_interruptions,
        }

    def compare_scenarios(self, scenario_names: List[str]) -> pd.DataFrame:
        """Compare multiple scenarios and return analysis results"""
        results = []
        db = next(get_db())
        scenarios = get_scenarios(db)
        scenario_map = {s.name: s for s in scenarios}

        for name in scenario_names:
            try:
                if name not in scenario_map:
                    raise ValueError(f"Scenario '{name}' not found in database")

                scenario_db = scenario_map[name]
                scenario = ScenarioConfig(
                    name=scenario_db.name,
                    description=scenario_db.description,
                    base_config=scenario_db.base_config,
                    interventions=scenario_db.interventions,
                    created_at=scenario_db.created_at
                )

                print(f"\nProcessing scenario: {name}")
                print(f"Protected time blocks: {scenario.interventions.get('protected_time_blocks', [])}")

                scenario_result = self.run_scenario(scenario)

                print(f"Calculated metrics for {name}:")
                print(f"Efficiency: {scenario_result['metrics'].get('baseline_efficiency', 0):.3f}")
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
            if not block:
                continue

            start_hour = block.get('start_hour', 0)
            end_hour = block.get('end_hour', 0)
            block_hours = end_hour - start_hour

            # Time of day effectiveness
            if 8 <= start_hour <= 11:
                time_factor = 1.2  # Most effective in morning
            elif 11 < start_hour <= 14:
                time_factor = 1.0  # Moderate effectiveness mid-day
            else:
                time_factor = 0.8  # Less effective in afternoon

            # Duration factor (longer blocks have diminishing returns)
            duration_factor = 1.0 if block_hours <= 3 else 0.9

            # Calculate final reduction factor
            reduction_factor = block.get('reduction_factor', 0.5) * time_factor * duration_factor

            # Only modify nursing questions
            if 'nursing_question' in self.simulator.interruption_scales:
                self.simulator.interruption_scales['nursing_question'] *= reduction_factor

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

    def _calculate_intervention_metrics(self, scenario: ScenarioConfig) -> Dict:
        """Calculate comprehensive metrics for the scenario"""
        if not scenario.interventions or 'protected_time_blocks' not in scenario.interventions:
            return {}

        blocks = scenario.interventions['protected_time_blocks']
        total_impact = 0.0
        protected_time_effectiveness = 0.0

        for block in blocks:
            if not block:
                continue

            start_hour = block.get('start_hour', 0)
            end_hour = block.get('end_hour', 0)
            block_hours = end_hour - start_hour

            # Calculate time-of-day effectiveness
            if 8 <= start_hour <= 11:
                time_effectiveness = 0.8
            elif 11 < start_hour <= 14:
                time_effectiveness = 0.6
            else:
                time_effectiveness = 0.4

            # Calculate block effectiveness
            block_effectiveness = time_effectiveness * (1.0 if block_hours <= 3 else 0.9)
            protected_time_effectiveness = max(protected_time_effectiveness, block_effectiveness)

            # Calculate impact based on protected activities ratio
            protected_ratio = 0.3  # Estimate: 30% of interruptions are nursing questions and consults
            impact = block_effectiveness * protected_ratio
            total_impact = max(total_impact, impact)

        return {
            'intervention_effectiveness': {
                'protected_time': protected_time_effectiveness,
                'staff_distribution': 0.0,
                'task_bundling': 0.0
            },
            'total_impact': total_impact
        }

    def _calculate_time_distribution(self, scenario: ScenarioConfig, baseline_metrics: Dict) -> Dict:
        """Calculate time distribution based on interventions and baseline metrics"""
        protected_ratio = 0.3  # Ratio of interruptions that are nursing questions and consults

        # Start with base distribution
        distribution = {
            'direct_care_time': 40,
            'interruption_time': 30,
            'critical_time': 20,
            'admin_time': 10
        }

        if 'protected_time_blocks' in scenario.interventions:
            blocks = scenario.interventions['protected_time_blocks']
            if blocks:
                total_protected_hours = sum(
                    (block['end_hour'] - block['start_hour'])
                    for block in blocks if block is not None
                )

                # Calculate reduction in protected interruptions
                protected_reduction = min(
                    total_protected_hours * 2 * protected_ratio,
                    distribution['interruption_time'] * protected_ratio
                )

                # Update time distribution
                distribution['interruption_time'] -= protected_reduction
                distribution['direct_care_time'] += protected_reduction

        return {'time_distribution': distribution}

    def export_scenario_analysis(self, scenario_names: List[str], format: str = 'csv') -> pd.DataFrame:
        """Export scenario analysis results"""
        comparison_results = self.compare_scenarios(scenario_names)

        if format == 'csv':
            return comparison_results
        # Add support for other export formats as needed

        return comparison_results