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

                # Run scenario and get results
                scenario_result = self.run_scenario(scenario)

                # Add additional metrics from database
                db_results = get_scenario_results(db, scenario_db.id)
                if db_results:
                    latest_result = db_results[0]
                    scenario_result['metrics'].update({
                        'efficiency': latest_result.efficiency,
                        'cognitive_load': latest_result.cognitive_load,
                        'burnout_risk': latest_result.burnout_risk,
                        'intervention_effectiveness': latest_result.intervention_effectiveness,
                        'risk_assessment': latest_result.statistical_significance,
                        'direct_care_time': 25,  # Default values for time distribution
                        'interruption_time': 15,
                        'critical_time': 20,
                        'admin_time': 40
                    })

                results.append(scenario_result)

            except Exception as e:
                results.append({
                    'scenario_name': name,
                    'metrics': {
                        'error': str(e),
                        'efficiency': 0,
                        'cognitive_load': 0,
                        'burnout_risk': 0
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
        total_protected_hours = 0
        for block in blocks:
            start_hour = block.get('start_hour', 0)
            end_hour = block.get('end_hour', 0)
            block_hours = end_hour - start_hour
            total_protected_hours += block_hours

            # Calculate the effectiveness based on time of day
            # Morning blocks (8-12) are most effective for reducing interruptions
            time_of_day_factor = 1.2 if 8 <= start_hour <= 12 else 1.0

            # Longer blocks are slightly less effective per hour
            duration_factor = 1.0 if block_hours <= 3 else 0.9

            reduction_factor = block.get('reduction_factor', 0.5) * time_of_day_factor * duration_factor

            # Adjust interruption frequencies during protected time
            for key in self.simulator.interruption_scales:
                self.simulator.interruption_scales[key] *= reduction_factor

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
        base_metrics = {
            'efficiency': self.simulator.simulate_provider_efficiency(
                sum(self.simulator.interruption_scales.values()),
                scenario.base_config.get('providers', 1),
                scenario.base_config.get('workload', 0.0),
                scenario.base_config.get('critical_events_per_day', 0),
                scenario.base_config.get('admissions', 0),
                scenario.base_config.get('adc', 0)
            ),
            'cognitive_load': self.simulator.calculate_cognitive_load(
                sum(self.simulator.interruption_scales.values()),
                scenario.base_config.get('critical_events_per_day', 0),
                scenario.base_config.get('admissions', 0),
                scenario.base_config.get('workload', 0.0)
            ),
            'burnout_risk': self.simulator.calculate_burnout_risk(
                scenario.base_config.get('workload', 0.0),
                sum(self.simulator.interruption_scales.values()),
                scenario.base_config.get('critical_events_per_day', 0)
            )
        }

        # Calculate intervention-specific metrics
        if scenario.interventions:
            intervention_metrics = self._calculate_intervention_metrics(scenario)

            # Calculate time distribution based on interventions
            time_distribution = self._calculate_time_distribution(scenario)

            base_metrics.update(intervention_metrics)
            base_metrics.update(time_distribution)

            # Calculate risk assessment
            risk_assessment = self._calculate_risk_assessment(scenario, base_metrics)
            base_metrics['risk_assessment'] = risk_assessment

        return base_metrics

    def _calculate_intervention_metrics(self, scenario: ScenarioConfig) -> Dict:
        """Calculate metrics specific to interventions"""
        metrics = {}

        if scenario.protected_time_blocks:
            metrics['protected_time_efficiency'] = self._calculate_protected_time_efficiency(
                scenario.protected_time_blocks
            )

        if scenario.staff_distribution:
            metrics['staff_distribution_impact'] = self._calculate_staff_distribution_impact(
                scenario.staff_distribution
            )

        if scenario.task_bundling:
            metrics['task_bundling_efficiency'] = self._calculate_task_bundling_efficiency(
                scenario.task_bundling
            )

        return metrics

    def _calculate_protected_time_efficiency(self, blocks: List[Dict]) -> float:
        """Calculate efficiency improvement from protected time blocks"""
        # Implementation for protected time efficiency calculation
        total_protected_hours = sum(
            block['end_hour'] - block['start_hour']
            for block in blocks
        )
        return min(1.0, 1.0 + (total_protected_hours * 0.02))  # 2% improvement per protected hour

    def _calculate_staff_distribution_impact(self, distribution: Dict) -> float:
        """Calculate impact of staff distribution changes"""
        # Implementation for staff distribution impact calculation
        return distribution.get('efficiency_factor', 1.0)

    def _calculate_task_bundling_efficiency(self, bundling: Dict) -> float:
        """Calculate efficiency gains from task bundling"""
        # Implementation for task bundling efficiency calculation
        return bundling.get('efficiency_factor', 1.0)

    def _calculate_time_distribution(self, scenario: ScenarioConfig) -> Dict:
        """Calculate time distribution based on interventions"""
        # Start with base distribution
        distribution = {
            'direct_care_time': 30,  # Base percentage
            'interruption_time': 20,
            'critical_time': 20,
            'admin_time': 30
        }

        if 'protected_time_blocks' in scenario.interventions:
            blocks = scenario.interventions['protected_time_blocks']
            total_protected_hours = sum(
                block['end_hour'] - block['start_hour']
                for block in blocks if block is not None
            )

            # Adjust distribution based on protected time
            interruption_reduction = min(total_protected_hours * 2, 10)  # Max 10% reduction
            distribution['interruption_time'] = max(10, distribution['interruption_time'] - interruption_reduction)
            distribution['direct_care_time'] += interruption_reduction

        return distribution

    def _calculate_risk_assessment(self, scenario: ScenarioConfig, metrics: Dict) -> Dict:
        """Calculate risk assessment based on scenario configuration and metrics"""
        base_risk = 0.5  # Default risk level

        # Calculate workflow disruption risk
        if 'protected_time_blocks' in scenario.interventions:
            blocks = scenario.interventions['protected_time_blocks']
            total_protected_hours = sum(
                block['end_hour'] - block['start_hour']
                for block in blocks if block is not None
            )

            # More protected time reduces workflow disruption risk
            workflow_disruption = max(0.2, base_risk - (total_protected_hours * 0.05))

            # But very long protected blocks might increase other risks
            implementation_risk = min(0.8, base_risk + (total_protected_hours * 0.03))
        else:
            workflow_disruption = base_risk
            implementation_risk = base_risk

        return {
            'workflow_disruption': workflow_disruption,
            'provider_burnout': max(0.2, metrics['burnout_risk'] - 0.1),
            'patient_care_impact': min(0.8, (1 - metrics['efficiency']) + 0.2),
            'resource_utilization': base_risk,
            'implementation_risk': implementation_risk
        }

    def export_scenario_analysis(self, scenario_names: List[str], format: str = 'csv') -> pd.DataFrame:
        """Export scenario analysis results"""
        comparison_results = self.compare_scenarios(scenario_names)

        if format == 'csv':
            return comparison_results
        # Add support for other export formats as needed

        return comparison_results