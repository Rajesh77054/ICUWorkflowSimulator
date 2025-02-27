import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from simulator import WorkflowSimulator

@dataclass
class ScenarioConfig:
    """Configuration for a workflow scenario"""
    name: str
    description: str
    base_config: Dict
    interventions: Dict
    created_at: datetime = field(default_factory=datetime.now)
    # Intervention parameters with proper defaults
    protected_time_blocks: List[Dict] = field(default_factory=list)
    staff_distribution: Dict = field(default_factory=dict)
    task_bundling: Dict = field(default_factory=dict)
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
        for name in scenario_names:
            if name not in self.scenarios:
                raise ValueError(f"Scenario '{name}' not found")
            
            scenario_result = self.run_scenario(self.scenarios[name])
            results.append(scenario_result)
        
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
        for block in blocks:
            start_hour = block.get('start_hour', 0)
            end_hour = block.get('end_hour', 0)
            reduction_factor = block.get('reduction_factor', 0.5)
            
            # Adjust interruption frequencies during protected time
            for key in self.simulator.interruption_scales:
                self.simulator.interruption_scales[key] *= (
                    reduction_factor if start_hour <= datetime.now().hour < end_hour
                    else 1.0
                )
    
    def _apply_staff_distribution(self, distribution: Dict):
        """Apply staff distribution patterns"""
        # Adjust provider-specific parameters based on distribution
        if 'physician_ratio' in distribution:
            self.simulator.provider_ratios = {
                'physician': distribution['physician_ratio'],
                'app': 1 - distribution['physician_ratio']
            }
    
    def _apply_task_bundling(self, bundling: Dict):
        """Apply task bundling strategies"""
        # Adjust task durations based on bundling efficiency
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
            base_metrics.update(intervention_metrics)
        
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
    
    def export_scenario_analysis(self, scenario_names: List[str], format: str = 'csv') -> pd.DataFrame:
        """Export scenario analysis results"""
        comparison_results = self.compare_scenarios(scenario_names)
        
        if format == 'csv':
            return comparison_results
        # Add support for other export formats as needed
        
        return comparison_results