from ai_assistant import AIAssistant
from datetime import datetime
import pandas as pd

class ScenarioAdvisor:
    def __init__(self):
        self.ai_assistant = AIAssistant()

    def get_optimization_advice(self, scenario_config, current_metrics):
        """Get AI-powered optimization advice for a scenario"""
        recommendations = self.ai_assistant.get_scenario_advice(
            scenario_config, current_metrics
        )

        if "error" in recommendations:
            return {
                "status": "error",
                "message": recommendations["error"],
                "recommendations": []
            }

        # Format recommendations in natural language and include configuration
        formatted_recommendations = []
        for rec in recommendations.get("recommendations", []):
            if isinstance(rec, dict):
                # Extract configuration for quick apply
                config = self._extract_config_from_recommendation(rec)

                # Calibrate and normalize impact values
                raw_impact = rec.get('expected_impact', {})
                calibrated_impact = {
                    'efficiency': min(max(raw_impact.get('efficiency', 0), -25), 25),  # Cap at ±25%
                    'cognitive_load': min(max(raw_impact.get('cognitive_load', 0), -30), 0),  # Negative values, max 30% reduction
                    'burnout_risk': min(max(raw_impact.get('burnout_risk', 0), -35), 0)  # Negative values, max 35% reduction
                }
                
                formatted_rec = {
                    "title": rec.get('title', 'Recommendation'),
                    "description": rec.get('description', ''),
                    "risk_factors": rec.get('risk_factors', []),
                    "config": config,  # Configuration for quick apply
                    "impact": calibrated_impact
                }
                formatted_recommendations.append(formatted_rec)
            else:
                # Handle plain text recommendations with default structure
                formatted_recommendations.append({
                    "title": "Recommendation",
                    "description": str(rec),
                    "risk_factors": [],
                    "config": {},
                    "impact": {
                        'efficiency': 0,
                        'cognitive_load': 0,
                        'burnout_risk': 0
                    }
                })

        return {
            "status": "success",
            "recommendations": formatted_recommendations,
            # Calculate average impact across all recommendations
            "impact_analysis": {
                "efficiency": sum(r.get('impact', {}).get('efficiency', 0) 
                                for r in formatted_recommendations) / len(formatted_recommendations),
                "cognitive_load": sum(r.get('impact', {}).get('cognitive_load', 0) 
                                    for r in formatted_recommendations) / len(formatted_recommendations),
                "burnout_risk": sum(r.get('impact', {}).get('burnout_risk', 0) 
                                  for r in formatted_recommendations) / len(formatted_recommendations)
            },
            "priority": recommendations.get("priority", "medium"),
            "confidence": recommendations.get("confidence", 0.0)
        }

    def _extract_config_from_recommendation(self, recommendation):
        """Extract configuration parameters from AI recommendation"""
        config = {}
        impact = recommendation.get('expected_impact', {})
        
        # Extract values directly from recommendation if available
        if 'protected_time' in recommendation.get('config', {}):
            pt_config = recommendation['config']['protected_time']
            config['protected_time'] = {
                'start_hour': pt_config.get('start_hour', 9),
                'duration': pt_config.get('duration', 2),
            }

        if 'staff_distribution' in recommendation.get('config', {}):
            staff_config = recommendation['config']['staff_distribution']
            config['staff_distribution'] = {
                'add_physician': staff_config.get('add_physician', False),
                'physician_start': staff_config.get('physician_start', 8),
                'physician_duration': staff_config.get('physician_duration', 4),
                'add_app': staff_config.get('add_app', False),
                'app_start': staff_config.get('app_start', 8),
                'app_duration': staff_config.get('app_duration', 4)
            }

        if 'task_bundling' in recommendation.get('config', {}):
            bundle_config = recommendation['config']['task_bundling']
            config['task_bundling'] = {
                'efficiency_factor': bundle_config.get('efficiency_factor', 0.2)
            }

        return config

    def analyze_intervention_strategy(self, scenario_name, intervention_config):
        """Analyze the potential impact of intervention strategies"""
        analysis = self.ai_assistant.analyze_intervention_impact(intervention_config)

        if "error" in analysis:
            return {
                "status": "error",
                "message": analysis["error"],
                "analysis": {}
            }

        return {
            "status": "success",
            "scenario_name": scenario_name,
            "analysis": {
                "impact_scores": analysis.get("impact", {}),
                "complexity": analysis.get("complexity", 3),
                "roi_factors": analysis.get("roi_factors", {}),
                "risks": analysis.get("risks", []),
                "mitigations": analysis.get("mitigations", [])
            },
            "confidence": analysis.get("confidence", 0.0)
        }

    def generate_intervention_suggestions(self, current_metrics, historical_data=None):
        """Generate intervention suggestions based on current metrics"""
        metrics_data = {
            "current_state": current_metrics
        }

        if historical_data is not None:
            metrics_data["historical_trends"] = historical_data

        # Get AI recommendations for interventions
        recommendations = self.ai_assistant.get_scenario_advice(
            {"metrics_data": metrics_data},
            current_metrics
        )

        return {
            "suggested_interventions": recommendations.get("suggestions", []),
            "priority_areas": recommendations.get("priority_areas", []),
            "expected_outcomes": recommendations.get("expected_outcomes", {}),
            "confidence": recommendations.get("confidence", 0.0)
        }