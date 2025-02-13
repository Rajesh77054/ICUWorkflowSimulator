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

                # Format structured recommendation with default values
                formatted_rec = {
                    "title": rec.get('suggestion', 'Recommendation'),
                    "description": rec.get('description', ''),
                    "risk_factors": rec.get('risk_factors', []),
                    "config": config,  # Configuration for quick apply
                    "impact": {
                        'efficiency': rec.get('expected_impact', {}).get('efficiency', 0),
                        'cognitive_load': rec.get('expected_impact', {}).get('cognitive_load', 0),
                        'burnout_risk': rec.get('expected_impact', {}).get('burnout_risk', 0)
                    }
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
            "impact_analysis": {
                "efficiency": recommendations.get("impact", {}).get("efficiency", 0) / 100,
                "cognitive_load": recommendations.get("impact", {}).get("cognitive_load", 0) / 100,
                "burnout_risk": recommendations.get("impact", {}).get("burnout_risk", 0) / 100
            },
            "priority": recommendations.get("priority", "medium"),
            "confidence": recommendations.get("confidence", 0.0)
        }

    def _extract_config_from_recommendation(self, recommendation):
        """Extract configuration parameters from AI recommendation"""
        config = {}

        suggestion = recommendation.get('suggestion', '').lower()
        description = recommendation.get('description', '').lower()

        # Map recommendation keywords to configuration parameters
        if 'protected time' in suggestion or 'protected time' in description:
            config['protected_time'] = {
                'start_hour': 9,  # Default to 9 AM
                'duration': 2,    # Default to 2 hours
            }

        if 'staff' in suggestion or 'staff distribution' in description:
            config['staff_distribution'] = {}
            if 'physician' in description:
                config['staff_distribution']['add_physician'] = True
                config['staff_distribution']['physician_start'] = 8
                config['staff_distribution']['physician_duration'] = 4
            if 'app' in description:
                config['staff_distribution']['add_app'] = True
                config['staff_distribution']['app_start'] = 8
                config['staff_distribution']['app_duration'] = 4

        if 'bundling' in suggestion or 'task bundling' in description:
            config['task_bundling'] = {
                'efficiency_factor': 0.2  # Default 20% efficiency gain
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