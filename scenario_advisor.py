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

                # Format structured recommendation
                formatted_rec = {
                    "title": rec.get('suggestion', ''),
                    "description": rec.get('description', ''),
                    "risk_factors": rec.get('risk_factors', []),
                    "config": config,  # Configuration for quick apply
                    "impact": rec.get('expected_impact', {})
                }
                formatted_recommendations.append(formatted_rec)
            else:
                # Handle plain text recommendations
                formatted_recommendations.append({
                    "title": "Recommendation",
                    "description": rec,
                    "config": {},
                    "impact": {}
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

        # Map recommendation keywords to configuration parameters
        if 'protected_time' in recommendation.get('suggestion', '').lower():
            config['protected_time'] = True
            config['protected_start'] = 9  # Default to 9 AM
            config['protected_duration'] = 2  # Default to 2 hours

        if 'staff' in recommendation.get('suggestion', '').lower():
            config['staff_distribution'] = True
            if 'physician' in recommendation.get('description', '').lower():
                config['add_physician'] = True
                config['physician_start'] = 8
                config['physician_duration'] = 4
            if 'app' in recommendation.get('description', '').lower():
                config['add_app'] = True
                config['app_start'] = 8
                config['app_duration'] = 4

        if 'bundling' in recommendation.get('suggestion', '').lower():
            config['task_bundling'] = True
            config['bundling_efficiency'] = 0.2  # Default 20% efficiency gain

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