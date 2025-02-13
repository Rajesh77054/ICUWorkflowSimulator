import os
from openai import OpenAI
from datetime import datetime
import json

class AIAssistant:
    def __init__(self):
        self.client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])
        self.system_context = """You are an expert ICU workflow optimization advisor. 
        Your role is to analyze workflow scenarios and provide actionable recommendations 
        for improving efficiency, reducing burnout risk, and optimizing resource allocation 
        in intensive care units. Provide recommendations in clear, natural language."""

    def get_scenario_advice(self, scenario_config, current_metrics):
        """Get AI recommendations for scenario optimization"""
        try:
            prompt = self._create_scenario_prompt(scenario_config, current_metrics)
            response = self.client.chat.completions.create(
                model="grok-2-1212",
                messages=[
                    {"role": "system", "content": self.system_context},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "error": str(e),
                "recommendations": ["Unable to generate AI recommendations at this time."],
                "confidence": 0.0
            }

    def analyze_intervention_impact(self, intervention_config):
        """Analyze potential impact of proposed interventions"""
        try:
            prompt = self._create_intervention_prompt(intervention_config)
            response = self.client.chat.completions.create(
                model="grok-2-1212",
                messages=[
                    {"role": "system", "content": self.system_context},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "error": str(e),
                "impact_analysis": {},
                "confidence": 0.0
            }

    def _create_scenario_prompt(self, scenario_config, current_metrics):
        """Create prompt for scenario analysis"""
        return f"""Analyze this ICU workflow scenario and provide optimization recommendations. 
Format your response as a JSON object with the following structure:
{{
    "recommendations": [
        {{
            "title": "Clear recommendation title",
            "description": "Detailed description of the recommendation",
            "risk_factors": ["Risk factor 1", "Risk factor 2"],
            "expected_impact": {{
                "efficiency": numeric_value_between_0_and_100,
                "cognitive_load": numeric_value_between_0_and_100,
                "burnout_risk": numeric_value_between_0_and_100
            }}
        }},
        // More recommendations...
    ],
    "impact": {{
        "efficiency": numeric_value_between_0_and_100,
        "cognitive_load": numeric_value_between_0_and_100,
        "burnout_risk": numeric_value_between_0_and_100
    }},
    "confidence": numeric_value_between_0_and_1
}}

Current Metrics:
- Efficiency: {current_metrics.get('efficiency', 0)}
- Cognitive Load: {current_metrics.get('cognitive_load', 0)}
- Burnout Risk: {current_metrics.get('burnout_risk', 0)}

Scenario Configuration:
{json.dumps(scenario_config, indent=2)}"""

    def _create_intervention_prompt(self, intervention_config):
        """Create prompt for intervention analysis"""
        return f"""Analyze the potential impact of these ICU workflow interventions. 
Format your response as a JSON object with the following structure:
{{
    "analysis": [
        {{
            "title": "Impact analysis title",
            "description": "Detailed description of expected impacts",
            "implementation": "Implementation timeline and complexity details",
            "resources": "Resource requirements and constraints",
            "roi": "Expected ROI and benefits",
            "risks": ["Risk factor 1", "Risk factor 2"]
        }},
        // More analysis points...
    ],
    "impact": {{
        "efficiency": numeric_value_between_0_and_100,
        "cognitive_load": numeric_value_between_0_and_100,
        "burnout_risk": numeric_value_between_0_and_100
    }},
    "confidence": numeric_value_between_0_and_1
}}

Intervention Configuration:
{json.dumps(intervention_config, indent=2)}"""