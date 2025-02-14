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
        in intensive care units.
        
        IMPORTANT: You will receive current metrics and workflow configuration in your context.
        You MUST:
        1. Begin each response by stating ONLY the metrics that are explicitly provided in the context
        2. For each metric, use ONLY the exact numerical values from the context:
           - Current efficiency (format as percentage)
           - Current cognitive load (format as percentage)
           - Current burnout risk (format as percentage)
           - ICU Census (ADC)
           - Number of providers
           - Consults per shift
           - Critical events per week
        3. DO NOT make assumptions about metrics - if a value is not provided, state "Not available"
        4. DO NOT reference historical or estimated values
        5. Format metric values exactly as they appear in the context (e.g., if efficiency is 0.65, show as 65%)
        
        Your recommendations must:
        1. Reference only the metrics that are explicitly provided
        2. Use specific numerical targets based on the actual current values
        3. State "Insufficient data" if key metrics needed for a recommendation are missing"""
        self.chat_history = []
        self.max_history = 10  # Maximum number of messages to maintain in history

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

    def chat_with_user(self, user_message, current_metrics=None, workflow_config=None, active_scenario=None):
        """Handle interactive chat with users"""
        try:
            # Prepare messages including chat history and current context
            messages = [{"role": "system", "content": self.system_context}]

            # Add comprehensive context about current application state
            context_parts = []

            if current_metrics:
                context_parts.append(f"""Current ICU Metrics:
                - Efficiency: {current_metrics.get('efficiency', 'N/A'):.1%}
                - Cognitive Load: {current_metrics.get('cognitive_load', 'N/A'):.1f}%
                - Burnout Risk: {current_metrics.get('burnout_risk', 'N/A'):.1%}""")

            if workflow_config:
                context_parts.append(f"""Current Workflow Configuration:
                - ICU Census (ADC): {workflow_config.get('adc', 'N/A')}
                - Providers: {workflow_config.get('providers', 'N/A')}
                - Consults per Shift: {workflow_config.get('consults', 'N/A')}
                - Critical Events per Week: {workflow_config.get('critical_events', 'N/A')}""")

            if active_scenario:
                context_parts.append(f"""Active Scenario:
                - Name: {active_scenario.get('name', 'N/A')}
                - Description: {active_scenario.get('description', 'N/A')}
                - Interventions: {', '.join(active_scenario.get('interventions', {}).keys())}""")

            if context_parts:
                full_context = "\n\n".join(context_parts)
                messages.append({"role": "system", "content": full_context})

            # Add chat history
            messages.extend(self.chat_history)

            # Add user's current message
            messages.append({"role": "user", "content": user_message})

            # Get response from Grok
            response = self.client.chat.completions.create(
                model="grok-2-1212",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )

            # Extract the response
            ai_response = response.choices[0].message.content

            # Update chat history
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": ai_response})

            # Maintain maximum history length
            if len(self.chat_history) > self.max_history * 2:  # *2 because each exchange has 2 messages
                self.chat_history = self.chat_history[-self.max_history * 2:]

            return {
                "status": "success",
                "response": ai_response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response": "I apologize, but I'm unable to process your request at the moment. Please try again later.",
                "timestamp": datetime.now().isoformat()
            }

    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
        return {"status": "success", "message": "Chat history cleared"}

    def _create_scenario_prompt(self, scenario_config, current_metrics):
        """Create prompt for scenario analysis"""
        return f"""Analyze this ICU workflow scenario and provide optimization recommendations. 
Format your response as a JSON object with the following structure:
{{
    "recommendations": [
        "A clear, actionable recommendation in natural language",
        "Another recommendation with specific details",
        ...
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
        "A clear description of expected impacts in natural language",
        "Implementation timeline and complexity details",
        "Resource requirements and constraints",
        "Expected ROI and benefits",
        "Risk factors and mitigation strategies"
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