import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta

class WorkflowPredictor:
    def __init__(self):
        self.workload_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.burnout_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_features(self, data_dict):
        """Convert input dictionary to feature array"""
        features = np.array([
            data_dict['nursing_questions_per_hour'],
            data_dict['exam_callbacks_per_hour'],
            data_dict['peer_interrupts_per_hour'],
            data_dict['providers'],
            data_dict['admissions_per_shift'],
            data_dict['consults_per_shift'],
            data_dict['transfers_per_shift'],
            data_dict['critical_events_per_week']
        ]).reshape(1, -1)
        return features

    def generate_synthetic_data(self, current_features, num_samples=100):
        """Generate synthetic data for initial training"""
        base_features = current_features.reshape(1, -1)
        
        # Add random variations to create synthetic samples
        variations = np.random.normal(0, 0.1, size=(num_samples, base_features.shape[1]))
        synthetic_features = base_features + variations * base_features
        
        # Ensure no negative values
        synthetic_features = np.maximum(synthetic_features, 0)
        
        # Generate synthetic targets using domain knowledge
        synthetic_workload = np.clip(
            0.3 * synthetic_features[:, :3].sum(axis=1) +  # interruption impact
            0.4 * (synthetic_features[:, 4:7].sum(axis=1) / synthetic_features[:, 3]) +  # admission load per provider
            0.3 * (synthetic_features[:, 7] / 7),  # critical events impact
            0, 1
        )
        
        synthetic_burnout = np.clip(
            0.4 * synthetic_workload +
            0.3 * (synthetic_features[:, :3].sum(axis=1) / synthetic_features[:, 3]) +
            0.3 * np.random.normal(0.5, 0.1, num_samples),  # random fatigue factor
            0, 1
        )
        
        return synthetic_features, synthetic_workload, synthetic_burnout

    def train_initial_model(self, current_features):
        """Train the model with synthetic data based on current state"""
        features, workload_targets, burnout_targets = self.generate_synthetic_data(current_features)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train models
        self.workload_model.fit(scaled_features, workload_targets)
        self.burnout_model.fit(scaled_features, burnout_targets)
        
        return {
            'workload_score': self.workload_model.score(scaled_features, workload_targets),
            'burnout_score': self.burnout_model.score(scaled_features, burnout_targets)
        }

    def predict(self, features):
        """Make predictions for workload and burnout risk"""
        scaled_features = self.scaler.transform(features)
        
        workload_pred = self.workload_model.predict(scaled_features)[0]
        burnout_pred = self.burnout_model.predict(scaled_features)[0]
        
        # Get feature importances
        workload_importance = dict(zip(
            ['nursing_q', 'callbacks', 'peer_int', 'providers', 'admissions', 'consults', 'transfers', 'critical'],
            self.workload_model.feature_importances_
        ))
        
        burnout_importance = dict(zip(
            ['nursing_q', 'callbacks', 'peer_int', 'providers', 'admissions', 'consults', 'transfers', 'critical'],
            self.burnout_model.feature_importances_
        ))
        
        return {
            'predicted_workload': float(workload_pred),
            'predicted_burnout': float(burnout_pred),
            'workload_importance': workload_importance,
            'burnout_importance': burnout_importance
        }

    def predict_next_week(self, current_features, num_days=7):
        """Predict workload and burnout trends for the next week"""
        predictions = []
        base_features = current_features.copy()
        
        for day in range(num_days):
            # Add small random variations to simulate daily changes
            daily_features = base_features * (1 + np.random.normal(0, 0.05, size=base_features.shape))
            daily_predictions = self.predict(daily_features)
            
            predictions.append({
                'day': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'workload': daily_predictions['predicted_workload'],
                'burnout': daily_predictions['predicted_burnout']
            })
        
        return predictions

    def save_models(self, path='models/'):
        """Save trained models and scaler"""
        import os
        if not os.path.exists(path):
            os.makedirs(path)
            
        joblib.dump(self.workload_model, f'{path}workload_model.joblib')
        joblib.dump(self.burnout_model, f'{path}burnout_model.joblib')
        joblib.dump(self.scaler, f'{path}scaler.joblib')

    def load_models(self, path='models/'):
        """Load trained models and scaler"""
        self.workload_model = joblib.load(f'{path}workload_model.joblib')
        self.burnout_model = joblib.load(f'{path}burnout_model.joblib')
        self.scaler = joblib.load(f'{path}scaler.joblib')
