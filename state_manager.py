from dataclasses import dataclass
from typing import Dict, Optional
import streamlit as st
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InterventionState:
    """Represents the state of intervention strategies"""
    # Protected Time settings
    protected_time: bool = False
    protected_start: int = 9
    protected_duration: int = 2

    # Staff Distribution settings
    staff_distribution: bool = False
    add_physician: bool = False
    physician_start: int = 8
    physician_duration: int = 4
    add_app: bool = False
    app_start: int = 8
    app_duration: int = 4

    # Task Bundling settings
    task_bundling: bool = False
    bundling_efficiency: float = 0.2

class StateManager:
    """Manages the application state and ensures synchronization"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize or reset session state with default values"""
        defaults = InterventionState()
        for field in defaults.__dataclass_fields__:
            if field not in st.session_state:
                st.session_state[field] = getattr(defaults, field)
                logger.info(f"Initialized {field} with default value: {getattr(defaults, field)}")

    @staticmethod
    def update_from_config(config: Dict):
        """Update session state from a configuration dictionary"""
        try:
            logger.info(f"Updating state from config: {config}")
            
            if config.get('protected_time'):
                st.session_state['protected_time'] = True
                st.session_state['protected_start'] = config['protected_time']['start_hour']
                st.session_state['protected_duration'] = config['protected_time']['duration']
                logger.info("Updated protected time settings")

            if config.get('staff_distribution'):
                st.session_state['staff_distribution'] = True
                staff_config = config['staff_distribution']

                if staff_config.get('add_physician'):
                    st.session_state['add_physician'] = True
                    st.session_state['physician_start'] = staff_config['physician_start']
                    st.session_state['physician_duration'] = staff_config['physician_duration']
                    logger.info("Updated physician staffing settings")

                if staff_config.get('add_app'):
                    st.session_state['add_app'] = True
                    st.session_state['app_start'] = staff_config['app_start']
                    st.session_state['app_duration'] = staff_config['app_duration']
                    logger.info("Updated APP staffing settings")

            if config.get('task_bundling'):
                st.session_state['task_bundling'] = True
                st.session_state['bundling_efficiency'] = config['task_bundling']['efficiency_factor']
                logger.info("Updated task bundling settings")

            # Log the final state for debugging
            StateManager.log_current_state()
            return True

        except Exception as e:
            logger.error(f"Error updating state from config: {str(e)}")
            return False

    @staticmethod
    def get_current_state() -> InterventionState:
        """Get the current intervention state"""
        return InterventionState(
            protected_time=st.session_state.get('protected_time', False),
            protected_start=st.session_state.get('protected_start', 9),
            protected_duration=st.session_state.get('protected_duration', 2),
            staff_distribution=st.session_state.get('staff_distribution', False),
            add_physician=st.session_state.get('add_physician', False),
            physician_start=st.session_state.get('physician_start', 8),
            physician_duration=st.session_state.get('physician_duration', 4),
            add_app=st.session_state.get('add_app', False),
            app_start=st.session_state.get('app_start', 8),
            app_duration=st.session_state.get('app_duration', 4),
            task_bundling=st.session_state.get('task_bundling', False),
            bundling_efficiency=st.session_state.get('bundling_efficiency', 0.2)
        )

    @staticmethod
    def log_current_state():
        """Log the current state for debugging"""
        state = StateManager.get_current_state()
        logger.info("Current State:")
        for field in state.__dataclass_fields__:
            logger.info(f"{field}: {getattr(state, field)}")

    @staticmethod
    def reset_state():
        """Reset the state to default values"""
        defaults = InterventionState()
        for field in defaults.__dataclass_fields__:
            st.session_state[field] = getattr(defaults, field)
        logger.info("State reset to defaults")
