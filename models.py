from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from datetime import datetime

# Get database URL from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class WorkflowRecord(Base):
    __tablename__ = "workflow_records"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Input parameters
    nursing_questions = Column(Float)
    exam_callbacks = Column(Float)
    peer_interrupts = Column(Float)
    providers = Column(Integer)
    admissions = Column(Integer)
    consults = Column(Integer)
    transfers = Column(Integer)
    critical_events = Column(Integer)

    # Calculated metrics
    interrupts_per_provider = Column(Float)
    time_lost = Column(Float)
    efficiency = Column(Float)
    cognitive_load = Column(Float)
    burnout_risk = Column(Float)

    # Time impacts
    interrupt_time = Column(Float)
    admission_time = Column(Float)
    critical_time = Column(Float)

    # ML predictions
    predicted_workload = Column(Float)
    predicted_burnout = Column(Float)

    # Additional data
    risk_components = Column(JSON)
    recommendations = Column(JSON)

class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Configuration parameters
    base_config = Column(JSON)  # Base workflow configuration
    interventions = Column(JSON)  # Intervention strategies

    # Relationships
    results = relationship("ScenarioResult", back_populates="scenario")

class ScenarioResult(Base):
    __tablename__ = "scenario_results"

    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Performance metrics
    efficiency = Column(Float)
    cognitive_load = Column(Float)
    burnout_risk = Column(Float)
    interruption_reduction = Column(Float)
    task_completion_rate = Column(Float)
    provider_satisfaction = Column(Float)

    # Cost-benefit metrics
    implementation_cost = Column(Float)
    benefit_score = Column(Float)
    roi = Column(Float)

    # Additional analysis
    risk_reduction = Column(JSON)
    intervention_effectiveness = Column(JSON)
    statistical_significance = Column(JSON)

    # Relationship
    scenario = relationship("Scenario", back_populates="results")

# Create all tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_workflow_record(
    db,
    nursing_q, exam_callbacks, peer_interrupts,
    providers, admissions, consults, transfers,
    critical_events, metrics, predictions
):
    """Save a workflow record to the database"""
    record = WorkflowRecord(
        nursing_questions=nursing_q,
        exam_callbacks=exam_callbacks,
        peer_interrupts=peer_interrupts,
        providers=providers,
        admissions=admissions,
        consults=consults,
        transfers=transfers,
        critical_events=critical_events,
        interrupts_per_provider=metrics['interrupts_per_provider'],
        time_lost=metrics['time_lost'],
        efficiency=metrics['efficiency'],
        cognitive_load=metrics['cognitive_load'],
        burnout_risk=metrics['burnout_risk'],
        interrupt_time=metrics['interrupt_time'],
        admission_time=metrics['admission_time'],
        critical_time=metrics['critical_time'],
        predicted_workload=predictions['predicted_workload'],
        predicted_burnout=predictions['predicted_burnout'],
        risk_components=predictions.get('risk_components', {}),
        recommendations=metrics.get('recommendations', [])
    )

    db.add(record)
    db.commit()
    return record

def save_scenario(db, name, description, base_config, interventions):
    """Save a new scenario configuration"""
    scenario = Scenario(
        name=name,
        description=description,
        base_config=base_config,
        interventions=interventions
    )

    db.add(scenario)
    db.commit()
    return scenario

def save_scenario_result(db, scenario_id, metrics, analysis):
    """Save scenario execution results"""
    # Ensure the intervention_effectiveness is a dictionary
    if not isinstance(metrics.get('intervention_effectiveness'), dict):
        metrics['intervention_effectiveness'] = {
            'protected_time': 0.0,
            'staff_distribution': 0.0,
            'task_bundling': 0.0
        }

    result = ScenarioResult(
        scenario_id=scenario_id,
        efficiency=metrics['efficiency'],
        cognitive_load=metrics['cognitive_load'],
        burnout_risk=metrics['burnout_risk'],
        interruption_reduction=metrics.get('interruption_reduction', 0.0),
        task_completion_rate=metrics.get('task_completion_rate', 0.0),
        provider_satisfaction=metrics.get('provider_satisfaction', 0.0),
        implementation_cost=analysis.get('implementation_cost', 0.0),
        benefit_score=analysis.get('benefit_score', 0.0),
        roi=analysis.get('roi', 0.0),
        risk_reduction=analysis.get('risk_reduction', {}),
        intervention_effectiveness=metrics.get('intervention_effectiveness', {}),
        statistical_significance=analysis.get('statistical_significance', {})
    )

    db.add(result)
    db.commit()
    return result

def get_historical_records(db, limit=100):
    """Retrieve historical workflow records"""
    return db.query(WorkflowRecord).order_by(
        WorkflowRecord.timestamp.desc()
    ).limit(limit).all()

def get_scenarios(db, limit=100):
    """Retrieve all scenarios"""
    return db.query(Scenario).order_by(
        Scenario.created_at.desc()
    ).limit(limit).all()

def get_scenario_results(db, scenario_id):
    """Retrieve results for a specific scenario"""
    return db.query(ScenarioResult).filter(
        ScenarioResult.scenario_id == scenario_id
    ).order_by(ScenarioResult.timestamp.desc()).all()