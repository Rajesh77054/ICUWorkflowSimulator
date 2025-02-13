from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import QueuePool
import os
from datetime import datetime
import logging
from urllib.parse import urlparse, parse_qs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Parse the URL to add SSL requirements if not present
parsed_url = urlparse(DATABASE_URL)
query_params = parse_qs(parsed_url.query)

# Add SSL mode if not present
if 'sslmode' not in query_params:
    if DATABASE_URL.endswith('/'):
        DATABASE_URL = f"{DATABASE_URL}?sslmode=require"
    else:
        DATABASE_URL = f"{DATABASE_URL}?sslmode=require"

# Configure SQLAlchemy engine with connection pooling and retry settings
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True,  # Enable connection health checks
    connect_args={
        "connect_timeout": 10,  # Connection timeout in seconds
        "keepalives": 1,       # Enable keepalive
        "keepalives_idle": 30  # Idle time before sending keepalive
    }
)

# Create session factory
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

# Improved database session management
def get_db():
    db = SessionLocal()
    try:
        # Test the connection
        db.execute("SELECT 1")
        yield db
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        db.close()

def save_workflow_record(db, nursing_q, exam_callbacks, peer_interrupts,
                        providers, admissions, consults, transfers,
                        critical_events, metrics, predictions):
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
        intervention_effectiveness=analysis.get('intervention_effectiveness', {}),
        statistical_significance=analysis.get('statistical_significance', {})
    )

    db.add(result)
    db.commit()
    return result

def get_historical_records(db, limit=100):
    return db.query(WorkflowRecord).order_by(
        WorkflowRecord.timestamp.desc()
    ).limit(limit).all()

def get_scenarios(db, limit=100):
    return db.query(Scenario).order_by(
        Scenario.created_at.desc()
    ).limit(limit).all()

def get_scenario_results(db, scenario_id):
    return db.query(ScenarioResult).filter(
        ScenarioResult.scenario_id == scenario_id
    ).order_by(ScenarioResult.timestamp.desc()).all()

def delete_scenario(db, scenario_id):
    db.query(ScenarioResult).filter(
        ScenarioResult.scenario_id == scenario_id
    ).delete(synchronize_session=False)

    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if scenario:
        db.delete(scenario)
        db.commit()
        return True
    return False

def check_scenario_exists(db, name):
    return db.query(Scenario).filter(Scenario.name == name).first() is not None

# Initialize database tables
def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

# Initialize the database on import
init_db()