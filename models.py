from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

# Get database URL from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL.startswith('postgres://'):
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

def get_historical_records(db, limit=100):
    """Retrieve historical workflow records"""
    return db.query(WorkflowRecord).order_by(
        WorkflowRecord.timestamp.desc()
    ).limit(limit).all()
