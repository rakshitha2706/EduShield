"""
EduShield - Pydantic Models
Data validation schemas for API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class StudentFeatures(BaseModel):
    """Student features for prediction"""
    # Demographics (5)
    gender_encoded: int = Field(..., ge=0, le=1, description="Gender (0=F, 1=M)")
    disability_encoded: int = Field(..., ge=0, le=1, description="Disability (0=N, 1=Y)")
    education_level: int = Field(..., ge=0, le=4, description="Education level (0-4)")
    age_encoded: int = Field(..., ge=1, le=3, description="Age band (1-3)")
    imd_numeric: float = Field(..., ge=0, le=100, description="Deprivation index")
    
    # Academic Background (3)
    num_of_prev_attempts: int = Field(..., ge=0, description="Previous attempts")
    has_previous_attempts: int = Field(..., ge=0, le=1, description="Has previous attempts")
    studied_credits: int = Field(..., ge=0, description="Credits enrolled")
    
    # Assessment Performance (11)
    avg_score: float = Field(..., ge=0, le=100, description="Average score")
    std_score: float = Field(..., ge=0, description="Score standard deviation")
    min_score: float = Field(..., ge=0, le=100, description="Minimum score")
    max_score: float = Field(..., ge=0, le=100, description="Maximum score")
    num_assessments: int = Field(..., ge=0, description="Number of assessments")
    first_half_avg_score: float = Field(..., ge=0, le=100, description="Early performance")
    second_half_avg_score: float = Field(..., ge=0, le=100, description="Recent performance")
    score_trend: float = Field(..., description="Performance trend")
    submission_consistency: float = Field(..., ge=0, description="Submission regularity")
    late_submissions: int = Field(..., ge=0, description="Late submissions count")
    avg_submission_day: float = Field(..., ge=0, description="Average submission day")
    
    # Registration (4)
    registration_day: int = Field(..., description="Registration timing")
    early_registration: int = Field(..., ge=0, le=1, description="Early registration")
    late_registration: int = Field(..., ge=0, le=1, description="Late registration")
    unregistered: int = Field(..., ge=0, le=1, description="Withdrew from course")


class PredictionRequest(BaseModel):
    """Request for dropout prediction"""
    student_id: str = Field(..., description="Unique student identifier")
    student_name: str = Field(..., description="Student name")
    department: Optional[str] = Field(None, description="Department/Course")
    features: StudentFeatures


class TopFeature(BaseModel):
    """Top contributing feature"""
    feature: str
    contribution: float
    feature_value: float


class PredictionResponse(BaseModel):
    """Response with prediction results"""
    student_id: str
    student_name: str
    probability: float = Field(..., description="Dropout probability (0-1)")
    risk_percentage: float = Field(..., description="Risk percentage (0-100)")
    risk_category: str = Field(..., description="Low/Medium/High Risk")
    risk_emoji: str = Field(..., description="Risk indicator emoji")
    top_features: List[TopFeature] = Field(..., description="Top contributing factors")
    timestamp: datetime


class InterventionRequest(BaseModel):
    """Request to schedule intervention"""
    student_id: str
    intervention_type: str = Field(..., description="Type of intervention")
    description: str = Field(..., description="Intervention details")
    scheduled_date: datetime
    assigned_to: Optional[str] = None


class InterventionUpdate(BaseModel):
    """Update intervention status"""
    status: str = Field(..., description="completed/cancelled/rescheduled")
    notes: Optional[str] = None
    completion_date: Optional[datetime] = None
    risk_after: Optional[float] = None


class StudentResponse(BaseModel):
    """Student information response"""
    student_id: str
    student_name: str
    department: Optional[str]
    gender: Optional[str]
    age_band: Optional[str]
    education_level: Optional[str]
    disability: Optional[str]
    avg_score: float
    num_assessments: int
    current_risk_score: float
    risk_category: str
    risk_emoji: str
    last_updated: datetime


class StatsResponse(BaseModel):
    """System statistics"""
    total_students: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_risk_score: float
    model_accuracy: Optional[float]
    last_updated: datetime


class ModelInfo(BaseModel):
    """ML Model information"""
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    total_students_trained: int
    last_trained: str
    top_features: List[Dict]