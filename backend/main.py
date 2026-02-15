"""
EduShield - FastAPI Backend
Complete API for student dropout prediction system
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

from backend.models import (
    PredictionRequest, PredictionResponse, StudentResponse,
    InterventionRequest, InterventionUpdate, StatsResponse,
    ModelInfo, TopFeature
)
from backend.database import db

# Initialize FastAPI app
app = FastAPI(
    title="EduShield API",
    description="Student Dropout Prediction System - Early Intervention Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
scaler = None
feature_names = None
metrics = None

MODEL_DIR = Path(__file__).parent / "ml_models"


def load_model_artifacts():
    """Load trained model and artifacts"""
    global model, scaler, feature_names, metrics
    
    try:
        # Load model
        with open(MODEL_DIR / 'random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature names
        with open(MODEL_DIR / 'feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load metrics
        with open(MODEL_DIR / 'metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        
        print("✅ Model artifacts loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  Model not loaded: {e}")
        return False


def calculate_risk_category(probability: float):
    """Calculate risk category and emoji"""
    risk_percentage = probability * 100
    
    if risk_percentage <= 40:
        return "Low Risk", "🟢"
    elif risk_percentage <= 70:
        return "Medium Risk", "🟡"
    else:
        return "High Risk", "🔴"


def get_top_contributing_features(features_dict: dict, feature_importance: np.ndarray, n_top: int = 3):
    """Get top N contributing features for this student"""
    features_array = np.array([features_dict[fname] for fname in feature_names])
    
    # Calculate contribution (feature_value * importance)
    contributions = np.abs(features_array * feature_importance)
    
    # Get top indices
    top_indices = np.argsort(contributions)[-n_top:][::-1]
    
    top_features = []
    for idx in top_indices:
        top_features.append(TopFeature(
            feature=feature_names[idx],
            contribution=float(contributions[idx]),
            feature_value=float(features_array[idx])
        ))
    
    return top_features


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    await db.connect_db()
    load_model_artifacts()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await db.close_db()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "EduShield API",
        "version": "1.0.0",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and performance metrics"""
    if not model or not metrics:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type="Random Forest Classifier",
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1_score=metrics['f1_score'],
        roc_auc=metrics['roc_auc'],
        total_students_trained=metrics['total_students_trained'],
        last_trained=metrics['last_trained'],
        top_features=metrics['top_features']
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_dropout(request: PredictionRequest):
    """Predict dropout risk for a student"""
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features_dict = request.features.dict()
        features_array = np.array([[features_dict[fname] for fname in feature_names]])
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Predict
        probability = model.predict_proba(features_scaled)[0][1]
        risk_category, risk_emoji = calculate_risk_category(probability)
        
        # Get top contributing features
        feature_importance = model.feature_importances_
        top_features = get_top_contributing_features(features_dict, feature_importance)
        
        # Create response
        response = PredictionResponse(
            student_id=request.student_id,
            student_name=request.student_name,
            probability=float(probability),
            risk_percentage=float(probability * 100),
            risk_category=risk_category,
            risk_emoji=risk_emoji,
            top_features=top_features,
            timestamp=datetime.now()
        )
        
        # Save prediction to database
        prediction_data = response.dict()
        await db.save_prediction(prediction_data)
        
        # Update student record
        student_data = {
            "student_id": request.student_id,
            "student_name": request.student_name,
            "department": request.department,
            "current_risk_score": probability,
            "risk_category": risk_category,
            "risk_emoji": risk_emoji,
            "last_updated": datetime.now(),
            **features_dict
        }
        await db.create_student(student_data)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/train-model")
async def train_model():
    """Train/retrain the ML model"""
    try:
        # Import training module
        from backend.ml_model import EduShieldModel
        
        # Train model
        trainer = EduShieldModel()
        trainer.train_pipeline()
        
        # Reload model artifacts
        load_model_artifacts()
        
        # Save metadata to database
        await db.save_model_metadata(metrics)
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.get("/students")
async def get_students(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    risk_filter: Optional[str] = Query(None, regex="^(Low Risk|Medium Risk|High Risk)$")
):
    """Get all students with optional filtering"""
    students = await db.get_all_students(skip=skip, limit=limit, risk_filter=risk_filter)
    
    # Convert ObjectId to string
    for student in students:
        student['_id'] = str(student['_id'])
    
    return {
        "total": len(students),
        "students": students
    }


@app.get("/students/{student_id}")
async def get_student_details(student_id: str):
    """Get detailed student information"""
    student = await db.get_student(student_id)
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Get prediction history
    predictions = await db.get_student_predictions(student_id)
    
    # Get interventions
    interventions = await db.get_student_interventions(student_id)
    
    # Convert ObjectIds to strings
    student['_id'] = str(student['_id'])
    for pred in predictions:
        pred['_id'] = str(pred['_id'])
    for inter in interventions:
        inter['_id'] = str(inter['_id'])
    
    return {
        "student": student,
        "prediction_history": predictions,
        "interventions": interventions
    }


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get system statistics"""
    stats = await db.get_statistics()
    
    # Add model accuracy if available
    if metrics:
        stats['model_accuracy'] = metrics.get('accuracy')
    
    return StatsResponse(**stats)


@app.get("/alerts")
async def get_alerts(threshold: float = Query(0.7, ge=0, le=1)):
    """Get high-risk students requiring attention"""
    alerts = await db.get_high_risk_alerts(threshold=threshold)
    
    # Convert ObjectIds
    for alert in alerts:
        alert['_id'] = str(alert['_id'])
        
        # Get recent predictions to show trend
        predictions = await db.get_student_predictions(alert['student_id'], limit=3)
        alert['recent_predictions'] = [p['risk_percentage'] for p in predictions]
    
    return {
        "threshold": threshold * 100,
        "count": len(alerts),
        "alerts": alerts
    }


@app.post("/intervention")
async def schedule_intervention(request: InterventionRequest):
    """Schedule an intervention for a student"""
    # Get current student risk
    student = await db.get_student(request.student_id)
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    intervention_data = {
        **request.dict(),
        "risk_before": student.get('current_risk_score', 0)
    }
    
    intervention_id = await db.create_intervention(intervention_data)
    
    return {
        "status": "success",
        "intervention_id": intervention_id,
        "message": f"Intervention scheduled for {request.scheduled_date}"
    }


@app.put("/intervention/{intervention_id}")
async def update_intervention_status(intervention_id: str, update: InterventionUpdate):
    """Update intervention status"""
    update_data = {
        **update.dict(exclude_none=True),
        "updated_at": datetime.now()
    }
    
    # If completed, calculate improvement
    if update.status == "completed" and update.risk_after is not None:
        # Get intervention details
        from bson import ObjectId
        intervention = await db.db.interventions.find_one({"_id": ObjectId(intervention_id)})
        
        if intervention and 'risk_before' in intervention:
            risk_before = intervention['risk_before']
            improvement = ((risk_before - update.risk_after) / risk_before) * 100
            update_data['improvement_percentage'] = improvement
    
    result = await db.update_intervention(intervention_id, update_data)
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Intervention not found")
    
    return {
        "status": "success",
        "message": "Intervention updated successfully"
    }


@app.get("/intervention-effectiveness")
async def get_intervention_effectiveness():
    """Get intervention effectiveness analytics"""
    effectiveness = await db.get_intervention_effectiveness()
    
    return {
        "total_interventions": len(effectiveness),
        "effectiveness_by_type": effectiveness
    }


@app.get("/batch-predict")
async def batch_predict_from_csv():
    """Batch predict for all students in processed data"""
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load processed data
        data_path = Path(__file__).parent / "data" / "processed_student_data.csv"
        df = pd.read_csv(data_path)
        
        predictions_made = 0
        
        for _, row in df.head(100).iterrows():  # Limit to first 100 for demo
            # Prepare features
            features_dict = {fname: row[fname] for fname in feature_names}
            features_array = np.array([[features_dict[fname] for fname in feature_names]])
            
            # Scale and predict
            features_scaled = scaler.transform(features_array)
            probability = model.predict_proba(features_scaled)[0][1]
            risk_category, risk_emoji = calculate_risk_category(probability)
            
            # Save to database
            student_data = {
                "student_id": str(row['id_student']),
                "student_name": f"Student {row['id_student']}",
                "department": row.get('code_module', 'Unknown'),
                "gender": row.get('gender', 'Unknown'),
                "age_band": row.get('age_band', 'Unknown'),
                "education_level": row.get('highest_education', 'Unknown'),
                "disability": row.get('disability', 'N'),
                "avg_score": float(row.get('avg_score', 0)),
                "num_assessments": int(row.get('num_assessments', 0)),
                "current_risk_score": float(probability),
                "risk_category": risk_category,
                "risk_emoji": risk_emoji,
                "last_updated": datetime.now()
            }
            
            await db.create_student(student_data)
            predictions_made += 1
        
        return {
            "status": "success",
            "predictions_made": predictions_made,
            "message": f"Batch prediction completed for {predictions_made} students"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)