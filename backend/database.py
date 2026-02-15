"""
EduShield - Database Operations
MongoDB connection and CRUD operations
"""

from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional, Dict
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


class Database:
    """MongoDB database manager"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        
    async def connect_db(self):
        """Connect to MongoDB"""
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        database_name = os.getenv("DATABASE_NAME", "edushield_db")
        
        self.client = AsyncIOMotorClient(mongodb_url)
        self.db = self.client[database_name]
        print(f"✅ Connected to MongoDB: {database_name}")
        
    async def close_db(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("❌ MongoDB connection closed")
    
    # Student operations
    async def create_student(self, student_data: dict):
        """Create or update student record"""
        result = await self.db.students.update_one(
            {"student_id": student_data["student_id"]},
            {"$set": student_data},
            upsert=True
        )
        return result
    
    async def get_student(self, student_id: str):
        """Get student by ID"""
        student = await self.db.students.find_one({"student_id": student_id})
        return student
    
    async def get_all_students(self, skip: int = 0, limit: int = 100, 
                              risk_filter: Optional[str] = None):
        """Get all students with optional filtering"""
        query = {}
        if risk_filter:
            query["risk_category"] = risk_filter
        
        cursor = self.db.students.find(query).skip(skip).limit(limit).sort("current_risk_score", -1)
        students = await cursor.to_list(length=limit)
        return students
    
    async def update_student_risk(self, student_id: str, risk_data: dict):
        """Update student risk assessment"""
        result = await self.db.students.update_one(
            {"student_id": student_id},
            {
                "$set": {
                    "current_risk_score": risk_data["probability"],
                    "risk_category": risk_data["risk_category"],
                    "risk_emoji": risk_data["risk_emoji"],
                    "last_updated": datetime.now()
                }
            }
        )
        return result
    
    # Prediction operations
    async def save_prediction(self, prediction_data: dict):
        """Save prediction result"""
        result = await self.db.predictions.insert_one(prediction_data)
        return result
    
    async def get_student_predictions(self, student_id: str, limit: int = 10):
        """Get prediction history for student"""
        cursor = self.db.predictions.find(
            {"student_id": student_id}
        ).sort("timestamp", -1).limit(limit)
        predictions = await cursor.to_list(length=limit)
        return predictions
    
    # Intervention operations
    async def create_intervention(self, intervention_data: dict):
        """Schedule new intervention"""
        intervention_data["created_at"] = datetime.now()
        intervention_data["status"] = "scheduled"
        result = await self.db.interventions.insert_one(intervention_data)
        return str(result.inserted_id)
    
    async def update_intervention(self, intervention_id: str, update_data: dict):
        """Update intervention status"""
        from bson import ObjectId
        result = await self.db.interventions.update_one(
            {"_id": ObjectId(intervention_id)},
            {"$set": update_data}
        )
        return result
    
    async def get_student_interventions(self, student_id: str):
        """Get interventions for student"""
        cursor = self.db.interventions.find(
            {"student_id": student_id}
        ).sort("scheduled_date", -1)
        interventions = await cursor.to_list(length=100)
        return interventions
    
    async def get_intervention_effectiveness(self):
        """Calculate intervention effectiveness"""
        pipeline = [
            {
                "$match": {
                    "status": "completed",
                    "risk_before": {"$exists": True},
                    "risk_after": {"$exists": True}
                }
            },
            {
                "$group": {
                    "_id": "$intervention_type",
                    "count": {"$sum": 1},
                    "avg_improvement": {
                        "$avg": {
                            "$subtract": ["$risk_before", "$risk_after"]
                        }
                    }
                }
            }
        ]
        
        cursor = self.db.interventions.aggregate(pipeline)
        results = await cursor.to_list(length=100)
        return results
    
    # Statistics operations
    async def get_statistics(self):
        """Get system statistics"""
        total_students = await self.db.students.count_documents({})
        high_risk = await self.db.students.count_documents({"risk_category": "High Risk"})
        medium_risk = await self.db.students.count_documents({"risk_category": "Medium Risk"})
        low_risk = await self.db.students.count_documents({"risk_category": "Low Risk"})
        
        # Calculate average risk
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_risk": {"$avg": "$current_risk_score"}
                }
            }
        ]
        cursor = self.db.students.aggregate(pipeline)
        avg_result = await cursor.to_list(length=1)
        avg_risk = avg_result[0]["avg_risk"] if avg_result else 0
        
        return {
            "total_students": total_students,
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "avg_risk_score": avg_risk,
            "last_updated": datetime.now()
        }
    
    async def get_high_risk_alerts(self, threshold: float = 0.7):
        """Get students requiring immediate attention"""
        cursor = self.db.students.find(
            {"current_risk_score": {"$gte": threshold}}
        ).sort("current_risk_score", -1).limit(20)
        alerts = await cursor.to_list(length=20)
        return alerts
    
    # Model metadata operations
    async def save_model_metadata(self, metadata: dict):
        """Save model training metadata"""
        result = await self.db.model_metadata.insert_one(metadata)
        return result
    
    async def get_latest_model_metadata(self):
        """Get latest model metadata"""
        cursor = self.db.model_metadata.find().sort("last_trained", -1).limit(1)
        metadata = await cursor.to_list(length=1)
        return metadata[0] if metadata else None


# Global database instance
db = Database()