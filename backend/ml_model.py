"""
EduShield - Machine Learning Model Training
Random Forest Classifier with SMOTE for class imbalance
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve
)
# SMOTE removed - using class_weight='balanced' instead
import warnings
warnings.filterwarnings('ignore')


class EduShieldModel:
    """Random Forest model for dropout prediction"""
    
    def __init__(self, data_path=None):
        # Default to backend/data/processed_student_data.csv
        if data_path is None:
            self.data_path = Path(__file__).parent / "data" / "processed_student_data.csv"
        else:
            self.data_path = Path(data_path)
        self.model_dir = Path(__file__).parent / "ml_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = {}

        self.load_model()
        
    def load_data(self):
        """Load processed data"""
        print("📚 Loading processed data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} records")
        
        # Define feature columns
        self.feature_names = [
            # Demographics (5)
            'gender_encoded', 'disability_encoded', 'education_level', 
            'age_encoded', 'imd_numeric',
            
            # Academic Background (3)
            'num_of_prev_attempts', 'has_previous_attempts', 'studied_credits',
            
            # Assessment Performance (11)
            'avg_score', 'std_score', 'min_score', 'max_score', 'num_assessments',
            'first_half_avg_score', 'second_half_avg_score', 'score_trend',
            'submission_consistency', 'late_submissions', 'avg_submission_day',
            
            # Registration (4)
            'registration_day', 'early_registration', 'late_registration', 'unregistered'
        ]
        
        # Prepare X and y
        self.X = self.df[self.feature_names].copy()
        self.y = self.df['dropout'].copy()
        
        print(f"   - Features: {len(self.feature_names)}")
        print(f"   - Samples: {len(self.X)}")
        print(f"   - Dropout rate: {self.y.mean()*100:.2f}%")
        
    def prepare_data(self):
        """Split and scale data"""
        print("\n🔀 Splitting data...")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"   - Training set: {len(self.X_train)} samples")
        print(f"   - Test set: {len(self.X_test)} samples")
        
        # Scale features
        print("\n📏 Scaling features...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   - Dropout rate in training: {self.y_train.mean()*100:.2f}%")
        print("   - Using class_weight='balanced' to handle imbalance")
        
    def train_model(self):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest Classifier...")
        print("   - Estimators: 200")
        print("   - Max depth: 20")
        print("   - Min samples split: 10")
        print("   - Class weight: balanced")
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handles class imbalance
        )
        
        self.model.fit(self.X_train_scaled, self.y_train)
        print("✅ Model training complete!")
        
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'total_students_trained': len(self.df),
            'last_trained': datetime.now().isoformat()
        }
        
        # Print results
        print(f"\n✨ Test Set Performance:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1-Score:  {f1*100:.2f}%")
        print(f"   ROC-AUC:   {roc_auc*100:.2f}%")
        
        print(f"\n📋 Confusion Matrix:")
        print(f"                Predicted")
        print(f"              Pass  Dropout")
        print(f"   Actual Pass  {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"         Drop  {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        # Classification report
        print(f"\n📈 Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Retained', 'Dropout']))
        
        # Cross-validation
        print(f"\n🔄 Cross-Validation (5-fold):")
        cv_scores = cross_val_score(self.model, self.X_train_scaled, 
                                    self.y_train, cv=5, 
                                    scoring='roc_auc')
        print(f"   Mean ROC-AUC: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        
    def feature_importance(self):
        """Display feature importance"""
        print("\n" + "="*60)
        print("🎯 FEATURE IMPORTANCE")
        print("="*60)
        
        # Get feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 15 Most Important Features:")
        print("-" * 60)
        
        for i, idx in enumerate(indices[:15], 1):
            print(f"{i:2d}. {self.feature_names[idx]:30s} {importances[idx]*100:6.2f}%")
        
        # Store top features
        self.metrics['top_features'] = [
            {
                'feature': self.feature_names[idx],
                'importance': float(importances[idx])
            }
            for idx in indices[:10]
        ]
        
    def save_model(self):
        """Save trained model and artifacts"""
        print("\n💾 Saving model artifacts...")
        
        # Save model
        model_path = self.model_dir / 'random_forest_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"   ✓ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✓ Scaler saved: {scaler_path}")
        
        # Save feature names
        features_path = self.model_dir / 'feature_names.pkl'
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"   ✓ Features saved: {features_path}")
        
        # Save metrics
        metrics_path = self.model_dir / 'metrics.pkl'
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        print(f"   ✓ Metrics saved: {metrics_path}")
        
    def train_pipeline(self):
        """Complete training pipeline"""
        print("\n" + "="*60)
        print("🛡️  EDUSHIELD - MODEL TRAINING PIPELINE")
        print("="*60)
        
        self.load_data()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()
        self.feature_importance()
        self.save_model()
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETE!")
        print("="*60)
        print(f"\n🎯 Model Performance Summary:")
        print(f"   Accuracy:  {self.metrics['accuracy']*100:.2f}%")
        print(f"   Precision: {self.metrics['precision']*100:.2f}%")
        print(f"   Recall:    {self.metrics['recall']*100:.2f}%")
        print(f"   F1-Score:  {self.metrics['f1_score']*100:.2f}%")
        print(f"   ROC-AUC:   {self.metrics['roc_auc']*100:.2f}%")
        print("\n🚀 Model ready for deployment!")


if __name__ == "__main__":
    trainer = EduShieldModel()
    trainer.train_pipeline()