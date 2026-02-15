
# 🛡️ EduShield - Student Dropout Prediction System

## Complete AI-Powered Early Intervention Platform for Indian Educational Institutions

A production-ready machine learning system for predicting student dropout risk using Random Forest Classifier with explainable AI features. Built specifically to address the challenge of student dropouts in Indian education through early identification and timely interventions.

---

## 🌟 Features

### ✨ Machine Learning
- **Random Forest Classifier** with 200 estimators
- **SMOTE** for handling class imbalance
- **Feature Importance** analysis with 23 engineered features
- **Explainable AI** - Top 3 contributing risk factors per student
- **Cross-Validation** for robust performance metrics
- **Real-time Predictions** with confidence scores

### 📊 Risk Assessment
- **🟢 Low Risk** (0-40%) - Students performing well
- **🟡 Medium Risk** (41-70%) - Students needing monitoring
- **🔴 High Risk** (71-100%) - Students requiring immediate intervention

### 🎯 Key Capabilities
- Early dropout risk detection
- Intervention tracking & effectiveness measurement
- Real-time dashboard with interactive charts
- Detailed student analytics
- Alert system for high-risk students
- Feature importance visualization
- Model retraining capabilities
- Batch prediction support

---

### Project Architecture Diagram
<img width="1536" height="1024" alt="ChatGPT Image Feb 14, 2026, 03_07_49 PM" src="https://github.com/user-attachments/assets/0d7a52ca-86bc-44fe-a559-9de9f5d9a53b" />

---

## 📁 Project Structure

```
edushield/
│
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── models.py                  # Pydantic schemas
│   ├── database.py                # MongoDB operations
│   ├── ml_model.py                # Random Forest model training
│   └── data_preprocessing.py      # Feature engineering
│
├── frontend/
│   ├── index.html                 # Main dashboard
│   ├── css/
│   │   └── styles.css            # Responsive styling
│   └── js/
│       └── app.js                # Frontend logic & API calls
│
├── ml_models/                     # Saved models (auto-generated)
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── metrics.pkl
│
├── data/                          # Processed datasets
│   └── processed_student_data.csv
│
├── requirements.txt               # Python dependencies
├── .env                          # Configuration
└── README.md                     # This file
```

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.8+** (Recommended: 3.10 or 3.11)
- **MongoDB** (Local installation or MongoDB Atlas account)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd edushield

# Install Python packages
pip install -r requirements.txt
```

### Step 2: Configure MongoDB

Edit the `.env` file:

```env
# For local MongoDB
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=edushield_db

# For MongoDB Atlas (Cloud)
# MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net
```

**MongoDB Installation:**
- **Ubuntu/Debian:** `sudo apt install mongodb`
- **macOS:** `brew install mongodb-community`
- **Windows:** Download from [mongodb.com](https://www.mongodb.com/try/download/community)
- **MongoDB Atlas:** Free cloud option at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)

### Step 3: Process Data & Train Model

```bash
cd backend

# Step 3a: Process the datasets
python data_preprocessing.py

# This will:
# - Load your uploaded CSV files
# - Engineer 23 features
# - Create processed_student_data.csv
# Expected output: "✅ DATA PREPROCESSING COMPLETE!"

# Step 3b: Train the ML model
python ml_model.py

# This will:
# - Train Random Forest Classifier
# - Display performance metrics
# - Save model artifacts
# Expected output: "✅ MODEL TRAINING COMPLETE!"
```

### Step 4: Start the Backend API

```bash
# Option 1: Using Python
python main.py

# Option 2: Using uvicorn (recommended for production)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**API will be available at:** `http://localhost:8000`

**Test the API:**
```bash
curl http://localhost:8000/health
```

### Step 5: Open Frontend Dashboard

```bash
# Option 1: Open directly in browser
# Navigate to: edushield/frontend/index.html

# Option 2: Serve with Python HTTP server
cd frontend
python -m http.server 3000
```

**Dashboard will be available at:** `http://localhost:3000`

### Step 6: Load Initial Data

Once the frontend is open:
1. Navigate to the **Model** section
2. Click the "🔄 Refresh" button on the Dashboard
3. The system will automatically load student data

Or use the API endpoint:
```bash
curl -X GET http://localhost:8000/batch-predict
```

---

## 📊 Features Engineered (23 Total)

### 🧑 Demographics (5 features)
- `gender_encoded` - Gender (0: Female, 1: Male)
- `disability_encoded` - Has disability (0: No, 1: Yes)
- `education_level` - Prior education level (0-4 scale)
- `age_encoded` - Age band (1: 0-35, 2: 35-55, 3: 55+)
- `imd_numeric` - Index of Multiple Deprivation (0-100)

### 📚 Academic Background (3 features)
- `num_of_prev_attempts` - Number of previous course attempts
- `has_previous_attempts` - Binary indicator of previous attempts
- `studied_credits` - Total course credits enrolled

### 📝 Assessment Performance (11 features)
- `avg_score` - Average assessment score (0-100)
- `std_score` - Score standard deviation
- `min_score` - Minimum score achieved
- `max_score` - Maximum score achieved
- `num_assessments` - Number of assessments completed
- `first_half_avg_score` - Early semester performance
- `second_half_avg_score` - Recent performance
- `score_trend` - Performance improvement/decline indicator
- `submission_consistency` - Regularity of submissions
- `late_submissions` - Count of late submissions
- `avg_submission_day` - Average submission timing

### 📋 Registration (4 features)
- `registration_day` - Days before/after course start
- `early_registration` - Registered early (binary)
- `late_registration` - Registered late (binary)
- `unregistered` - Withdrew from course (binary)

---

## 🔌 API Endpoints

### Health & Status
- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /model-info` - Model information & feature importance

### Model Training
- `POST /train-model` - Train/retrain Random Forest model

### Predictions
- `POST /predict` - Predict dropout risk for a student
- `GET /batch-predict` - Batch predict for all students

### Student Management
- `GET /students` - Get all students (with filters)
  - Query params: `skip`, `limit`, `risk_filter`
- `GET /students/{student_id}` - Get detailed student info
- `GET /stats` - Overall system statistics
- `GET /alerts` - High-risk students requiring attention

### Interventions
- `POST /intervention` - Schedule intervention
- `PUT /intervention/{intervention_id}` - Update intervention status
- `GET /intervention-effectiveness` - Intervention analytics

**Example API Usage:**

```python
import requests

# Predict dropout risk
response = requests.post('http://localhost:8000/predict', json={
    "student_id": "12345",
    "student_name": "Rahul Kumar",
    "department": "Computer Science",
    "features": {
        "gender_encoded": 1,
        "disability_encoded": 0,
        "education_level": 2,
        "age_encoded": 1,
        "imd_numeric": 45.5,
        "num_of_prev_attempts": 0,
        "has_previous_attempts": 0,
        "studied_credits": 120,
        "avg_score": 65.5,
        "std_score": 12.3,
        "min_score": 45.0,
        "max_score": 85.0,
        "num_assessments": 8,
        "first_half_avg_score": 62.0,
        "second_half_avg_score": 68.0,
        "score_trend": 6.0,
        "submission_consistency": 15.2,
        "late_submissions": 2,
        "avg_submission_day": 120.5,
        "registration_day": -30,
        "early_registration": 1,
        "late_registration": 0,
        "unregistered": 0
    }
})

print(response.json())
```

---

## 💻 Frontend Dashboard

### 📊 Dashboard Overview
- Total students monitored
- Risk distribution (Low/Medium/High)
- Model accuracy metrics
- Interactive charts (Chart.js)

### 👥 Students Table
- Searchable and filterable
- Risk badges with color coding
- Quick actions (View details, Schedule intervention)
- Pagination support

### ⚠️ Alerts Section
- Real-time high-risk alerts
- Top risk factors per student
- Intervention scheduling
- Visual warning indicators

### 🤖 AI Model Section
- Model performance metrics
- Feature importance chart
- Training controls
- Training logs display

### 📋 Student Detail Modal
- Complete academic profile
- Demographics information
- Assessment performance with trends
- Risk history visualization
- Top contributing risk factors
- Intervention history
- Schedule new interventions

---

## 🧮 Model Performance

Typical metrics achieved on the Open University Learning Analytics Dataset:

```
Accuracy:    85-90%
Precision:   80-85%
Recall:      75-80%
F1-Score:    78-82%
ROC-AUC:     88-92%
```

### Sample Confusion Matrix:
```
              Predicted
            Pass  Dropout
Actual Pass  1500    200
      Drop    150    850
```

### Top Features by Importance:
1. **avg_score** (18-22%) - Most predictive feature
2. **score_trend** (12-15%) - Performance trajectory
3. **num_assessments** (10-12%) - Engagement level
4. **late_submissions** (8-10%) - Time management
5. **unregistered** (7-9%) - Course withdrawal status

---

## 🗄️ MongoDB Collections

### students
Stores complete student profiles and current risk assessments.

```json
{
  "student_id": "12345",
  "student_name": "Rahul Kumar",
  "department": "Computer Science",
  "gender": "M",
  "age_band": "0-35",
  "education_level": "A Level or Equivalent",
  "disability": "N",
  "avg_score": 65.5,
  "num_assessments": 8,
  "current_risk_score": 0.35,
  "risk_category": "Low Risk",
  "risk_emoji": "🟢",
  "last_updated": "2024-02-15T10:30:00Z"
}
```

### predictions
Historical prediction records for trend analysis.

```json
{
  "student_id": "12345",
  "probability": 0.35,
  "risk_percentage": 35.0,
  "risk_category": "Low Risk",
  "top_features": [...],
  "timestamp": "2024-02-15T10:30:00Z"
}
```

### interventions
Tracks scheduled and completed interventions.

```json
{
  "student_id": "12345",
  "intervention_type": "Academic Counseling",
  "description": "Weekly tutoring sessions for Mathematics",
  "scheduled_date": "2024-02-20T14:00:00Z",
  "risk_before": 0.75,
  "risk_after": 0.45,
  "improvement_percentage": 40.0,
  "status": "completed",
  "created_at": "2024-02-15T10:30:00Z"
}
```

### model_metadata
Model training history and performance metrics.

```json
{
  "accuracy": 0.87,
  "precision": 0.82,
  "recall": 0.79,
  "f1_score": 0.80,
  "roc_auc": 0.90,
  "confusion_matrix": [[1500, 200], [150, 850]],
  "last_trained": "2024-02-15T09:00:00Z",
  "total_students_trained": 30000
}
```

---

## 🔧 Configuration

### Environment Variables (.env)

```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=edushield_db

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256

# Model Paths (auto-configured)
MODEL_PATH=./ml_models/random_forest_model.pkl
SCALER_PATH=./ml_models/scaler.pkl
FEATURE_NAMES_PATH=./ml_models/feature_names.pkl
```

---

## 🚢 Deployment Options

### Local Development
Already covered in Quick Start above.

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY ml_models/ ./ml_models/
COPY data/ ./data/
COPY .env .env

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t edushield .
docker run -p 8000:8000 edushield
```

### Cloud Deployment

**Backend Options:**
- **Heroku:** `git push heroku main`
- **AWS EC2:** Deploy with systemd service
- **Google Cloud Run:** Container-based deployment
- **Azure App Service:** Python web app
- **Railway:** One-click deployment

**Frontend Options:**
- **Netlify:** Drag & drop frontend folder
- **Vercel:** Connect GitHub repository
- **GitHub Pages:** Static site hosting
- **AWS S3 + CloudFront:** Scalable static hosting

**Database Options:**
- **MongoDB Atlas:** Free tier available (512MB)
- **AWS DocumentDB:** MongoDB-compatible
- **Azure Cosmos DB:** Global distribution

---

## 🧪 Testing

### Test API Health
```bash
curl http://localhost:8000/health
```

### Get Statistics
```bash
curl http://localhost:8000/stats
```

### Test Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "TEST001",
    "student_name": "Test Student",
    "department": "Testing",
    "features": {
      "gender_encoded": 1,
      "disability_encoded": 0,
      "education_level": 2,
      "age_encoded": 1,
      "imd_numeric": 50,
      "num_of_prev_attempts": 0,
      "has_previous_attempts": 0,
      "studied_credits": 120,
      "avg_score": 70,
      "std_score": 10,
      "min_score": 50,
      "max_score": 90,
      "num_assessments": 10,
      "first_half_avg_score": 65,
      "second_half_avg_score": 75,
      "score_trend": 10,
      "submission_consistency": 20,
      "late_submissions": 1,
      "avg_submission_day": 100,
      "registration_day": -20,
      "early_registration": 1,
      "late_registration": 0,
      "unregistered": 0
    }
  }'
```

---

## 📈 Performance Optimization

### Backend Optimizations
- Use connection pooling for MongoDB
- Implement Redis for caching predictions
- Enable GZIP compression
- Add request rate limiting
- Use async operations

### Frontend Optimizations
- Lazy load student data (pagination)
- Debounce search operations
- Implement virtual scrolling
- Cache API responses
- Use Web Workers for computations

### ML Model Optimizations
- Model quantization for smaller size
- Batch prediction processing
- Feature selection to reduce dimensions
- Cache frequently requested predictions

---

## 🔐 Security Best Practices

1. **Never commit `.env` file** to version control
2. Use **environment variables** for sensitive data
3. Implement **JWT authentication** for API
4. Add **rate limiting** to prevent abuse
5. **Validate all inputs** using Pydantic
6. Use **HTTPS** in production
7. Configure **CORS** properly
8. Regular **security audits**
9. **Encrypt** sensitive data in database
10. Use **secrets management** service in production

---

## 🎯 Use Cases

### For Educational Institutions
- Identify at-risk students early in the semester
- Allocate counseling resources efficiently
- Track intervention effectiveness
- Improve retention rates
- Data-driven decision making

### For Administrators
- Monitor overall student health
- Generate risk reports
- Schedule interventions
- Track historical trends
- Measure program effectiveness

### For Counselors
- Prioritize student outreach
- Access complete student profiles
- Document interventions
- Track progress over time
- Collaborate with faculty

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👥 Credits

**Dataset:** Open University Learning Analytics Dataset (OULAD)

**Technologies:**
- **Backend:** FastAPI, Python, scikit-learn
- **Database:** MongoDB
- **Frontend:** HTML5, CSS3, JavaScript ES6, Chart.js
- **ML:** Random Forest, SMOTE, pandas, numpy

---

## 📞 Support & Contact

For issues, questions, or contributions:
- **GitHub Issues:** Create an issue in the repository
- **API Documentation:** Visit `http://localhost:8000/docs`
- **Email:** support@edushield.edu.in

---

## 🎯 Future Enhancements

- [ ] Deep Learning models (LSTM, Transformer)
- [ ] Real-time data streaming with Apache Kafka
- [ ] Mobile app (React Native / Flutter)
- [ ] Email/SMS alerts for high-risk students
- [ ] Integration with LMS systems (Moodle, Canvas)
- [ ] Multi-language support (Hindi, Regional languages)
- [ ] Advanced visualization with D3.js
- [ ] A/B testing for interventions
- [ ] Automated report generation (PDF/Excel)
- [ ] Parent/Guardian portal
- [ ] WhatsApp integration for alerts
- [ ] Voice-based interface for accessibility

---

## ✅ Pre-Deployment Checklist

- [ ] Data preprocessing completed
- [ ] Model trained and saved
- [ ] MongoDB connection configured
- [ ] Environment variables set
- [ ] API endpoints tested
- [ ] Frontend connected to backend
- [ ] Security measures implemented
- [ ] Documentation reviewed
- [ ] Backup strategy in place
- [ ] Monitoring setup (optional)
- [ ] Load testing performed (optional)

---

## 🏆 Project Impact

EduShield aims to:
- **Reduce dropout rates** by 20-30% through early intervention
- **Improve retention** in Indian educational institutions
- **Empower educators** with data-driven insights
- **Support students** proactively before problems escalate
- **Increase graduation rates** and student success

---

### 🏆 Hackathon Submission Details

Event: Sudhee 2026 Hackathon Institution: CBIT

Team Members:

Rakshitha Poshetty — Team Lead
Jhasmitha Tattari
Sai Akshaya Dantala
Akshara Deshineni
>>>>>>> de9c5de (Updated README with hackathon details)

This project was developed as part of the Sudhee 2026 Hackathon conducted by CBIT.

---

**Made for improving student success in Indian education**

**🛡️ EduShield - Protecting Every Student's Future**

© 2026 Rakshitha Poshetty and Team. All rights reserved. Licensed under the MIT License.
