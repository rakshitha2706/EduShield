#!/bin/bash

# EduShield - Startup Script
# This script starts both backend and frontend

echo "🛡️  Starting EduShield System..."
echo ""

# Check if MongoDB is running
echo "📊 Checking MongoDB..."
if pgrep -x "mongod" > /dev/null
then
    echo "✅ MongoDB is running"
else
    echo "⚠️  MongoDB is not running. Please start MongoDB first:"
    echo "   Ubuntu/Debian: sudo systemctl start mongodb"
    echo "   macOS: brew services start mongodb-community"
    echo "   Windows: net start MongoDB"
    echo ""
    echo "Or use MongoDB Atlas (cloud): Update MONGODB_URL in .env"
    echo ""
fi

# Check if processed data exists
if [ ! -f "data/processed_student_data.csv" ]; then
    echo "📚 Processed data not found. Running data preprocessing..."
    cd backend
    python data_preprocessing.py
    cd ..
    echo ""
fi

# Check if model exists
if [ ! -f "ml_models/random_forest_model.pkl" ]; then
    echo "🤖 Model not found. Training model..."
    cd backend
    python ml_model.py
    cd ..
    echo ""
fi

# Start backend
echo "🚀 Starting Backend API..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

echo "✅ Backend started (PID: $BACKEND_PID)"
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""

# Wait for backend to start
sleep 3

# Start frontend
echo "🌐 Starting Frontend..."
cd frontend
python -m http.server 3000 &
FRONTEND_PID=$!
cd ..

echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo "   Dashboard: http://localhost:3000"
echo ""

echo "============================================================"
echo "🛡️  EDUSHIELD IS RUNNING!"
echo "============================================================"
echo ""
echo "📊 Dashboard: http://localhost:3000"
echo "🔌 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user interrupt
trap "echo ''; echo '🛑 Stopping EduShield...'; kill $BACKEND_PID $FRONTEND_PID; echo '✅ Services stopped'; exit 0" INT

wait