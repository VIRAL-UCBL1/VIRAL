#!/bin/bash

echo "🚀 Starting the Video Rating application..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install it first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install it first."
    exit 1
fi

# 1️⃣ Start the Flask backend
echo "📡 Starting the Flask server..."
cd backend

# Check if Conda is available
if command -v conda &> /dev/null; then
    echo "🐍 Conda found. Attempting to use Conda environment..."

    # Load Conda (adjust the path based on your installation)
    source $HOME/miniconda3/etc/profile.d/conda.sh

    # Check if the "video" Conda environment exists, if not, create it
    if ! conda info --envs | grep -q "video"; then
        echo "🐍 Creating the Conda environment 'video'..."
        conda create --name video python=3.8 -y
    fi

    # Activate the Conda environment
    conda activate video

    # Install dependencies if necessary
    pip install -r requirements.txt
else
    echo "❌ Conda is not available. Using venv instead."

    # Check if virtual environment exists, if not, create it
    if [ ! -d "venv" ]; then
        echo "🔧 Creating a virtual environment..."
        python3 -m venv venv
    fi

    # Activate the virtual environment
    source venv/bin/activate

    # Install dependencies if necessary
    pip install -r requirements.txt
fi

# Run Flask in the background and store its PID
python app.py &  
FLASK_PID=$!

# Return to the project root
cd ..

# 2️⃣ Start the Vue.js frontend
echo "🌐 Starting the Vue.js frontend..."
cd frontend

# Check if `node_modules` exists, if not, install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Vue.js dependencies..."
    npm install
fi

# Run Vue.js in the background and store its PID
npm run dev &  
VUE_PID=$!

# Return to the project root
cd ..

# Function to properly stop processes
cleanup() {
    echo "🛑 Stopping the applications..."

    # Stop Flask and Vue.js
    kill $FLASK_PID
    kill $VUE_PID

    echo "✅ Applications stopped."
}

# Catch Ctrl+C and call cleanup
trap cleanup SIGINT

echo "✅ Everything is running!"
echo "🔹 Flask: http://127.0.0.1:5000"
echo "🔹 Vue.js: http://localhost:5173"

# Keep the processes running until interrupted
wait
