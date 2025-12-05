#!/bin/bash

echo "🚀 Setting up AI Anti-Spam Shield Service..."
echo "============================================="

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r app/requirements.txt

# Create sample dataset
echo "📊 Creating sample dataset..."
cd datasets
python3 prepare_data.py
cd ..

# Train model
echo "🎯 Training spam classification model..."
cd app
python3 model/train.py

echo ""
echo "============================================="
echo "✅ Setup complete!"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  cd app"
echo "  python main.py"
echo ""
echo "Or use:"
echo "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo "============================================="

