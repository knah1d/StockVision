#!/bin/bash

# StockVision API Startup Script

echo "🚀 Starting StockVision API Server..."
echo "📊 Make sure you have the required data files in data/processed/"
echo "⚡ Installing dependencies..."

# Install requirements
python3 -m pip install -r requirements.txt

echo "🔥 Starting FastAPI server..."
echo "📍 API will be available at: http://localhost:8000"
echo "📚 API Docs will be available at: http://localhost:8000/docs"
echo ""

# Start the server
python3 main.py
