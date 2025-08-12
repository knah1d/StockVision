#!/usr/bin/env python3
"""
Test script for file upload functionality with StockVision LLM API

This script demonstrates:
1. Creating sample stock chart images
2. Uploading images for analysis
3. Getting beginner-friendly explanations
"""

import requests
import json
from PIL import Image, ImageDraw, ImageFont
import io
import random
import numpy as np

API_BASE = "http://localhost:8001"

def create_sample_stock_chart(filename="sample_chart.png"):
    """Create a sample stock chart image for testing"""
    print(f"📊 Creating sample stock chart: {filename}")
    
    # Create a simple stock chart
    width, height = 800, 600
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw axes
    draw.line([(50, height-50), (width-50, height-50)], fill='black', width=2)  # X-axis
    draw.line([(50, 50), (50, height-50)], fill='black', width=2)  # Y-axis
    
    # Draw title
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((width//2-50, 20), "Stock Price Chart", fill='black', font=font)
    draw.text((width//2-50, 40), "AAPL - 6 Month View", fill='gray', font=font)
    
    # Generate sample price data
    days = 150
    start_price = 150
    prices = [start_price]
    
    for i in range(days):
        change = random.uniform(-0.05, 0.05) * prices[-1]
        new_price = max(prices[-1] + change, 50)  # Don't go below $50
        prices.append(new_price)
    
    # Normalize prices to fit in chart
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price
    
    # Draw price line
    points = []
    for i, price in enumerate(prices):
        x = 50 + (i / len(prices)) * (width - 100)
        y = height - 50 - ((price - min_price) / price_range) * (height - 100)
        points.append((x, y))
    
    # Draw the price line
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill='blue', width=2)
    
    # Add some labels
    draw.text((60, height-70), f"${min_price:.1f}", fill='black', font=font)
    draw.text((60, 60), f"${max_price:.1f}", fill='black', font=font)
    draw.text((width-100, height-30), "6 months", fill='black', font=font)
    draw.text((60, height-30), "Today", fill='black', font=font)
    
    # Add some trend indicators
    if prices[-1] > prices[0]:
        draw.text((width-150, 80), "↗ Upward Trend", fill='green', font=font)
    else:
        draw.text((width-150, 80), "↘ Downward Trend", fill='red', font=font)
    
    # Save image
    img.save(filename)
    print(f"✅ Sample chart saved as {filename}")
    return filename

def test_file_upload_google():
    """Test file upload with Google AI"""
    print("\n🌐 Testing file upload with Google AI...")
    
    # Create sample chart
    chart_file = create_sample_stock_chart("test_chart_google.png")
    
    # Upload and analyze
    with open(chart_file, 'rb') as f:
        files = {'file': ('chart.png', f, 'image/png')}
        data = {
            'question': 'What does this chart tell me about the stock performance?',
            'context_text': 'This is a 6-month stock price chart for Apple (AAPL) showing daily price movements.',
            'use_local': False
        }
        
        response = requests.post(f"{API_BASE}/api/ai/explain/upload", files=files, data=data)
        
        if response.status_code == 200:
            print("✅ Success!")
            explanation = response.json()["explanation"]
            print(f"📖 Explanation: {explanation[:300]}...")
        else:
            print(f"❌ Failed: {response.text}")

def test_file_upload_local():
    """Test file upload with local LLM"""
    print("\n🏠 Testing file upload with Local LLM...")
    
    # Create sample chart
    chart_file = create_sample_stock_chart("test_chart_local.png")
    
    # Upload and analyze with local LLM
    with open(chart_file, 'rb') as f:
        files = {'file': ('chart.png', f, 'image/png')}
        data = {
            'question': 'Explain this chart for a beginner investor',
            'context_text': 'This is a stock chart showing price movements over time. Please explain what beginners should understand from this.'
        }
        
        response = requests.post(f"{API_BASE}/api/ai/explain/upload/local", files=files, data=data)
        
        if response.status_code == 200:
            print("✅ Success!")
            explanation = response.json()["explanation"]
            print(f"📖 Local LLM Explanation: {explanation[:300]}...")
        else:
            print(f"❌ Failed: {response.text}")

def test_beginner_questions():
    """Test various beginner-friendly questions"""
    print("\n🎓 Testing beginner-friendly questions...")
    
    questions = [
        "What should I look for in this chart as a new investor?",
        "Is this stock going up or down? What does that mean?",
        "How do I read this chart? What are the key things to understand?",
        "Should a beginner invest in this stock based on this chart?",
        "What are the risks and opportunities shown in this chart?"
    ]
    
    chart_file = create_sample_stock_chart("test_chart_beginner.png")
    
    for i, question in enumerate(questions, 1):
        print(f"\n📚 Question {i}: {question}")
        
        with open(chart_file, 'rb') as f:
            files = {'file': ('chart.png', f, 'image/png')}
            data = {
                'question': question,
                'context_text': 'This is a stock market chart. Please explain it in simple terms for someone new to investing.',
                'use_local': False
            }
            
            response = requests.post(f"{API_BASE}/api/ai/explain/upload", files=files, data=data)
            
            if response.status_code == 200:
                explanation = response.json()["explanation"]
                print(f"💡 Answer: {explanation[:200]}...")
            else:
                print(f"❌ Failed: {response.text[:100]}")

def cleanup_test_files():
    """Clean up test files"""
    import os
    test_files = [
        "test_chart_google.png",
        "test_chart_local.png", 
        "test_chart_beginner.png",
        "sample_chart.png"
    ]
    
    for file in test_files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"🗑️  Cleaned up {file}")
        except Exception as e:
            print(f"⚠️  Could not remove {file}: {e}")

if __name__ == "__main__":
    print("🚀 Testing StockVision File Upload with Beginner-Friendly Analysis\n")
    
    try:
        # Test file uploads
        test_file_upload_google()
        test_file_upload_local()
        test_beginner_questions()
        
        print("\n✨ Testing complete!")
        
        print("""
💡 File Upload API Summary:

1. 📤 Upload with Google AI: POST /api/ai/explain/upload
   - Supports multimodal analysis (text + images)
   - Best for detailed technical analysis
   
2. 🏠 Upload with Local LLM: POST /api/ai/explain/upload/local
   - Uses local Ollama model
   - Privacy-friendly, works offline
   
3. 📊 Perfect for:
   - Stock chart analysis
   - Beginner-friendly explanations
   - Technical pattern recognition
   - Risk assessment

4. 📁 File Storage:
   - Images saved to temp_uploads/ folder
   - Auto-cleanup after 24 hours
   - Supports PNG, JPG, GIF formats
""")
        
    except requests.ConnectionError:
        print("❌ Backend not running. Start with:")
        print("   python backend/main.py")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cleanup_test_files()
