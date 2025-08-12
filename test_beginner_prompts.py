#!/usr/bin/env python3
"""
Test the improved beginner-friendly prompts for StockVision LLM API
"""

import requests
import json
from pathlib import Path

API_BASE = "http://localhost:8001"

def test_beginner_text_explanation():
    """Test beginner-friendly text explanation"""
    print("🎓 Testing Beginner-Friendly Text Analysis")
    print("=" * 50)
    
    data = {
        "context": "BEXIMCO stock has a P/E ratio of 15, ROE of 12%, and debt-to-equity ratio of 0.8. The stock has gained 25% this year but is down 5% this month due to textile sector concerns.",
        "question": "Should I invest in BEXIMCO? What do these numbers mean?"
    }
    
    response = requests.post(f"{API_BASE}/api/ai/explain/", json=data)
    if response.status_code == 200:
        print("✅ Beginner-friendly analysis generated!")
        print("\n📚 AI Response:")
        print("-" * 40)
        print(response.json()["explanation"])
    else:
        print("❌ Failed:", response.text)

def test_beginner_comparison():
    """Test beginner-friendly stock comparison"""
    print("\n🔄 Testing Beginner-Friendly Stock Comparison")
    print("=" * 50)
    
    data = {
        "context": [
            {
                "type": "text",
                "content": "Stock A: High growth tech company, P/E ratio 30, very volatile"
            },
            {
                "type": "text", 
                "content": "Stock B: Stable utility company, P/E ratio 12, pays 4% dividend, low volatility"
            }
        ],
        "question": "I'm new to investing. Which stock is better for a beginner like me?"
    }
    
    response = requests.post(f"{API_BASE}/api/ai/explain/", json=data)
    if response.status_code == 200:
        print("✅ Beginner-friendly comparison generated!")
        print("\n📊 AI Response:")
        print("-" * 40)
        print(response.json()["explanation"])
    else:
        print("❌ Failed:", response.text)

def create_sample_chart():
    """Create a simple test chart image"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import os
        
        # Create a simple stock chart-like image
        width, height = 800, 400
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw axes
        draw.line([(50, height-50), (width-50, height-50)], fill='black', width=2)  # X-axis
        draw.line([(50, 50), (50, height-50)], fill='black', width=2)  # Y-axis
        
        # Draw a simple upward trending line
        points = []
        for i in range(0, 8):
            x = 50 + (i * 90)
            y = height - 100 - (i * 25) + (10 if i % 2 == 0 else -10)  # Upward trend with fluctuations
            points.append((x, y))
        
        # Draw the trend line
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill='blue', width=3)
            draw.ellipse([points[i][0]-4, points[i][1]-4, points[i][0]+4, points[i][1]+4], fill='blue')
        
        # Add labels
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((width//2 - 50, height - 30), "Time →", fill='black', font=font)
        draw.text((10, height//2), "Price ↑", fill='black', font=font)
        draw.text((width//2 - 50, 20), "Sample Stock Chart", fill='black', font=font)
        
        # Save the image
        os.makedirs("test_images", exist_ok=True)
        img_path = "test_images/sample_stock_chart.png"
        img.save(img_path)
        return img_path
        
    except ImportError:
        print("⚠️  PIL not available, skipping image test")
        return None

def test_beginner_chart_upload():
    """Test beginner-friendly chart analysis via upload"""
    print("\n📊 Testing Beginner-Friendly Chart Upload")
    print("=" * 50)
    
    # Create a sample chart
    img_path = create_sample_chart()
    if not img_path:
        print("⚠️  Skipping chart test - no PIL")
        return
    
    try:
        with open(img_path, 'rb') as f:
            files = {'file': ('chart.png', f, 'image/png')}
            data = {
                'question': 'What does this chart show? Is this good for investing?',
                'context_text': 'This is a stock price chart showing performance over time.',
                'use_local': 'false'
            }
            
            response = requests.post(f"{API_BASE}/api/ai/explain/upload", files=files, data=data)
            
        if response.status_code == 200:
            print("✅ Beginner-friendly chart analysis generated!")
            print("\n📈 AI Response:")
            print("-" * 40)
            print(response.json()["explanation"])
        else:
            print("❌ Failed:", response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Testing Improved Beginner-Friendly Stock Market Explanations")
    print("🎯 Goal: Make investing knowledge accessible to everyone!\n")
    
    try:
        test_beginner_text_explanation()
        test_beginner_comparison()
        test_beginner_chart_upload()
        
        print("\n" + "=" * 60)
        print("✨ Beginner-friendly testing complete!")
        print("""
💡 **Key Improvements Made:**

🎓 **Better Teaching Style:**
- Uses simple, everyday language
- Explains financial terms clearly
- Includes emojis and clear structure
- Focuses on practical learning

📊 **Enhanced Prompts:**
- Structured responses with headings
- Beginner-focused explanations
- Encouraging and patient tone
- Real-world analogies and examples

🌟 **Educational Focus:**
- Emphasizes learning opportunities
- Provides actionable insights
- Reminds about investment risks
- Makes complex concepts accessible

Perfect for helping newcomers start their investment journey! 🚀
""")
        
    except requests.ConnectionError:
        print("❌ Backend not running. Start with:")
        print("   python backend/main.py")
    except Exception as e:
        print(f"❌ Error: {e}")
