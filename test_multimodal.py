#!/usr/bin/env python3
"""
Test script demonstrating multimodal (text + image) context with the StockVision LLM API

This script shows how to:
1. Use text-only context
2. Use image-only context  
3. Use both text and image context together
"""

import requests
import base64
import json

API_BASE = "http://localhost:8001"

def test_text_only():
    """Test with text-only context"""
    print("🔤 Testing text-only context...")
    
    data = {
        "context": "AAPL stock has risen 15% in Q4 2024. The company reported strong iPhone sales and AI integration.",
        "question": "What factors might be driving this stock performance?"
    }
    
    response = requests.post(f"{API_BASE}/api/ai/explain/", json=data)
    if response.status_code == 200:
        print("✅ Success!")
        print("Response:", response.json()["explanation"][:200] + "...")
    else:
        print("❌ Failed:", response.text)
    print()

def test_image_only():
    """Test with image-only context"""
    print("🖼️  Testing image-only context...")
    
    # Using a working stock chart image URL (Yahoo Finance)
    data = {
        "context": [
            {
                "type": "image",
                "content": "https://chart.yahoo.com/z?s=AAPL&t=1y&q=l&l=on&z=s&p=m50,m200"
            }
        ],
        "question": "What can you tell me about this stock chart?"
    }
    
    response = requests.post(f"{API_BASE}/api/ai/explain/", json=data)
    if response.status_code == 200:
        print("✅ Success!")
        print("Response:", response.json()["explanation"][:200] + "...")
    else:
        print("❌ Failed:", response.text)
    print()

def test_multimodal():
    """Test with both text and image context"""
    print("🔤🖼️  Testing multimodal context (text + image)...")
    
    data = {
        "context": [
            {
                "type": "text",
                "content": "This is Apple's stock performance chart. The company has been focusing on AI integration and services revenue growth."
            },
            {
                "type": "image", 
                "content": "https://chart.yahoo.com/z?s=AAPL&t=6m&q=l&l=on&z=s&p=m20"
            },
            {
                "type": "text",
                "content": "Recent news: Apple announced new AI features and increased services revenue in latest earnings."
            }
        ],
        "question": "Based on the chart and recent developments, what's your analysis of Apple's prospects?"
    }
    
    response = requests.post(f"{API_BASE}/api/ai/explain/", json=data)
    if response.status_code == 200:
        print("✅ Success!")
        print("Response:", response.json()["explanation"][:200] + "...")
    else:
        print("❌ Failed:", response.text)
    print()

def test_base64_image():
    """Test with base64 encoded image"""
    print("📷 Testing base64 image context...")
    
    # Example: If you have a local image file
    # with open("stock_chart.png", "rb") as img_file:
    #     b64_string = base64.b64encode(img_file.read()).decode('utf-8')
    
    # For demo, using a small placeholder
    b64_string = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    data = {
        "context": [
            {
                "type": "text",
                "content": "This is a stock chart showing recent price movements."
            },
            {
                "type": "image",
                "content": f"data:image/png;base64,{b64_string}"
            }
        ],
        "question": "What does this chart show?"
    }
    
    response = requests.post(f"{API_BASE}/api/ai/explain/", json=data)
    if response.status_code == 200:
        print("✅ Success!")
        print("Response:", response.json()["explanation"][:200] + "...")
    else:
        print("❌ Failed:", response.text)
    print()

def test_invalid_url():
    """Test with invalid image URL to show graceful handling"""
    print("⚠️  Testing invalid image URL handling...")
    
    data = {
        "context": [
            {
                "type": "text",
                "content": "Here's some context about a stock."
            },
            {
                "type": "image",
                "content": "https://invalid-url-that-does-not-exist.com/chart.png"
            }
        ],
        "question": "What can you tell me about this data?"
    }
    
    response = requests.post(f"{API_BASE}/api/ai/explain/", json=data)
    if response.status_code == 200:
        print("✅ Success! (Invalid URL handled gracefully)")
        print("Response:", response.json()["explanation"][:200] + "...")
    else:
        print("❌ Failed:", response.text)
    print()

if __name__ == "__main__":
    print("🚀 Testing StockVision Multimodal LLM API\n")
    
    # Test different context types
    test_text_only()
    test_invalid_url()  # Test invalid URL handling
    test_base64_image()
    # test_image_only()  # Comment out for now as it might not work with external URLs
    # test_multimodal()  # Comment out for now as it might not work with external URLs
    
    print("✨ Testing complete!")
    
    print("""
💡 Usage Examples:

1. Text Only:
{
  "context": "Your text context here",
  "question": "Your question here"
}

2. Image Only:
{
  "context": [
    {"type": "image", "content": "https://example.com/image.png"}
  ],
  "question": "What do you see in this image?"
}

3. Mixed Content:
{
  "context": [
    {"type": "text", "content": "Text context"},
    {"type": "image", "content": "data:image/jpeg;base64,/9j/4AAQ..."},
    {"type": "text", "content": "More text context"}
  ],
  "question": "Analyze both the text and image"
}
""")
