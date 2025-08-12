#!/usr/bin/env python3
"""
Test script for local Ollama LLM integration with StockVision

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Pull gemma2:1b model: ollama pull gemma2:1b
3. Start Ollama service: ollama serve (if not running as daemon)
"""

import requests
import json

API_BASE = "http://localhost:8001"

def test_local_text_only():
    """Test local LLM with text-only context"""
    print("🔤 Testing local LLM with text-only context...")
    
    data = {
        "context": "DSE stock XYZ has shown 20% growth in Q4 2024. The company operates in the textile sector and has strong export revenues.",
        "question": "What factors might drive continued growth for this textile company?"
    }
    
    response = requests.post(f"{API_BASE}/api/ai/explain/local", json=data)
    if response.status_code == 200:
        print("✅ Success!")
        print("Local LLM Response:", response.json()["explanation"][:300] + "...")
    else:
        print("❌ Failed:", response.text)
    print()

def test_local_multimodal():
    """Test local LLM with multimodal context (mostly images)"""
    print("🖼️  Testing local LLM with image-heavy context...")
    
    # This will trigger local LLM due to minimal text + images
    data = {
        "context": [
            {
                "type": "text",
                "content": "Stock chart"  # Very minimal text
            },
            {
                "type": "image",
                "content": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            },
            {
                "type": "image", 
                "content": "https://invalid-chart-url.com/chart.png"
            }
        ],
        "question": "What can you tell me about this financial data?"
    }
    
    response = requests.post(f"{API_BASE}/api/ai/explain/", json=data)  # Using main endpoint (should auto-switch to local)
    if response.status_code == 200:
        print("✅ Success! (Auto-switched to local LLM)")
        print("Response:", response.json()["explanation"][:300] + "...")
    else:
        print("❌ Failed:", response.text)
    print()

def test_comparison_google_vs_local():
    """Compare Google AI vs Local LLM responses"""
    print("⚖️  Comparing Google AI vs Local LLM...")
    
    context = "BEXIMCO stock has shown volatile performance with 15% gains followed by 8% decline. The conglomerate operates in textiles, pharmaceuticals, and energy sectors."
    question = "What diversification benefits does this conglomerate structure provide?"
    
    data = {
        "context": context,
        "question": question
    }
    
    # Test Google AI
    print("🌐 Google AI Response:")
    response_google = requests.post(f"{API_BASE}/api/ai/explain/", json=data)
    if response_google.status_code == 200:
        print("✅", response_google.json()["explanation"][:200] + "...")
    else:
        print("❌", response_google.text[:100])
    
    print()
    
    # Test Local LLM
    print("🏠 Local LLM Response:")
    response_local = requests.post(f"{API_BASE}/api/ai/explain/local", json=data)
    if response_local.status_code == 200:
        print("✅", response_local.json()["explanation"][:200] + "...")
    else:
        print("❌", response_local.text[:100])
    
    print()

def check_ollama_status():
    """Check if Ollama is running and has the required model"""
    print("🔍 Checking Ollama status...")
    
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "gemma2:1b" in result.stdout:
            print("✅ Ollama is running and gemma2:1b model is available")
            return True
        else:
            print("❌ gemma2:1b model not found. Run: ollama pull gemma2:1b")
            return False
    except FileNotFoundError:
        print("❌ Ollama not installed. Install from: https://ollama.ai/")
        return False
    except Exception as e:
        print(f"⚠️  Could not check Ollama status: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing StockVision Local LLM Integration\n")
    
    # Check prerequisites
    if not check_ollama_status():
        print("\n💡 Setup Instructions:")
        print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Pull model: ollama pull gemma2:1b")
        print("3. Restart this test")
        exit(1)
    
    print()
    
    # Run tests
    test_local_text_only()
    test_local_multimodal()
    test_comparison_google_vs_local()
    
    print("✨ Testing complete!")
    
    print("""
💡 Usage Summary:

1. Main endpoint (auto-fallback to local): POST /api/ai/explain/
   - Uses Google AI for text-heavy content
   - Auto-switches to local LLM for image-heavy content with minimal text
   
2. Force local LLM: POST /api/ai/explain/local
   - Always uses local Ollama model
   - Good for privacy-sensitive analysis
   
3. Local LLM handles:
   - Pure text financial analysis
   - Image descriptions (acknowledges images but analyzes text context)
   - Fallback when Google AI is unavailable
""")
