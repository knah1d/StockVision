#!/usr/bin/env python3
"""
Practical example: Using StockVision's multimodal LLM API for real financial analysis
"""

import requests
import json

API_BASE = "http://localhost:8001"

def analyze_with_real_data():
    """Example with real DSE stock data context"""
    print("📊 Real DSE Stock Analysis Example")
    print("=" * 50)
    
    # Get some real data from your StockVision API first
    stats_response = requests.get(f"{API_BASE}/api/analysis/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        
        # Get ticker info
        tickers_response = requests.get(f"{API_BASE}/api/analysis/tickers?limit=5")
        if tickers_response.status_code == 200:
            tickers = tickers_response.json()["tickers"]
            
            # Create context with real data
            context_text = f"""
            DSE Market Overview:
            - Total Records: {stats['data_shape'][0]:,}
            - Total Tickers: {stats['total_tickers']}
            - Total Sectors: {stats['total_sectors']}
            - Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}
            
            Top Tickers:
            """
            
            for ticker in tickers[:3]:
                context_text += f"- {ticker['ticker']}: {ticker['company_name']}\n"
            
            # Ask for analysis
            data = {
                "context": context_text,
                "question": "Based on this DSE market data, what are the key characteristics of this market? What insights can you provide about market composition and timeframe?"
            }
            
            response = requests.post(f"{API_BASE}/api/ai/explain/", json=data)
            if response.status_code == 200:
                print("✅ Analysis Generated!")
                print("\n📈 AI Analysis:")
                print("-" * 40)
                print(response.json()["explanation"])
            else:
                print("❌ Failed:", response.text)
    
def analyze_stock_comparison():
    """Example comparing multiple stocks"""
    print("\n🔄 Stock Comparison Analysis")
    print("=" * 50)
    
    # Example with multiple ticker analysis
    tickers_to_analyze = ["SQURPHARMA", "BEXIMCO", "GRAMEENPHONE"]
    
    comparison_context = "Stock Performance Comparison:\n"
    
    for ticker in tickers_to_analyze:
        # You could fetch real data for each ticker here
        comparison_context += f"- {ticker}: [Real performance data would go here]\n"
    
    data = {
        "context": [
            {
                "type": "text",
                "content": comparison_context
            },
            {
                "type": "text", 
                "content": "These are major stocks from Bangladesh DSE market representing pharmaceuticals, conglomerate, and telecom sectors."
            }
        ],
        "question": "What would be a good diversification strategy using these three stocks? What sectors do they represent and how might they complement each other in a portfolio?"
    }
    
    response = requests.post(f"{API_BASE}/api/ai/explain/", json=data)
    if response.status_code == 200:
        print("✅ Comparison Analysis Generated!")
        print("\n💼 Portfolio Analysis:")
        print("-" * 40)
        print(response.json()["explanation"])
    else:
        print("❌ Failed:", response.text)

def test_chart_analysis():
    """Example of how to analyze uploaded charts"""
    print("\n📊 Chart Analysis Example")
    print("=" * 50)
    
    # This would be for when you upload a real chart image
    instructions = """
To analyze stock charts with images:

1. Convert your chart to base64:
   import base64
   with open("stock_chart.png", "rb") as img_file:
       b64_string = base64.b64encode(img_file.read()).decode('utf-8')

2. Send to API:
   {
     "context": [
       {
         "type": "text",
         "content": "This is a 6-month chart for SQURPHARMA showing daily prices and volume"
       },
       {
         "type": "image",
         "content": "data:image/png;base64," + b64_string
       }
     ],
     "question": "What technical patterns do you see? What are the support and resistance levels?"
   }

3. The AI will analyze:
   - Price trends and patterns
   - Support/resistance levels  
   - Volume patterns
   - Technical indicators
   - Potential breakout points
    """
    
    print(instructions)

if __name__ == "__main__":
    print("🚀 StockVision Multimodal LLM - Practical Examples\n")
    
    try:
        analyze_with_real_data()
        analyze_stock_comparison() 
        test_chart_analysis()
        
        print("\n" + "=" * 60)
        print("✨ Examples complete!")
        print("\n💡 Tips:")
        print("- Upload stock charts as base64 images for technical analysis")
        print("- Combine market data with news for comprehensive insights")
        print("- Ask specific questions about patterns, trends, and strategies")
        print("- Use sector comparison for diversification advice")
        
    except requests.ConnectionError:
        print("❌ Backend not running. Start with:")
        print("   python backend/main.py")
    except Exception as e:
        print(f"❌ Error: {e}")
