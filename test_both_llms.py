#!/usr/bin/env python3
"""
Test script to verify both Gemini-2.0-Flash and local Gemma3:1b
give brief, beginner-friendly explanations
"""
import os
import sys
sys.path.append('/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision')

from backend.services.llm_service import LLMService

def test_brief_explanations():
    """Test that both LLMs provide brief explanations"""
    print("🔍 Testing Brief Explanations from Both LLMs")
    print("=" * 60)
    
    # Initialize the service
    LLMService.initialize_ai_components()
    LLMService.initialize_local_llm()
    
    # Test question
    test_question = "What does P/E ratio mean?"
    test_context = "Apple Inc. has a P/E ratio of 25, while the industry average is 18."
    
    print("📝 Test Question:", test_question)
    print("📊 Test Context:", test_context)
    print()
    
    # Test Google AI (Gemini)
    print("🤖 Testing Google AI Gemini-2.0-Flash")
    print("-" * 40)
    try:
        gemini_response = LLMService.generate_explanation(test_context, test_question, use_local=False)
        print("✅ Gemini Response:")
        print(gemini_response)
        print(f"\n📏 Response Length: {len(gemini_response)} characters")
        print()
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        print()
    
    # Test Local LLM (Gemma3)
    print("🏠 Testing Local Gemma3:1b")
    print("-" * 40)
    try:
        local_response = LLMService.generate_explanation(test_context, test_question, use_local=True)
        print("✅ Local LLM Response:")
        print(local_response)
        print(f"\n📏 Response Length: {len(local_response)} characters")
        print()
    except Exception as e:
        print(f"❌ Local LLM Error: {e}")
        print()
    
    print("🎯 Analysis Complete!")
    print("Both LLMs should provide brief, beginner-friendly explanations")

if __name__ == "__main__":
    test_brief_explanations()
