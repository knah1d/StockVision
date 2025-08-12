from fastapi import HTTPException
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import List, TypedDict

# Global variables for LLM components
llm = None
embeddings = None
prompt = None

class LLMService:

    @classmethod
    def initialize_ai_components(cls):
        global llm, embeddings, prompt
        
        # Set up Google API key
        if not os.environ.get("GOOGLE_API_KEY"):
            # For production, you should set this as an environment variable
            # For now, we'll use a placeholder - you need to set this
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set in environment variables")
        
        try:
            # Initialize LLM and embeddings
            llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
            print(llm.invoke("hello"))
            
            # Create a prompt template for stock analysis
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial analyst AI assistant. Use the provided context to answer questions about stocks, market trends, and financial data. 
                
                Context: {context}
                
                Provide accurate, helpful analysis based on the data. If you cannot answer based on the context, say so clearly."""),
                ("human", "{question}")
            ])
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize AI components: {str(e)}. Please check your GOOGLE_API_KEY is valid."
            )


    @classmethod
    def generate_explanation(cls, context: str, question: str) -> str:
        global llm, prompt
        
        try:
            if llm is None or prompt is None:
                cls.initialize_ai_components()
                
            messages = prompt.invoke({"question": question, "context": context})
            response = llm.invoke(messages)
            return response.content
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating explanation: {str(e)}"
            )

    