from fastapi import HTTPException, UploadFile
import os
from dotenv import load_dotenv
import base64
import requests
import tempfile
import shutil
from pathlib import Path
import uuid
from typing import TYPE_CHECKING

# Load environment variables from .env file
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing_extensions import List, TypedDict, Union
from langchain_ollama import OllamaLLM

if TYPE_CHECKING:
    from models.llm_schemas import ContextItem


# Global variables for LLM components
llm = None
local_llm = None
embeddings = None
prompt = None

class LLMService:

    @staticmethod
    def setup_temp_directories():
        """Setup temporary directories for image storage"""
        base_dir = Path("temp_uploads")
        base_dir.mkdir(exist_ok=True)
        return base_dir

    @staticmethod
    def save_uploaded_file(file: UploadFile) -> str:
        """Save uploaded file and return the file path"""
        try:
            # Setup temp directory
            temp_dir = LLMService.setup_temp_directories()
            
            # Generate unique filename
            file_extension = Path(file.filename or "image.jpg").suffix
            unique_filename = f"{uuid.uuid4().hex}{file_extension}"
            file_path = temp_dir / unique_filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            return str(file_path)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error saving file: {str(e)}")

    @staticmethod
    def cleanup_old_files(max_age_hours: int = 24):
        """Clean up old uploaded files"""
        try:
            temp_dir = Path("temp_uploads")
            if not temp_dir.exists():
                return
                
            import time
            current_time = time.time()
            
            for file_path in temp_dir.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > (max_age_hours * 3600):  # Convert hours to seconds
                        file_path.unlink()
                        
        except Exception as e:
            print(f"Warning: Could not cleanup old files: {e}")

    @staticmethod
    def create_beginner_friendly_prompt(question: str, context_text: str = "", has_image: bool = False) -> str:
        """Create a beginner-friendly prompt for stock market analysis"""
        base_prompt = """You are a friendly financial educator speaking to a stock market beginner. 
        
        Your goal is to:
        1. Explain financial concepts in simple, easy-to-understand language
        2. Avoid complex jargon or use simple explanations when technical terms are necessary
        3. Provide practical insights that a beginner can understand and learn from
        4. Use analogies and examples to make concepts clear
        5. Be encouraging and educational
        
        """
        
        if has_image:
            base_prompt += f"""
        Context: {context_text}
        
        You are looking at a stock market chart/graph. Please analyze what you see and explain it in beginner-friendly terms.
        
        Question: {question}
        
        Please explain:
        - What the chart shows (price movements, trends, patterns)
        - What these patterns might mean for investors
        - Key lessons a beginner should learn from this chart
        - Any important concepts demonstrated in the visual data
        
        Remember to keep your explanation simple and educational for someone new to investing."""
        else:
            base_prompt += f"""
        Context: {context_text}
        
        Question: {question}
        
        Please provide a clear, beginner-friendly explanation that helps someone new to the stock market understand the concepts involved."""
        
        return base_prompt

    @classmethod
    def initialize_ai_components(cls):
        global llm, prompt
        
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
    def initialize_local_llm(cls):
        """Initialize local Ollama LLM"""
        global local_llm
        try:
            local_llm = OllamaLLM(model="gemma3:1b")
            # Test if the model is available
            test_response = local_llm.invoke("Hello")
            print(f"Local LLM initialized: {test_response[:50]}...")
        except Exception as e:
            print(f"Warning: Local LLM not available: {e}")
            local_llm = None

    @classmethod
    def generate_local_explanation(cls, context: Union[str, List["ContextItem"]], question: str) -> str:
        """Generate explanation using local Ollama model"""
        global local_llm
        
        if local_llm is None:
            cls.initialize_local_llm()
            if local_llm is None:
                raise HTTPException(
                    status_code=503, 
                    detail="Local LLM not available. Please install Ollama and pull gemma2:1b model."
                )
        
        # Process context to create a good prompt
        if isinstance(context, str):
            # Simple text context
            prompt_text = f"""You are a financial analyst AI assistant. Analyze the following data and answer the question.

Context: {context}

Question: {question}

Provide a detailed financial analysis based on the context provided."""
        else:
            # Multimodal context - extract text parts and describe image parts
            text_parts = []
            image_count = 0
            
            for item in context:
                if item.type == "text":
                    text_parts.append(item.content)
                elif item.type == "image_path":
                    image_count += 1
                    if os.path.exists(item.content):
                        text_parts.append(f"[Image {image_count}: Stock chart/financial document at {Path(item.content).name} - Local model will analyze based on text context]")
                    else:
                        text_parts.append(f"[Image {image_count}: File not found at {item.content}]")
                elif item.type == "image":
                    image_count += 1
                    text_parts.append(f"[Image {image_count}: Financial chart/document provided but cannot be directly processed by local model]")
            
            combined_context = "\n".join(text_parts)
            
            prompt_text = f"""You are a financial analyst AI assistant. Analyze the following financial data and answer the question.

Context: {combined_context}
Note: {image_count} financial images/charts were provided but cannot be directly analyzed by this local model. Please provide analysis based on the text context and acknowledge the presence of visual data.

Question: {question}

Provide a detailed financial analysis based on the available context."""
        
        try:
            response = local_llm.invoke(prompt_text)
            return response
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error with local LLM: {str(e)}"
            )

    @classmethod
    def alternative_local_llama_via_ollama(cls, context: Union[str, List["ContextItem"]], question: str) -> str:
        """Alternative method using local Ollama - simplified version of generate_local_explanation"""
        return cls.generate_local_explanation(context, question)


    @classmethod
    def generate_explanation(cls, context: Union[str, List["ContextItem"]], question: str, use_local: bool = False) -> str:
        global llm, prompt
        
        # If explicitly requested to use local LLM or if it's non-text content without sufficient text
        if use_local or cls._should_use_local_llm(context):
            return cls.generate_local_explanation(context, question)
        
        try:
            if llm is None or prompt is None:
                cls.initialize_ai_components()
            
            # Handle different context types
            if isinstance(context, str):
                # Simple text context - use the existing prompt template
                messages = prompt.invoke({"question": question, "context": context})
                response = llm.invoke(messages)
            else:
                # Multimodal context (text + images)
                content_parts = []
                text_context = ""
                
                # Process each context item
                for item in context:
                    if item.type == "text":
                        text_context += item.content + "\n"
                    elif item.type == "image_path":
                        # Handle local image file
                        try:
                            if os.path.exists(item.content):
                                # Convert local file to base64 for Gemini
                                with open(item.content, "rb") as img_file:
                                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                                    
                                # Determine image type from file extension
                                file_ext = Path(item.content).suffix.lower()
                                if file_ext in ['.png']:
                                    mime_type = "image/png"
                                elif file_ext in ['.jpg', '.jpeg']:
                                    mime_type = "image/jpeg"
                                elif file_ext in ['.gif']:
                                    mime_type = "image/gif"
                                else:
                                    mime_type = "image/jpeg"  # default
                                
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                                })
                            else:
                                text_context += f"[Image file not found: {item.content}]\n"
                        except Exception as e:
                            text_context += f"[Error processing image file: {str(e)}]\n"
                    elif item.type == "image":
                        # Handle base64 or URL images (legacy support)
                        try:
                            if item.content.startswith("data:image"):
                                # Base64 encoded image
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": item.content}
                                })
                            elif item.content.startswith("http"):
                                # Image URL - validate it exists
                                try:
                                    response_check = requests.head(item.content, timeout=5)
                                    if response_check.status_code == 200:
                                        content_parts.append({
                                            "type": "image_url", 
                                            "image_url": {"url": item.content}
                                        })
                                    else:
                                        text_context += f"[Image URL not accessible: {item.content}]\n"
                                except requests.RequestException:
                                    text_context += f"[Image URL not accessible: {item.content}]\n"
                            else:
                                # Assume it's base64 without data URL prefix
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{item.content}"}
                                })
                        except Exception as e:
                            text_context += f"[Image processing error: {str(e)}]\n"
                
                # Add text content
                text_content = f"""You are a financial analyst AI assistant. Analyze the provided images and text context to answer questions about stocks, market trends, and financial data.

Text Context: {text_context}

Question: {question}

Provide accurate, helpful analysis based on the data. If you cannot answer based on the context, say so clearly."""
                
                content_parts.insert(0, {"type": "text", "text": text_content})
                
                # Create a message with multimodal content
                message = HumanMessage(content=content_parts)
                response = llm.invoke([message])
            
            return response.content
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            # If Google AI fails, try local LLM as fallback
            print(f"Google AI failed: {e}, trying local LLM...")
            try:
                return cls.generate_local_explanation(context, question)
            except:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating explanation with both Google AI and local LLM: {str(e)}"
                )
    
    @classmethod
    def analyze_uploaded_image(cls, file_path: str, question: str, context_text: str = "This is a stock market chart/graph for analysis.", use_local: bool = False) -> str:
        """Analyze an uploaded image with beginner-friendly explanations"""
        try:
            # Clean up old files first
            cls.cleanup_old_files()
            
            # Create context with the image file using proper ContextItem objects
            from models.llm_schemas import ContextItem
            
            context = [
                ContextItem(type="text", content=context_text),
                ContextItem(type="image_path", content=file_path)
            ]
            
            # Create beginner-friendly prompt
            beginner_question = cls.create_beginner_friendly_prompt(question, context_text, has_image=True)
            
            # Use the main generate_explanation method
            result = cls.generate_explanation(context, beginner_question, use_local=use_local)
            
            return result
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing uploaded image: {str(e)}"
            )

    @classmethod
    def _should_use_local_llm(cls, context: Union[str, List["ContextItem"]]) -> bool:
        """Determine if we should use local LLM based on context type"""
        if isinstance(context, str):
            return False  # Text-only, use Google AI
        
        # Check if context has mostly images with minimal text
        text_length = 0
        image_count = 0
        
        for item in context:
            if item.type == "text":
                text_length += len(item.content.strip())
            elif item.type in ["image", "image_path"]:
                image_count += 1
        
        # Use local LLM if there are images and very little text (less than 50 characters)
        return image_count > 0 and text_length < 50

    