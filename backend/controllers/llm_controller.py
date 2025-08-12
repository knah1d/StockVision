from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from typing import List, Optional
from models.llm_schemas import ExplainRequest, ExplainResponse, FileUploadRequest
from services.llm_service import LLMService

router = APIRouter(prefix="/api/ai/explain", tags=["AI Explain"])



@router.post("/", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
):
    """Generate an explanation for a given context and question using Google AI (with local fallback)"""
    try:
        # Call the LLM service to get the explanation
        explanation = LLMService.generate_explanation(request.context, request.question)
        return ExplainResponse(explanation=explanation)
    except HTTPException:
        # Re-raise HTTP exceptions from the service
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/local", response_model=ExplainResponse)
def explain_local(
    request: ExplainRequest,
):
    """Generate an explanation using local Ollama model (gemma2:1b)"""
    try:
        # Force use of local LLM
        explanation = LLMService.generate_explanation(request.context, request.question, use_local=True)
        return ExplainResponse(explanation=explanation)
    except HTTPException:
        # Re-raise HTTP exceptions from the service
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error with local LLM: {str(e)}")


@router.post("/upload", response_model=ExplainResponse)
def explain_with_upload(
    file: UploadFile = File(..., description="Stock chart/graph image to analyze"),
    question: str = Form(..., description="Your question about the chart"),
    context_text: str = Form(default="This is a stock market chart/graph for beginner analysis.", description="Additional context about the image"),
    use_local: bool = Form(default=False, description="Force use of local LLM")
):
    """Upload a stock chart image and get beginner-friendly analysis"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save the uploaded file
        file_path = LLMService.save_uploaded_file(file)
        
        # Analyze the image with beginner-friendly explanations
        explanation = LLMService.analyze_uploaded_image(
            file_path=file_path,
            question=question,
            context_text=context_text,
            use_local=use_local
        )
        
        return ExplainResponse(explanation=explanation)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@router.post("/upload/local", response_model=ExplainResponse)
def explain_with_upload_local(
    file: UploadFile = File(..., description="Stock chart/graph image to analyze"),
    question: str = Form(..., description="Your question about the chart"),
    context_text: str = Form(default="This is a stock market chart/graph for beginner analysis.", description="Additional context about the image")
):
    """Upload a stock chart image and get analysis using local LLM only"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save the uploaded file
        file_path = LLMService.save_uploaded_file(file)
        
        # Analyze using local LLM
        explanation = LLMService.analyze_uploaded_image(
            file_path=file_path,
            question=question,
            context_text=context_text,
            use_local=True
        )
        
        return ExplainResponse(explanation=explanation)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload with local LLM: {str(e)}")

