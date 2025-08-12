from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from models.llm_schemas import ExplainRequest, ExplainResponse
from services.llm_service import LLMService

router = APIRouter(prefix="/api/ai/explain", tags=["AI Explain"])



@router.post("/", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
):
    """Generate an explanation for a given context and question"""
    try:
        # Call the LLM service to get the explanation
        explanation = LLMService.generate_explanation(request.context, request.question)
        return ExplainResponse(explanation=explanation)
    except HTTPException:
        # Re-raise HTTP exceptions from the service
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

