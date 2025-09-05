"""
Prediction Controller

FastAPI controller for stock price prediction endpoints.
"""

from fastapi import HTTPException
from typing import List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.prediction_service import PredictionService
from models.schemas import PredictionRequest, PredictionResponse, ModelComparisonResponse, RecommendationResponse
import logging

logger = logging.getLogger(__name__)

class PredictionController:
    """Controller for prediction-related endpoints."""
    
    def __init__(self):
        self.prediction_service = PredictionService()
    
    async def predict_stock(self, ticker: str, days: int = 90, model: str = 'linear_regression') -> dict:
        """
        Predict stock price for a single ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days of historical data
            model (str): ML model to use for prediction
            
        Returns:
            dict: Prediction results
        """
        try:
            if not ticker:
                raise HTTPException(status_code=400, detail="Ticker symbol is required")
            
            if days <= 0 or days > 1000:
                raise HTTPException(status_code=400, detail="Days must be between 1 and 1000")
            
            result = self.prediction_service.predict_ticker(ticker.upper(), days, model)
            
            if 'error' in result:
                raise HTTPException(status_code=404, detail=result['error'])
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in predict_stock: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during prediction")
    
    async def compare_models(self, ticker: str, days: int = 90) -> dict:
        """
        Compare all available models for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days of historical data
            
        Returns:
            dict: Model comparison results
        """
        try:
            if not ticker:
                raise HTTPException(status_code=400, detail="Ticker symbol is required")
            
            result = self.prediction_service.compare_models_for_ticker(ticker.upper(), days)
            
            if 'error' in result:
                raise HTTPException(status_code=404, detail=result['error'])
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in compare_models: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during model comparison")
    
    async def predict_multiple(self, tickers: List[str], days: int = 90, model: str = 'linear_regression') -> dict:
        """
        Predict stock prices for multiple tickers.
        
        Args:
            tickers (List[str]): List of stock ticker symbols
            days (int): Number of days of historical data
            model (str): ML model to use for prediction
            
        Returns:
            dict: Prediction results for all tickers
        """
        try:
            if not tickers:
                raise HTTPException(status_code=400, detail="At least one ticker symbol is required")
            
            if len(tickers) > 20:
                raise HTTPException(status_code=400, detail="Maximum 20 tickers allowed per request")
            
            # Convert to uppercase
            tickers_upper = [ticker.upper() for ticker in tickers]
            
            result = self.prediction_service.predict_multiple_tickers(tickers_upper, days, model)
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in predict_multiple: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during multiple predictions")
    
    async def get_recommendations(self, ticker: str, days: int = 90) -> dict:
        """
        Get investment recommendations for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days of historical data
            
        Returns:
            dict: Investment recommendations
        """
        try:
            if not ticker:
                raise HTTPException(status_code=400, detail="Ticker symbol is required")
            
            result = self.prediction_service.get_prediction_recommendations(ticker.upper(), days)
            
            if 'error' in result:
                raise HTTPException(status_code=404, detail=result['error'])
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in get_recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during recommendation generation")
    
    async def get_model_info(self) -> dict:
        """
        Get information about available models and features.
        
        Returns:
            dict: Model and feature information
        """
        try:
            return self.prediction_service.get_model_info()
        except Exception as e:
            logger.error(f"Error in get_model_info: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error while fetching model info")
    
    async def get_health_status(self) -> dict:
        """
        Get health status of the prediction service.
        
        Returns:
            dict: Health status information
        """
        try:
            return self.prediction_service.get_health_status()
        except Exception as e:
            logger.error(f"Error in get_health_status: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error while checking health status")

# Initialize controller instance
prediction_controller = PredictionController()
