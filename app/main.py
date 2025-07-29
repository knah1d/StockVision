from fastapi import FastAPI
from app.api.v1.endpoints import data

app = FastAPI(
    title="StockVision",
    description="Visualize and Predict Stock Trends using ML",
    version="1.0"
)

app.include_router(data.router, prefix="/api/v1/data", tags=["Data"])
