from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/")
def read_data():
    return JSONResponse(content={"message": "StockVision API is working!"})
