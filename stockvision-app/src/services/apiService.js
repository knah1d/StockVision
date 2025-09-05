// API Service for StockVision Backend
import axios from "axios";

const API_BASE_URL = "http://localhost:8001/api";
const ANALYSIS_BASE_URL = "http://localhost:8001/api/analysis";
const PREDICTION_BASE_URL = "http://localhost:8001/api/v1/predict";

class ApiService {
    constructor() {
        this.api = axios.create({
            baseURL: ANALYSIS_BASE_URL,
            timeout: 10000,
            headers: {
                "Content-Type": "application/json",
            },
        });

        this.predictionApi = axios.create({
            baseURL: PREDICTION_BASE_URL,
            timeout: 15000, // Longer timeout for ML predictions
            headers: {
                "Content-Type": "application/json",
            },
        });
    }

    // Get available tickers
    async getTickers(sector = null, limit = null) {
        try {
            const params = {};
            if (sector) params.sector = sector;
            if (limit) params.limit = limit;

            const response = await this.api.get("/tickers", { params });
            return response.data;
        } catch (error) {
            console.error("Error fetching tickers:", error);
            throw error;
        }
    }

    // Analyze single ticker
    async analyzeTicker(ticker, days = 90) {
        try {
            const response = await this.api.post("/ticker", {
                ticker,
                days,
            });
            return response.data;
        } catch (error) {
            console.error("Error analyzing ticker:", error);
            throw error;
        }
    }

    // Compare multiple tickers
    async compareTickers(tickers, days = 90) {
        try {
            const response = await this.api.post("/compare", {
                tickers,
                days,
            });
            return response.data;
        } catch (error) {
            console.error("Error comparing tickers:", error);
            throw error;
        }
    }

    // Analyze sector
    async analyzeSector(sector, days = 90, topN = 10) {
        try {
            const response = await this.api.post("/sector", {
                sector,
                days,
                top_n: topN,
            });
            return response.data;
        } catch (error) {
            console.error("Error analyzing sector:", error);
            throw error;
        }
    }

    // Get sector overview
    async getSectorOverview(days = 90) {
        try {
            const response = await this.api.get(`/sectors/${days}`);
            return response.data;
        } catch (error) {
            console.error("Error getting sector overview:", error);
            throw error;
        }
    }

    // Find volatile stocks
    async findVolatileStocks(sector = null, days = 90, topN = 10) {
        try {
            const response = await this.api.post("/volatility", {
                sector,
                days,
                top_n: topN,
            });
            return response.data;
        } catch (error) {
            console.error("Error finding volatile stocks:", error);
            throw error;
        }
    }

    // Health check
    async healthCheck() {
        try {
            const response = await this.api.get("/health");
            return response.data;
        } catch (error) {
            console.error("Health check failed:", error);
            throw error;
        }
    }

    // Get API stats
    async getStats() {
        try {
            const response = await this.api.get("/stats");
            return response.data;
        } catch (error) {
            console.error("Error fetching stats:", error);
            throw error;
        }
    }

    // ========== PREDICTION API METHODS ==========

    // Predict stock price for a single ticker
    async predictStock(ticker, days = 90, model = "linear_regression") {
        try {
            const response = await this.predictionApi.get(`/${ticker}`, {
                params: { days, model },
            });
            return response.data;
        } catch (error) {
            console.error("Error predicting stock:", error);
            throw error;
        }
    }

    // Compare models for a ticker
    async compareModels(ticker, days = 90) {
        try {
            const response = await this.predictionApi.get(
                `/${ticker}/compare`,
                {
                    params: { days },
                }
            );
            return response.data;
        } catch (error) {
            console.error("Error comparing models:", error);
            throw error;
        }
    }

    // Predict multiple stocks
    async predictMultipleStocks(
        tickers,
        days = 90,
        model = "linear_regression"
    ) {
        try {
            const response = await this.predictionApi.post(
                "/multiple",
                tickers,
                {
                    params: { days, model },
                }
            );
            return response.data;
        } catch (error) {
            console.error("Error predicting multiple stocks:", error);
            throw error;
        }
    }

    // Get investment recommendations
    async getRecommendations(ticker, days = 90) {
        try {
            const response = await this.predictionApi.get(
                `/${ticker}/recommend`,
                {
                    params: { days },
                }
            );
            return response.data;
        } catch (error) {
            console.error("Error getting recommendations:", error);
            throw error;
        }
    }

    // Get model information
    async getModelInfo() {
        try {
            const response = await this.predictionApi.get("/models/info");
            return response.data;
        } catch (error) {
            console.error("Error getting model info:", error);
            throw error;
        }
    }

    // Get prediction service health
    async getPredictionHealth() {
        try {
            const response = await this.predictionApi.get("/health");
            return response.data;
        } catch (error) {
            console.error("Prediction health check failed:", error);
            throw error;
        }
    }
}

export default new ApiService();
