// API Service for StockVision Backend
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001/api/analysis';

class ApiService {
  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // Get available tickers
  async getTickers(sector = null, limit = null) {
    try {
      const params = {};
      if (sector) params.sector = sector;
      if (limit) params.limit = limit;
      
      const response = await this.api.get('/tickers', { params });
      return response.data;
    } catch (error) {
      console.error('Error fetching tickers:', error);
      throw error;
    }
  }

  // Analyze single ticker
  async analyzeTicker(ticker, days = 90) {
    try {
      const response = await this.api.post('/ticker', {
        ticker,
        days
      });
      return response.data;
    } catch (error) {
      console.error('Error analyzing ticker:', error);
      throw error;
    }
  }

  // Compare multiple tickers
  async compareTickers(tickers, days = 90) {
    try {
      const response = await this.api.post('/compare', {
        tickers,
        days
      });
      return response.data;
    } catch (error) {
      console.error('Error comparing tickers:', error);
      throw error;
    }
  }

  // Analyze sector
  async analyzeSector(sector, days = 90, topN = 10) {
    try {
      const response = await this.api.post('/sector', {
        sector,
        days,
        top_n: topN
      });
      return response.data;
    } catch (error) {
      console.error('Error analyzing sector:', error);
      throw error;
    }
  }

  // Get sector overview
  async getSectorOverview(days = 90) {
    try {
      const response = await this.api.get(`/sectors/${days}`);
      return response.data;
    } catch (error) {
      console.error('Error getting sector overview:', error);
      throw error;
    }
  }

  // Find volatile stocks
  async findVolatileStocks(sector = null, days = 90, topN = 10) {
    try {
      const response = await this.api.post('/volatility', {
        sector,
        days,
        top_n: topN
      });
      return response.data;
    } catch (error) {
      console.error('Error finding volatile stocks:', error);
      throw error;
    }
  }

  // Health check
  async healthCheck() {
    try {
      const response = await this.api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  // Get API stats
  async getStats() {
    try {
      const response = await this.api.get('/stats');
      return response.data;
    } catch (error) {
      console.error('Error fetching stats:', error);
      throw error;
    }
  }
}

export default new ApiService();
