// API Service for StockVision Backend
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/analysis';

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
      
      // Ensure the response data has the expected structure
      const data = response.data;
      
      // If chart_data is missing but price_data exists, copy it
      if (!data.chart_data && data.price_data) {
        data.chart_data = data.price_data;
      }
      
      // Ensure ticker_info exists
      if (!data.ticker_info) {
        data.ticker_info = {
          ticker: data.ticker,
          sector: data.sector,
          current_price: data.summary?.current_price || 0,
          price_change_pct: data.summary?.price_change || 0,
          volatility: data.statistics?.volatility || 0,
          avg_volume: data.summary?.volume || 0,
          date_range: data.summary?.date_range || ''
        };
      }
      
      return data;
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
      
      // Ensure the response data has the expected structure
      const data = response.data;
      
      // If chart_data is missing but price_data exists, copy it
      if (!data.chart_data && data.price_data) {
        data.chart_data = data.price_data;
      }
      
      // If both performance_metrics and comparison_summary/performance_ranking exist, reorganize
      if (!data.comparison_summary && data.performance_metrics && data.performance_metrics.summary) {
        data.comparison_summary = data.performance_metrics.summary;
      }
      
      if (!data.performance_ranking && data.performance_metrics && data.performance_metrics.ranking) {
        data.performance_ranking = data.performance_metrics.ranking;
      }
      
      return data;
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

  // Prediction API methods
  async predictStock(ticker, days = 90, model = 'linear_regression') {
    try {
      const response = await axios.get(`http://localhost:8000/api/v1/predict/${ticker}`, {
        params: { days, model }
      });
      return response.data;
    } catch (error) {
      console.error('Error predicting stock:', error);
      throw error;
    }
  }

  async getRecommendation(ticker, days = 90) {
    try {
      const response = await axios.get(`http://localhost:8000/api/v1/predict/${ticker}/recommend`, {
        params: { days }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting recommendation:', error);
      throw error;
    }
  }

  async compareModels(ticker, days = 90) {
    try {
      const response = await axios.get(`http://localhost:8000/api/v1/predict/${ticker}/compare`, {
        params: { days }
      });
      return response.data;
    } catch (error) {
      console.error('Error comparing models:', error);
      throw error;
    }
  }

  async predictMultiple(tickers, days = 90, model = 'linear_regression') {
    try {
      const response = await axios.post('http://localhost:8000/api/v1/predict/multiple', {
        tickers,
        days,
        model
      });
      return response.data;
    } catch (error) {
      console.error('Error predicting multiple stocks:', error);
      throw error;
    }
  }

  async getModelInfo() {
    try {
      const response = await axios.get('http://localhost:8000/api/v1/predict/models/info');
      return response.data;
    } catch (error) {
      console.error('Error getting model info:', error);
      throw error;
    }
  }
}

export default new ApiService();
