// LLM Service for AI-powered explanations
import axios from 'axios';

const LLM_API_BASE_URL = 'http://localhost:8001/api/ai/explain';

class LLMService {
  constructor() {
    this.api = axios.create({
      baseURL: LLM_API_BASE_URL,
      timeout: 30000, // Longer timeout for AI responses
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // Text-based explanation (uses Google AI)
  async explainText(context, question) {
    try {
      const response = await this.api.post('/', {
        context,
        question
      });
      return response.data.explanation;
    } catch (error) {
      console.error('Error getting text explanation:', error);
      throw error;
    }
  }

  // Local LLM explanation (for fallback or when needed)
  async explainLocal(context, question) {
    try {
      const response = await this.api.post('/local', {
        context,
        question
      });
      return response.data.explanation;
    } catch (error) {
      console.error('Error getting local explanation:', error);
      throw error;
    }
  }

  // Image/chart explanation with file upload
  async explainChart(chartImageBlob, question, contextText = "This is a stock market chart/graph for beginner analysis.", useLocal = false) {
    try {
      const formData = new FormData();
      formData.append('file', chartImageBlob, 'chart.png');
      formData.append('question', question);
      formData.append('context_text', contextText);
      formData.append('use_local', useLocal.toString());

      const response = await this.api.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data.explanation;
    } catch (error) {
      console.error('Error getting chart explanation:', error);
      throw error;
    }
  }

  // Helper method to capture chart as image blob
  async captureChartAsImage(chartElement) {
    return new Promise((resolve, reject) => {
      try {
        // If it's a Chart.js canvas
        if (chartElement && chartElement.toBlob) {
          chartElement.toBlob((blob) => {
            resolve(blob);
          }, 'image/png');
        }
        // If it's a regular canvas
        else if (chartElement && chartElement.canvas && chartElement.canvas.toBlob) {
          chartElement.canvas.toBlob((blob) => {
            resolve(blob);
          }, 'image/png');
        }
        // Fallback: try to get canvas from Chart.js instance
        else if (chartElement && chartElement.chart && chartElement.chart.canvas) {
          chartElement.chart.canvas.toBlob((blob) => {
            resolve(blob);
          }, 'image/png');
        }
        else {
          reject(new Error('Unable to capture chart as image'));
        }
      } catch (error) {
        reject(error);
      }
    });
  }

  // Smart explanation method - decides whether to use text or image API
  async explainChartData(chartData, chartRef, question, useLocal = false) {
    try {
      // Try to capture chart as image first
      if (chartRef && chartRef.current) {
        try {
          const chartBlob = await this.captureChartAsImage(chartRef.current);
          return await this.explainChart(chartBlob, question, 
            "This is a stock market chart showing price movements, trends, and technical indicators for educational purposes.", 
            useLocal);
        } catch (imageError) {
          console.warn('Failed to capture chart image, falling back to text explanation:', imageError);
        }
      }

      // Fallback to text-based explanation
      const contextText = this.formatChartDataForText(chartData);
      if (useLocal) {
        return await this.explainLocal(contextText, question);
      } else {
        return await this.explainText(contextText, question);
      }
    } catch (error) {
      console.error('Error in smart explanation:', error);
      throw error;
    }
  }

  // Helper to format chart data as text context
  formatChartDataForText(chartData) {
    if (!chartData) return "Stock market data for analysis.";
    
    let context = "Stock market data analysis:\n";
    
    if (chartData.datasets) {
      chartData.datasets.forEach(dataset => {
        if (dataset.label && dataset.data) {
          const dataLength = dataset.data.length;
          const lastValue = dataset.data[dataLength - 1];
          const firstValue = dataset.data[0];
          context += `${dataset.label}: Latest value ${lastValue}, Starting value ${firstValue}, Data points: ${dataLength}\n`;
        }
      });
    }
    
    if (chartData.labels) {
      const dateRange = `From ${chartData.labels[0]} to ${chartData.labels[chartData.labels.length - 1]}`;
      context += `Time period: ${dateRange}\n`;
    }
    
    return context;
  }
}

export default new LLMService();
