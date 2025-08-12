# StockVision 📈

**Visualize and Predict Stock Trends using Machine Learning**

StockVision is a FastAPI-based application that provides stock market data visualization and trend prediction capabilities using machine learning algorithms. The project combines historical stock data analysis with predictive modeling to help users understand market trends and make informed decisions.

## 🚀 Features

-   **RESTful API**: Built with FastAPI for high-performance data serving
-   **Stock Data Analysis**: Process and analyze historical stock market data (2008-2022)
-   **Machine Learning Predictions**: Implement ML models for stock trend forecasting
-   **Data Visualization**: Generate insightful charts and visualizations
-   **Interactive Notebooks**: Jupyter notebooks for data preprocessing and model training

## 📁 Project Structure

```
StockVision/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           └── data.py  # Data API endpoints
│   ├── ml/
│   │   └── predict.py       # Machine learning prediction models
│   └── services/
│       └── loader.py        # Data loading services
├── data/
│   ├── raw/
│   │   └── archive/         # Historical stock data (2008-2022)
│   │       ├── prices_YYYY.json
│   │       └── securities.json
│   └── processed/
│       ├── all_data.csv     # Processed stock data
│       └── securities.csv   # Securities information
├── models/                  # Trained ML models storage
├── notebooks/
│   ├── Preprocessing.ipynb  # Data preprocessing notebook
│   ├── train.ipynb         # Model training notebook
│   └── test.ipynb          # Model testing notebook
├── requirements.txt        # Python dependencies
├── run.sh                 # Application startup script
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

-   Python 3.8 or higher
-   pip package manager

### Setup

1. **Clone the repository**

    ```bash
    git clone https://github.com/knah1d/StockVision.git
    cd StockVision
    ```

2. **Create a virtual environment** (recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Running the API Server

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

-   **API Base URL**: http://localhost:8000
-   **Interactive API Docs**: http://localhost:8000/docs
-   **ReDoc Documentation**: http://localhost:8000/redoc

### API Endpoints

-   `GET /api/v1/data/` - Health check endpoint
-   More endpoints will be available as development progresses

### Working with Notebooks

Navigate to the `notebooks/` directory and start Jupyter:

```bash
jupyter notebook notebooks/
```

Available notebooks:

-   **Preprocessing.ipynb**: Data cleaning and preprocessing
-   **train.ipynb**: Model training and evaluation
-   **test.ipynb**: Model testing and validation

## 📊 Data

The project includes historical stock market data spanning from 2008 to 2022:

-   **Price Data**: Daily stock prices for multiple securities
-   **Securities Information**: Company metadata and classification
-   **Processed Data**: Cleaned and formatted datasets ready for analysis

## 🤖 Machine Learning

StockVision implements various ML algorithms for stock prediction:

-   Time series forecasting
-   Trend analysis
-   Pattern recognition
-   Risk assessment

### Technologies Used

-   **FastAPI**: Modern, fast web framework for building APIs
-   **Pandas**: Data manipulation and analysis
-   **NumPy**: Numerical computing
-   **Scikit-learn**: Machine learning library
-   **Matplotlib**: Data visualization
-   **Uvicorn**: ASGI server implementation

## 🔧 Development

### Project Status

This project is currently in active development. Core features being implemented:

-   [ ] Data loading and preprocessing pipeline
-   [ ] ML model implementation
-   [ ] API endpoint development
-   [ ] Visualization components
-   [ ] Model training and evaluation

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



