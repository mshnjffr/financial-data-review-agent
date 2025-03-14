## Financial Analysis Agent
A Streamlit application for stock analysis using Cody API and yfinance.

## Overview
The Financial Analysis Agent is a powerful tool that allows users to analyze individual stocks or compare multiple stocks using data from yfinance and AI-powered analysis through the Cody API. The application provides comprehensive financial insights, including price data, analyst recommendations, and fundamental metrics.

## Features
* Single Ticker Analysis: Analyze individual stocks with customizable data options
* Ticker Comparison: Compare two stocks side-by-side to make informed investment decisions
* Customizable Analysis: Choose which data to include (price data, analyst recommendations, fundamentals)
* Model Selection: Select from available AI models for analysis
* Adjustable Parameters: Fine-tune the AI response with parameters like max tokens, temperature, and top_p


## Installation
1. Clone the repository
2. Install the required dependencies
3. Create a .env file in the project root with the following variables:
```
sg_token=your_sourcegraph_token
sg_chat_endpoint=your_chat_endpoint
sg_models_endpoint=your_models_endpoint
x_requested_with=your_x_requested_with_value
```

## Usage
1. Run the application:
```
streamlit run main.py
```

2. Access the application in your web browser (typically at http://localhost:8501)

3. For single ticker analysis:
* Enter a ticker symbol (e.g., AAPL)
* Select which data to include (price, recommendations, fundamentals)
Click "Analyze Ticker"

4. For ticker comparison:

* Enter two ticker symbols
* Click "Compare Tickers"

## Configuration
The application allows you to configure various parameters:

* Model Selection: Choose from available AI models in the sidebar
* Max Tokens: Adjust the maximum length of the generated analysis
* Temperature: Control the creativity of the AI responses (higher values = more creative)
* Top P: Adjust the diversity of the AI responses


## Technical Details
The application uses:

* Streamlit: For the web interface
* yfinance: To fetch stock data from Yahoo Finance
* Cody API: For AI-powered analysis of the financial data
* Python-dotenv: For environment variable management


## Data Sources
The application fetches the following data for analysis:

* Price Data: Current price, daily and monthly changes, price history
* Analyst Recommendations: Recent analyst ratings and price targets
* Fundamentals: Company info, income statements, balance sheets, cash flow statements, and key financial metrics


## Acknowledgements
* This application uses the Cody API from Sourcegraph
* Stock data is provided by Yahoo Finance through the yfinance package

## Project structure
```
financial_data_review
├── README.md
├── app.py
├── main.py
├── pyproject.toml
└── uv.lock```