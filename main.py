import streamlit as st
import os
import dotenv
import requests
import json
import yfinance as yf
from datetime import datetime

# Load environment variables
dotenv.load_dotenv()

# Constants
SG_TOKEN = os.getenv("sg_token")
SG_CHAT_ENDPOINT = os.getenv("sg_chat_endpoint")
SG_MODELS_ENDPOINT = os.getenv("sg_models_endpoint")
X_REQUESTED_WITH = os.getenv("x_requested_with")

# Function to fetch available models
def fetch_models():
    try:
        headers = {
            "Authorization": SG_TOKEN,
            "Content-Type": "application/json",
            "X-Requested-With": X_REQUESTED_WITH
        }
        
        print(f"REQUEST: GET {SG_MODELS_ENDPOINT}")
        print(f"REQUEST HEADERS: {json.dumps(headers, indent=2)}")
        
        response = requests.get(SG_MODELS_ENDPOINT, headers=headers)
        
        print(f"RESPONSE STATUS: {response.status_code}")
        print(f"RESPONSE BODY: {response.text}")
        
        if response.status_code == 200:
            models = response.json().get("data", [])
            print(f"Successfully fetched models from {SG_MODELS_ENDPOINT}")
            
            # Print all available models
            print("Available models:")
            for i, model in enumerate(models):
                print(f"  Model {i+1}: {json.dumps(model, indent=2)}")
            
            return models
        else:
            print(f"Error fetching models: {response.status_code}")
            print(f"Response body: {response.text}")
            return []
    except Exception as e:
        print(f"Exception while fetching models: {str(e)}")
        return []

# Function to get stock data using yfinance
def get_stock_data(ticker_symbol, include_price=True, include_recommendations=True, include_fundamentals=True):
    data = {}
    
    try:
        print(f"Fetching stock data for {ticker_symbol}")
        print(f"Options: include_price={include_price}, include_recommendations={include_recommendations}, include_fundamentals={include_fundamentals}")
        
        ticker = yf.Ticker(ticker_symbol)
        if include_price:
            # Get historical price data
            hist = ticker.history(period="1mo")
            if not hist.empty:
                data["current_price"] = hist.iloc[-1].Close
                data["price_change_1d"] = hist.Close.pct_change().iloc[-1] if len(hist) > 1 else None
                data["price_change_1m"] = (hist.iloc[-1].Close / hist.iloc[0].Close - 1) if len(hist) > 1 else None
                data["price_history"] = hist.Close.to_dict() if hasattr(hist.Close, 'to_dict') else dict(enumerate(hist.Close.values))
            
        if include_recommendations:
            # Get analyst recommendations
            try:
                recs = ticker.recommendations
                if recs is not None and not recs.empty and hasattr(recs, 'to_dict'):
                    data["recommendations"] = recs.tail(10).to_dict()
                else:
                    data["recommendations"] = {}
            except Exception as e:
                print(f"Error fetching recommendations: {str(e)}")
                data["recommendations"] = {}
            
            # Get analyst price targets
            try:
                targets = ticker.analyst_price_targets
                if targets is not None and hasattr(targets, 'to_dict'):
                    data["analyst_price_targets"] = targets.to_dict()
                elif isinstance(targets, dict):
                    data["analyst_price_targets"] = targets
                else:
                    data["analyst_price_targets"] = {}
            except Exception as e:
                print(f"Error fetching price targets: {str(e)}")
                data["analyst_price_targets"] = {}
            
        if include_fundamentals:
            # Get company info
            data["info"] = ticker.info
            
            # Get income statement (contains net income, which replaces earnings)
            try:
                income = ticker.income_stmt
                if income is not None and hasattr(income, 'to_dict'):
                    data["income_stmt"] = income.to_dict()
                elif isinstance(income, dict):
                    data["income_stmt"] = income
                else:
                    data["income_stmt"] = {}
                
                # Extract net income specifically (replacement for earnings)
                if "Net Income" in income.index:
                    data["net_income"] = income.loc["Net Income"].to_dict() if hasattr(income.loc["Net Income"], 'to_dict') else dict(enumerate(income.loc["Net Income"].values))
            except Exception as e:
                print(f"Error fetching income statement: {str(e)}")
                data["income_stmt"] = {}
                data["net_income"] = {}
            
            # Get balance sheet
            try:
                balance = ticker.balance_sheet
                if balance is not None and hasattr(balance, 'to_dict'):
                    data["balance_sheet"] = balance.to_dict()
                elif isinstance(balance, dict):
                    data["balance_sheet"] = balance
                else:
                    data["balance_sheet"] = {}
            except Exception as e:
                print(f"Error fetching balance sheet: {str(e)}")
                data["balance_sheet"] = {}
            
            # Get cash flow statement
            try:
                cashflow = ticker.cashflow
                if cashflow is not None and hasattr(cashflow, 'to_dict'):
                    data["cashflow"] = cashflow.to_dict()
                elif isinstance(cashflow, dict):
                    data["cashflow"] = cashflow
                else:
                    data["cashflow"] = {}
            except Exception as e:
                print(f"Error fetching cashflow: {str(e)}")
                data["cashflow"] = {}
            
            # Get key financial metrics
            try:
                data["key_metrics"] = {
                    "market_cap": ticker.info.get("marketCap"),
                    "pe_ratio": ticker.info.get("trailingPE"),
                    "forward_pe": ticker.info.get("forwardPE"),
                    "dividend_yield": ticker.info.get("dividendYield"),
                    "beta": ticker.info.get("beta"),
                    "52_week_high": ticker.info.get("fiftyTwoWeekHigh"),
                    "52_week_low": ticker.info.get("fiftyTwoWeekLow")
                }
            except Exception as e:
                print(f"Error fetching key metrics: {str(e)}")
                data["key_metrics"] = {}
            
        print(f"Successfully fetched data for {ticker_symbol}")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {str(e)}")
        return {"error": str(e)}

def generate_response(prompt, model, max_tokens=1000, temperature=0.7, top_p=0.9, max_iterations=3):
    try:
        headers = {
            "Authorization": SG_TOKEN,
            "Content-Type": "application/json",
            "X-Requested-With": X_REQUESTED_WITH,
            "api-version" : "6"
        }
        
        # Define tools for financial analysis
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_data",
                    "description": "Get financial data for a stock ticker",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker_symbol": {
                                "type": "string",
                                "description": "The stock ticker symbol (e.g., AAPL)"
                            },
                            "include_price": {
                                "type": "boolean",
                                "description": "Whether to include price data"
                            },
                            "include_recommendations": {
                                "type": "boolean",
                                "description": "Whether to include analyst recommendations"
                            },
                            "include_fundamentals": {
                                "type": "boolean",
                                "description": "Whether to include fundamental data"
                            }
                        },
                        "required": ["ticker_symbol"]
                    }
                }
            }
        ]
        
        # Create the messages array
        messages = [
            {
                "role": "assistant",
                "content": "You are a financial analysis assistant. Analyze stock data and provide insights based on the information provided."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Iterative approach to handle multiple rounds of tool calls
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"API call iteration {iteration}")
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "tools": tools
            }
            
            print(f"REQUEST: POST {SG_CHAT_ENDPOINT}")
            print(f"REQUEST HEADERS: {json.dumps(headers, indent=2)}")
            print(f"REQUEST PAYLOAD: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                SG_CHAT_ENDPOINT,
                headers=headers,
                json=payload
            )
            
            print(f"RESPONSE STATUS: {response.status_code}")
            print(f"RESPONSE BODY: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Received successful response from API")
                
                # Check if the response contains tool calls
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0]["message"]
                    
                    # If there are no tool calls, return the content
                    if "tool_calls" not in message or not message["tool_calls"]:
                        return message.get("content", "")
                    
                    # Process tool calls
                    messages.append({
                        "role": "assistant",
                        "content": message.get("content", ""),
                        "tool_calls": message["tool_calls"]
                    })
                    
                    # Process each tool call
                    for tool_call in message["tool_calls"]:
                        if tool_call["function"]["name"] == "get_stock_data":
                            # Parse the arguments
                            args = json.loads(tool_call["function"]["arguments"])
                            ticker_symbol = args.get("ticker_symbol")
                            include_price = args.get("include_price", True)
                            include_recommendations = args.get("include_recommendations", True)
                            include_fundamentals = args.get("include_fundamentals", True)
                            
                            # Execute the function
                            print(f"Executing tool call: get_stock_data for {ticker_symbol}")
                            stock_data = get_stock_data(
                                ticker_symbol,
                                include_price=include_price,
                                include_recommendations=include_recommendations,
                                include_fundamentals=include_fundamentals
                            )
                            
                            # Add the tool result to messages
                            messages.append({
                                "tool_call_id": tool_call["id"],
                                "role": "assistant",
                                "name": "get_stock_data",
                                "content": json.dumps(convert_timestamps_to_str(stock_data))
                            })
                else:
                    return result.get("completion", "")
            else:
                error_msg = f"Error from API: {response.status_code}"
                print(error_msg)
                print(f"Response body: {response.text}")
                return f"Error: {error_msg}\nResponse: {response.text}"
        
        # If we've reached the maximum number of iterations without a final response
        return "Analysis could not be completed after multiple attempts. Please try again."
    except Exception as e:
        error_msg = f"Exception during API call: {str(e)}"
        print(error_msg)
        return f"Error: {error_msg}"    

def convert_timestamps_to_str(obj):
    """Convert pandas Timestamp objects to ISO format strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_timestamps_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps_to_str(i) for i in obj]
    elif hasattr(obj, 'isoformat'):  # This will catch datetime and Timestamp objects
        return obj.isoformat()
    else:
        return obj
    
# Streamlit app
def main():
    st.set_page_config(page_title="Financial Analysis Agent", layout="wide")
    
    st.title("Financial Analysis Agent")
    st.subheader("Powered by Cody API and yfinance")
    
    # Sidebar for configuration
    st.sidebar.header("Model Configuration")
    
    # Fetch available models
    available_models = fetch_models()
    model_names = [model.get("id", "unknown") for model in available_models] if available_models else ["default"]    
    # Model selection
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_names[0] if model_names else "default"
        
    selected_model = st.sidebar.selectbox(
        "Select Model",
        model_names,
        index=model_names.index(st.session_state.selected_model) if st.session_state.selected_model in model_names else 0
    )
    
    # Save model selection
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.sidebar.success(f"Model set to {selected_model}")
    
    # Model parameters
    max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=4000, value=1000, step=100)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    top_p = st.sidebar.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
    
    # Main content area
    st.header("Stock Analysis")
    
    # Single ticker analysis
    st.subheader("Single Ticker Analysis")
    ticker_symbol = st.text_input("Enter Ticker Symbol (e.g., AAPL)", "")
    
    # Analysis options
    st.write("Analysis Options:")
    col1, col2, col3 = st.columns(3)
    with col1:
        include_price = st.checkbox("Include Price Data", value=True)
    with col2:
        include_recommendations = st.checkbox("Include Analyst Recommendations", value=True)
    with col3:
        include_fundamentals = st.checkbox("Include Fundamentals", value=True)
    
    if st.button("Analyze Ticker"):
        if ticker_symbol:
            with st.spinner(f"Analyzing {ticker_symbol}..."):
                # Get stock data
                stock_data = get_stock_data(
                    ticker_symbol, 
                    include_price=include_price,
                    include_recommendations=include_recommendations,
                    include_fundamentals=include_fundamentals
                )
                
                print(stock_data)
                # Create prompt for LLM
                prompt = f"""
                Analyze the following stock data for {ticker_symbol} and provide a comprehensive summary:
                
                {stock_data}
                
                Please include:
                1. A brief overview of the company
                2. Current price and recent performance
                3. Summary of analyst recommendations
                4. Key financial metrics and fundamentals
                5. Strengths and weaknesses based on the data
                6. Potential outlook
                
                Format your response in markdown with clear sections.
                """
                
                # Generate response
                response = generate_response(
                    prompt, 
                    selected_model, 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # Display response
                st.markdown(response)
        else:
            st.error("Please enter a ticker symbol")
    
    # Ticker comparison
    st.subheader("Compare Two Tickers")
    col1, col2 = st.columns(2)
    with col1:
        ticker1 = st.text_input("First Ticker Symbol", "")
    with col2:
        ticker2 = st.text_input("Second Ticker Symbol", "")
    
    if st.button("Compare Tickers"):
        if ticker1 and ticker2:
            with st.spinner(f"Comparing {ticker1} and {ticker2}..."):
                # Get stock data for both tickers
                stock_data1 = get_stock_data(
                    ticker1, 
                    include_price=True,
                    include_recommendations=True,
                    include_fundamentals=True
                )
                
                stock_data2 = get_stock_data(
                    ticker2, 
                    include_price=True,
                    include_recommendations=True,
                    include_fundamentals=True
                )
                
                # Create prompt for LLM
                prompt = f"""
                Compare the following two stocks and provide a detailed analysis:
                
                Stock 1: {ticker1}
                {stock_data1}
                
                Stock 2: {ticker2}
                {stock_data2}
                
                Please include only the following in your analysis:
                {("1. Current price and recent performance" if include_price else "")}
                {("2. Analyst recommendations" if include_recommendations else "")}
                {("3. Key financial metrics and fundamentals" if include_fundamentals else "")}
                4. Relative strengths and weaknesses
                5. Which stock appears to be a better investment based on the available data
                
                Format your response in markdown with clear sections and tables where appropriate.
                """                
                # Generate response
                response = generate_response(
                    prompt, 
                    selected_model, 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # Display response
                st.markdown(response)
        else:
            st.error("Please enter both ticker symbols")

if __name__ == "__main__":
    main()