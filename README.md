# StockingStuffer
The Tri-AI Driven, LSTM NN Based, Stock Market Assistant 

## Important Note
All placeholder values within the script where chosen at random by AI. This was done purposefuly so as to randomize the baseline to prevent targeted abuse of the script. 

## What is StockingStuffer?
This script assists investors with their decision-making process, using machine learning models to predict stock prices, and guiding investors in buying, selling or holding stocks. It focuses specifically on the AMCR stock but can be customized for other stocks.

## Dependencies

You'll need the following Python libraries:

- `yfinance`: Fetches historical market data from Yahoo Finance.
- `sklearn`: Offers data preprocessing capabilities and the Linear Regression model.
- `tensorflow`: Provides the LSTM model for prediction.
- `statsmodels`: Provides the ARIMA model for prediction.
- `numpy` and `pandas`: Used for data manipulation.

## How It Works

1. **Fetch Historical Data**: The script fetches historical data for the given ticker from Yahoo Finance.

2. **Diversification**: The script also fetches data for other tickers, adding diversification to your portfolio.

3. **Data Preprocessing**: The fetched data is scaled using the MinMaxScaler to bring all values between 0 and 1, suitable for our LSTM model.

4. **Training LSTM model**: A LSTM model is trained on the historical data to learn patterns in the stock's closing price. The architecture of this model includes two LSTM layers and one Dense layer.

5. **Predicting with LSTM model**: The script uses the trained LSTM model to predict the next day's closing price for the stock.

6. **Linear Regression Model**: A simple Linear Regression model is also trained on the historical data and used to predict the next day's closing price for the stock.

7. **Predictions for Other Tickers**: The script also predicts the next day's closing price for other tickers in the portfolio using the LSTM model.

8. **Buy or Sell Recommendation**: Based on the predictions and a user-defined risk tolerance, the script provides a simple recommendation of whether to buy or sell each stock in the portfolio.

## Limitations

1. **Risk Tolerance**: Risk tolerance is a user-defined parameter that depends on a number of personal factors. The script currently uses a placeholder value.

2. **Economic Factors and Company Fundamentals**: The models used in this script don't consider broader economic factors or company fundamentals, which are important considerations in an investment strategy.

3. **Investing Involves Risk**: Investing in the stock market always involves risk. While this script provides data-driven guidance, it does not guarantee profit or protect against loss.

4. **Lack of Complexity**: This script provides a basic strategy. A comprehensive investment strategy may need to take into account more factors and use more complex models.

## Instructions for Use

Follow these steps to get the script up and running:

## Install Dependencies

Ensure Python 3 is installed and set up on your system. After confirming that Python is installed, you will need to install several Python libraries that the script uses. To install these libraries, open your terminal and type:

```bash
pip install yfinance sklearn tensorflow statsmodels numpy pandas
```

## Download the Script

Download the StockingStuffer Python script to a directory on your computer. 

## Customize the Script

Open the script in a text editor. There are a few values in the script you can customize:

- `ticker`: This is the primary stock ticker that the script will provide predictions for. Change it to the ticker symbol of the stock you're interested in.
- `other_tickers`: This is a list of other stocks that the script considers for diversification. Change these to the ticker symbols of the stocks you're interested in.
- `risk_tolerance`: This is a user-defined value that depends on your personal risk tolerance. Change this to a value that matches your risk tolerance.

## Run the Script

Open a terminal, navigate to the directory where you saved the script, and type:

```bash
python StockingStuffer.py
```

## Interpret the Output

The script will output the predicted closing prices for the specified date for the stocks you entered. It will also provide a simple buy or sell recommendation for each stock, based on the predicted prices and your defined risk tolerance. Keep in mind that these predictions and recommendations are not guarantees, and investing in the stock market always involves risk.

## Schedule Regular Runs

For the script to be most useful, you'll want to run it regularly. How often you run it will depend on your investment strategy. Some investors might want to run it daily, while others might run it weekly or monthly. You can use tools like cron (on Unix-based systems) or Task Scheduler (on Windows) to run the script automatically at regular intervals. 

Remember, always do your own research and consider seeking advice from a financial advisor before making investment decisions.

## Conclusion

This script is a helpful tool for guiding investment decisions, but it's not a substitute for professional investment advice. Always do your own research or consult with a financial advisor before making investment decisions.

## Developed By: 
https://abtzpro.github.io
