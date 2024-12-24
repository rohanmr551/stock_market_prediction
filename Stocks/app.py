import os
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from flask import Flask, jsonify, request, send_file
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

try:
    model = load_model('Stocks/Stock Predictions Model.keras')  
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

GRAPH_DIR = '/tmp/generated_graphs'
os.makedirs(GRAPH_DIR, exist_ok=True)

@app.route('/predict', methods=['GET'])
def predict_stock():
    try:
        if model is None:
            return jsonify({'error': 'Model is not loaded properly. Please check the model file path.'}), 500

        stock = request.args.get('stock')
        if not stock:
            return jsonify({'error': 'Stock symbol is required'}), 400

        start = '2012-01-01'
        end = '2024-10-01'
        stock_data = yf.download(stock, start=start, end=end)

        if stock_data.empty:
            return jsonify({'error': 'No data found for the given stock symbol'}), 400

        stock_data.reset_index(inplace=True)  # Reset index to access 'Date' as a column
        ma_50_days = stock_data['Close'].rolling(50).mean()
        ma_100_days = stock_data['Close'].rolling(100).mean()
        ma_200_days = stock_data['Close'].rolling(200).mean()

        graph1_path = os.path.join(GRAPH_DIR, 'price_vs_ma50.png')
        graph2_path = os.path.join(GRAPH_DIR, 'price_vs_ma50_ma100.png')
        graph3_path = os.path.join(GRAPH_DIR, 'price_vs_ma100_ma200.png')
        graph4_path = os.path.join(GRAPH_DIR, 'actual_vs_predicted.png')
        graph5_path = os.path.join(GRAPH_DIR, 'future_predictions.png')

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

        # Prepare test data
        x_test = []
        for i in range(100, len(data_scaled)):
            x_test.append(data_scaled[i-100:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Predict stock prices
        predicted_stock_price = model.predict(x_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        # Generate historical vs predicted graph
        plt.figure(figsize=(10, 6), dpi=200)
        plt.plot(stock_data['Date'][100:], stock_data['Close'].values[100:], 'g', label='Actual Price')
        plt.plot(stock_data['Date'][100:], predicted_stock_price, 'r', label='Predicted Price')
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(graph4_path)
        plt.close()

        # Future price prediction (next 30 days)
        last_100_days = stock_data['Close'].values[-100:].reshape(-1, 1)
        last_100_days_scaled = scaler.transform(last_100_days)

        n = 30  # Number of future days
        future_prices = []
        current_input = last_100_days_scaled

        for _ in range(n):
            current_input = current_input[-100:] 
            current_input_reshaped = np.reshape(current_input, (1, 100, 1))
            predicted_price_scaled = model.predict(current_input_reshaped)
            predicted_price = scaler.inverse_transform(predicted_price_scaled)
            future_prices.append(predicted_price[0][0])
            current_input = np.append(current_input, predicted_price_scaled, axis=0)

        future_dates = pd.date_range(start=stock_data['Date'].iloc[-1], periods=n + 1, freq='B')[1:]

        plt.figure(figsize=(10, 6), dpi=200)
        plt.plot(stock_data['Date'], stock_data['Close'], 'g', label='Historical Price')
        plt.plot(future_dates, future_prices, 'r', label='Future Predictions')
        plt.title('Future Price Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(graph5_path)
        plt.close()

        def save_graph(data, ma1, ma2=None, title="", filename=""):
            plt.figure(figsize=(10, 6), dpi=200)
            plt.plot(stock_data['Date'], data, 'g', label='Close Price')
            if ma1 is not None:
                plt.plot(stock_data['Date'], ma1, 'r', label='MA 50 Days')
            if ma2 is not None:
                plt.plot(stock_data['Date'], ma2, 'b', label='MA 100 Days')
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        save_graph(stock_data['Close'], ma_50_days, None, 'Price vs MA50', graph1_path)
        save_graph(stock_data['Close'], ma_50_days, ma_100_days, 'Price vs MA50 vs MA100', graph2_path)
        save_graph(stock_data['Close'], ma_100_days, ma_200_days, 'Price vs MA100 vs MA200', graph3_path)

        return jsonify({
            'graph1': request.host_url + 'get_graph/price_vs_ma50',
            'graph2': request.host_url + 'get_graph/price_vs_ma50_ma100',
            'graph3': request.host_url + 'get_graph/price_vs_ma100_ma200',
            'graph4': request.host_url + 'get_graph/actual_vs_predicted',
            'graph5': request.host_url + 'get_graph/future_predictions'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_graph/<filename>', methods=['GET'])
def get_graph(filename):
    try:
        graph_path = os.path.join(GRAPH_DIR, f"{filename}.png")
        if os.path.exists(graph_path):
            return send_file(graph_path, mimetype='image/png')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
