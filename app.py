import io
import os
import re
from flask import Flask, render_template, request, send_file
from fpdf import FPDF
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

ETHERSCAN_API_KEY = os.getenv("API_KEY")

def is_valid_eth_address(address):
    return bool(re.match(r"^0x[a-fA-F0-9]{40}$", address))


def plot_time_series(df, anomaly_col='anomaly', save_path='static/time_series.png'):
    plt.figure(figsize=(10, 5))

    normal = df[df[anomaly_col] != -1]
    anomaly = df[df[anomaly_col] == -1]

    # Plot normal transactions
    plt.plot(normal['timeStamp'], normal['value'], 'o', label='Normal', color='blue', alpha=0.6)
    
    # Plot anomalies
    plt.plot(anomaly['timeStamp'], anomaly['value'], 'o', label='Anomaly', color='red', alpha=0.9)

    plt.xlabel('Timestamp')
    plt.ylabel('Transaction Value (ETH)')
    plt.title('Transaction Value Over Time')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def fetch_transactions(address):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "desc",
        "apikey": ETHERSCAN_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] == "1":
        df = pd.DataFrame(data["result"])
        df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s')
        df['value'] = df['value'].astype(float) / 1e18
        df['gasPrice'] = df['gasPrice'].astype(float) / 1e9
        return df
    return pd.DataFrame()

def plot_value_vs_gas(df, anomaly_col='anomaly', save_path='static/scatter_plot.png'):
    """
    Generates a scatter plot of transaction value vs. gas price with anomalies highlighted.
    """
    plt.figure(figsize=(8, 6))

    normal = df[df[anomaly_col] != -1]
    anomaly = df[df[anomaly_col] == -1]

    plt.scatter(normal['value'], normal['gasPrice'], label='Normal', color='green', alpha=0.5)
    plt.scatter(anomaly['value'], anomaly['gasPrice'], label='Anomaly', color='orange', alpha=0.9)

    plt.xlabel('Transaction Value (ETH)')
    plt.ylabel('Gas Price (Gwei)')
    plt.title('Value vs. Gas Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_differences(df):
    df['tx_interval'] = df['timeStamp'].diff().dt.total_seconds().fillna(0)
    df['anomaly'] = df['anomaly'].astype(int)

    normal = df[df['anomaly'] != -1]
    anomaly = df[df['anomaly'] == -1]

    features = ['value', 'gasPrice', 'tx_interval']
    normal_means = normal[features].mean()
    anomaly_means = anomaly[features].mean()

    x = range(len(features))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x, normal_means, width, label='Normal')
    plt.bar([i + width for i in x], anomaly_means, width, label='Anomalies')
    plt.xticks([i + width / 2 for i in x], features)
    plt.ylabel('Feature Value (mean)')
    plt.title('Feature Differences: Normal vs Anomalous Transactions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/feature_comparison.png')
    plt.close()

def detect_anomalies(df):
    if df.empty or len(df) < 5:
        return pd.DataFrame()

    tx_features = df[['value', 'gasPrice']].copy()
    tx_features['tx_interval'] = df['timeStamp'].diff().dt.total_seconds().fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(tx_features)

    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(scaled_features)
    df['anomaly_score'] = model.decision_function(scaled_features)  # Higher = more normal

    anomalies = df[df['anomaly'] == -1]
    return anomalies[['hash', 'from', 'to', 'value', 'gasPrice', 'timeStamp', 'anomaly_score']]

@app.route('/', methods=['GET', 'POST'])
def index():
    anomalies = None
    global anomalies_df
    if request.method == 'POST':
        address = request.form['address'].strip()
        if not is_valid_eth_address(address):
            return render_template('index.html', error="❌ Invalid Ethereum address format.")
        df = fetch_transactions(address)
        anomalies = detect_anomalies(df)
        anomalies_df = anomalies.copy()
        if not anomalies.empty:
            plot_feature_differences(df)
            plot_time_series(df)
            plot_value_vs_gas(df)  
    return render_template('index.html', anomalies=anomalies)

@app.route('/download_csv')
def download_csv():
    if 'anomalies_df' in globals():
        output = io.StringIO()
        anomalies_df.to_csv(output, index=False)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode()),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='anomalies_report.csv')
    return "No data available to download."


@app.route('/download_pdf')
def download_pdf():
    if 'anomalies_df' in globals():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="EthGuard - Ethereum Anomaly Report", ln=1, align='C')
        pdf.ln(10)

        for index, row in anomalies_df.head(20).iterrows():
            line = f"{row['hash'][:10]} | {row['from'][:6]}... -> {row['to'][:6]} | {row['value']} ETH | Score: {round(row['anomaly_score'], 4)}"
            pdf.cell(200, 10, txt=line, ln=True)

        pdf_output = "static/anomaly_report.pdf"
        pdf.output(pdf_output)
        return send_file(pdf_output, as_attachment=True)
    return "No data available for PDF report."


if __name__ == "__main__":
    app.run(debug=True)
