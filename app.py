from flask import Flask, jsonify, send_from_directory
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# === Serve index.html and static frontend files ===
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# === Load all Parquet files from 'monthly_data/' folder ===
def load_parquet_files():
    dataframes = []
    for file in os.listdir('monthly_data'):
        if file.endswith('.parquet'):
            df = pd.read_parquet(os.path.join('monthly_data', file))
            dataframes.append(df)
    if not dataframes:
        raise FileNotFoundError("No parquet files found in monthly_data/")
    return pd.concat(dataframes, ignore_index=True)

# === Calculate co-occurring tags for 'python' ===
def calculate_co_occurrence(df):
    grouped = df.groupby('Question ID')['Tag'].apply(list)
    python_questions = grouped[grouped.apply(lambda x: 'python' in x)]
    co_occurrence_counts = {}
    for tags in python_questions:
        for tag in tags:
            if tag != 'python':
                co_occurrence_counts[tag] = co_occurrence_counts.get(tag, 0) + 1
    return sorted(co_occurrence_counts.items(), key=lambda x: x[1], reverse=True)[:10]

@app.route('/data')
def get_data():
    try:
        df = load_parquet_files()
        df.columns = df.columns.str.strip()

        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        df.dropna(subset=['DateTime'], inplace=True)

        df['Month'] = df['DateTime'].dt.to_period('M').astype(str)
        df = df[df['Month'] >= '2021-10']

        monthly_data = df.groupby(['Month', 'Tag']).size().unstack(fill_value=0)
        formatted_months = [pd.to_datetime(month).strftime('%b-%Y') for month in monthly_data.index]

        python_tags = calculate_co_occurrence(df)

        # === Count ===
        count_data = {
            tag: {
                "months": formatted_months,
                "values": [int(v) for v in monthly_data[tag].tolist()]
            } for tag in monthly_data.columns
        }

        # === Percentage ===
        monthly_totals = monthly_data.sum(axis=1)
        percentage_data = {
            tag: {
                "months": formatted_months,
                "values": [
                    round((val / total) * 100, 2) if total != 0 else 0
                    for val, total in zip(monthly_data[tag], monthly_totals)
                ]
            } for tag in monthly_data.columns
        }

        # === Volatility ===
        volatile_data = {
            tag: {
                "months": formatted_months,
                "values": [int(v) for v in monthly_data[tag].diff().fillna(0).abs().tolist()]
            } for tag in monthly_data.columns
        }

        # === Growth ===
        growth_data = {
            tag: {
                "months": formatted_months[1:],
                "values": [
                    round(((curr / prev) - 1) * 100, 2) if prev != 0 else 0
                    for prev, curr in zip(monthly_data[tag][:-1], monthly_data[tag][1:])
                ]
            } for tag in monthly_data.columns
        }

        # === Pie Chart (Top tags of latest month) ===
        latest_month = formatted_months[-1]
        latest_counts = {tag: int(monthly_data[tag][-1]) for tag in monthly_data.columns}
        top_tags = sorted(latest_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        pie_data = {
            "labels": [tag for tag, _ in top_tags],
            "values": [value for _, value in top_tags]
        }

        return jsonify({
            "count": count_data,
            "percentage": percentage_data,
            "volatile": volatile_data,
            "growth": growth_data,
            "pie": pie_data,
            "python_tags": python_tags
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # for Railway
    app.run(host="0.0.0.0", port=port)
