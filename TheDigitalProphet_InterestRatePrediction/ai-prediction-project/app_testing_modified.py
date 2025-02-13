import os
import h2o
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from h2o.automl import H2OAutoML
from flask import Flask, render_template, request, redirect, url_for

# Start H2O server before connecting
h2o.init()

# Set random seed globally
np.random.seed(42)

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize H2O
h2o.init(nthreads=-1, max_mem_size="2G", strict_version_check=False)

# 🏠 Home route: File upload form
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            return redirect(url_for("train_model", filename=file.filename))
    return render_template("upload.html")

# 🚀 Train Model & Generate Graphs
@app.route("/train/<filename>")
def train_model(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    # Load dataset
    df = pd.read_csv(file_path)
    df.set_index("Category", inplace=True)
    df = df.T  # Transpose so that dates become the index
    df.index = pd.to_datetime(df.index, format="%d-%b-%y")

    # Convert percentage strings to numeric
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace("%", "").astype(float)

    # Debugging: Print first few rows to verify data integrity
    print("Dataset after preprocessing:")
    print(df.head())

    # Convert to H2OFrame
    data = h2o.H2OFrame(df)

    # Define target and features
    target = "Net Interest Gap"
    features = [col for col in data.columns if col != target]

    # Train-test split with fixed seed
    train, test = data.split_frame(ratios=[0.8], seed=42)

    # Train AutoML with fixed seed
    aml = H2OAutoML(max_models=15, seed=42, max_runtime_secs=180)
    aml.train(x=features, y=target, training_frame=train)

    # Get best model
    lb = aml.leaderboard.as_data_frame()
    best_model_id = str(lb.iloc[0]["model_id"])
    final_model = h2o.get_model(best_model_id)

    # Predictions
    predictions = final_model.predict(test)
    pred_df = predictions.as_data_frame()
    test_df = test.as_data_frame()[[target]]  # Actual values

    # Extract dates
    test_dates = df.index[-len(test_df):]
    actual_df = pd.DataFrame({"Date": test_dates, "Net Interest Gap": test_df[target].values})
    predicted_df = pd.DataFrame({"Date": test_dates, "Net Interest Gap": pred_df["predict"].values})

    # Debugging: Print actual vs predicted values
    print("Actual vs Predicted Values:")
    print(actual_df.head())
    print(predicted_df.head())

    # 🌞 Predict future values (using last date in dataset)
    prediction_days = 20  # Set duration dynamically
    last_date = df.index[-1]  # Get the last available date from the dataset
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)

    future_df = pd.DataFrame(index=future_dates, columns=features)
    
    for col in features:
        future_df[col] = df[col].iloc[-5:].mean()  # Use last 5 values
    
    # Apply fixed random noise
    np.random.seed(42)
    future_df += np.random.normal(loc=0, scale=0.02, size=future_df.shape)

    future_h2o = h2o.H2OFrame(future_df)
    future_predictions = final_model.predict(future_h2o).as_data_frame()
    future_pred_df = pd.DataFrame({"Date": future_dates, "Net Interest Gap": future_predictions["predict"].values})

    # 📊 Graphs
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=actual_df["Date"], y=actual_df["Net Interest Gap"], mode='lines+markers', name="Actual Net Interest Gap"))
    fig1.add_trace(go.Scatter(x=predicted_df["Date"], y=predicted_df["Net Interest Gap"], mode='lines+markers', name="Predicted Net Interest Gap"))
    fig1.add_trace(go.Scatter(x=future_pred_df["Date"], y=future_pred_df["Net Interest Gap"], mode='lines+markers', name="Future Predicted Net Interest Gap"))
    fig1.update_layout(title="Actual vs Predicted vs Future Predictions", xaxis_title="Date", yaxis_title="Net Interest Gap", xaxis=dict(dtick="10D"))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=future_pred_df["Date"], y=future_pred_df["Net Interest Gap"], mode='lines+markers', name="Future Predicted Net Interest Gap"))
    fig2.update_layout(title="Future Predictions", xaxis_title="Date", yaxis_title="Net Interest Gap", xaxis=dict(dtick="10D"))

    fig3 = go.Figure()
    try:
        if hasattr(final_model, "varimp"):
            feature_importance = final_model.varimp(use_pandas=True)
            if feature_importance is not None:
                fig3.add_trace(go.Bar(x=feature_importance["variable"], y=feature_importance["relative_importance"], name="Feature Importance"))
                fig3.update_layout(title="Feature Importance", xaxis_title="Feature", yaxis_title="Relative Importance")
            else:
                fig3.add_trace(go.Scatter(x=[0], y=[0], mode='text', text="Feature importance data is empty", showlegend=False))
        else:
            fig3.add_trace(go.Scatter(x=[0], y=[0], mode='text', text="Feature importance not available", showlegend=False))
    except AttributeError:
        fig3.add_trace(go.Scatter(x=[0], y=[0], mode='text', text="Feature importance not available", showlegend=False))

    return render_template("results.html",
                           fig1=fig1.to_html(full_html=False),
                           fig2=fig2.to_html(full_html=False),
                           fig3=fig3.to_html(full_html=False))

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
