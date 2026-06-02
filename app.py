# =========================================================
# AI-POWERED ENERGY CONSUMPTION FORECASTING SYSTEM
# Updated for Streamlit + TensorFlow/Keras 3
# =========================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(
    page_title="AI Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)

page_bg = """
<style>

/* Background */
[data-testid="stAppViewContainer"]{
    background-image:
    linear-gradient(
        rgba(0,0,0,0.75),
        rgba(0,0,0,0.75)
    ),
    url("https://images.unsplash.com/photo-1504384308090-c894fdcc538d");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color: rgba(0,0,0,0.7);
}

/* Force ALL text to white */
html,
body,
p,
span,
div,
label,
small,
strong,
h1,
h2,
h3,
h4,
h5,
h6,
li,
a,
[data-testid="stMarkdownContainer"],
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    color: white !important;
}

/* Streamlit titles */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: white !important;
}

/* Metrics */
[data-testid="metric-container"] {
    color: white !important;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px;
}

/* Buttons */
.stButton > button {
    color: white !important;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Selectbox text */
.stSelectbox label,
.stSlider label {
    color: white !important;
}

/* Dropdown options */
.stSelectbox div[data-baseweb="select"] > div {
    color: black !important;
    background-color: white !important;
}

div[role="option"] {
    color: black !important;
}

</style>
"""
st.markdown(
    """
    <h1 style='color:white !important; text-align:center;'>
    ⚡ AI-Powered Energy Consumption Forecasting
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='color:white !important; text-align:center;'>
    Industry-Level Deep Learning Forecasting Dashboard
    </h3>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    df = pd.read_csv("AEP_hourly.csv")
    df.columns = ["Datetime", "Energy"]
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")
    df.set_index("Datetime", inplace=True)
    return df

df = load_data()

df["hour"] = df.index.hour
df["day"] = df.index.day
df["month"] = df.index.month
df["day_of_week"] = df.index.dayofweek
df["year"] = df.index.year
df["Rolling_Mean_24"] = df["Energy"].rolling(24).mean()

st.sidebar.title("⚙️ Forecast Settings")
sequence_length = st.sidebar.slider("Sequence Length", 12, 72, 24)
forecast_hours = st.sidebar.slider("Forecast Hours", 1, 24, 6)

st.subheader("📂 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

st.subheader("📊 Energy KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Energy", f"{df['Energy'].mean():.2f}")
c2.metric("Maximum Energy", f"{df['Energy'].max():.2f}")
c3.metric("Minimum Energy", f"{df['Energy'].min():.2f}")
c4.metric("Peak Hour", f"{df['hour'].mode()[0]}:00")

st.subheader("📈 Energy Consumption Analysis")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Energy"], mode="lines", name="Energy"))
fig.add_trace(go.Scatter(x=df.index, y=df["Rolling_Mean_24"], mode="lines", name="24H Avg"))
fig.update_layout(template="plotly_dark", height=550)
st.plotly_chart(fig, width="stretch")

monthly = df.resample("ME").mean(numeric_only=True)
month_fig = go.Figure()
month_fig.add_trace(go.Bar(x=monthly.index, y=monthly["Energy"]))
month_fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(month_fig, width="stretch")

data = df[["Energy"]]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1],1)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0
    )
    return model, history

# =========================================================
# MODEL LOADING / TRAINING
# =========================================================

st.subheader("🧠 Deep Learning Model")

history = None
MODEL_PATH = "energy_forecasting_model.keras"

try:

    if os.path.exists(MODEL_PATH):

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        model.compile(
            optimizer="adam",
            loss="mean_squared_error"
        )

        st.success("✅ Pretrained Model Loaded Successfully!")

    else:
        raise FileNotFoundError("Model file not found")

except Exception as e:

    st.warning(f"⚠️ Model Load Failed: {e}")

    # Delete incompatible model
    if os.path.exists(MODEL_PATH):
        try:
            os.remove(MODEL_PATH)
            st.info("🗑️ Incompatible model removed.")
        except:
            pass

    st.info("🚀 Training New Model...")

    with st.spinner("Training model..."):

        model, history = train_model(
            X_train,
            y_train,
            X_test,
            y_test
        )

        model.save(MODEL_PATH)

    st.success("✅ Model Trained & Saved Successfully!")

if history:
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(y=history.history["loss"], name="Train Loss"))
    loss_fig.add_trace(go.Scatter(y=history.history["val_loss"], name="Validation Loss"))
    st.plotly_chart(loss_fig, width="stretch")

predictions = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

mae = mean_absolute_error(y_test_actual, predictions)
mse = mean_squared_error(y_test_actual, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, predictions)

st.subheader("📌 Model Performance")
a,b,c,d = st.columns(4)
a.metric("MAE", f"{mae:.2f}")
b.metric("MSE", f"{mse:.2f}")
c.metric("RMSE", f"{rmse:.2f}")
d.metric("R²", f"{r2:.4f}")

pred_fig = go.Figure()
pred_fig.add_trace(go.Scatter(y=y_test_actual.flatten(), name="Actual"))
pred_fig.add_trace(go.Scatter(y=predictions.flatten(), name="Predicted"))
st.plotly_chart(pred_fig, width="stretch")

st.subheader("⚡ Future Energy Forecast")

future_input = X_test[-1]
current_batch = future_input.reshape(1, sequence_length, 1)

future_predictions = []

for _ in range(forecast_hours):
    future = model.predict(current_batch, verbose=0)[0]
    future_predictions.append(future)

    future_reshaped = future.reshape(1,1,1)
    current_batch = np.concatenate(
        [current_batch[:,1:,:], future_reshaped],
        axis=1
    )

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
)

forecast_fig = go.Figure()
forecast_fig.add_trace(
    go.Scatter(y=future_predictions.flatten(), mode="lines+markers")
)
st.plotly_chart(forecast_fig, width="stretch")

st.metric(
    "⚡ Next Hour Energy Prediction",
    f"{future_predictions[0][0]:.2f}"
)

csv = df.to_csv().encode("utf-8")
st.download_button(
    "📥 Download Dataset",
    csv,
    "energy_data.csv",
    "text/csv"
)

st.markdown("---")
st.markdown("### 🚀 Developed with Streamlit + Deep Learning + AI Forecasting")
