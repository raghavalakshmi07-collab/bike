import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# ==========================================
# Load trained ANN model
# ==========================================

model = tf.keras.models.load_model("bike_rental_ann.h5")

# ==========================================
# Load scaler
# ==========================================

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# ==========================================
# Streamlit App
# ==========================================

st.title("🚲 Bike Rental Demand Prediction")

st.write(
    "Enter the weather and time details to predict bike rental demand."
)


# ==========================================
# Input Fields
# ==========================================

hr = st.number_input(
    "Hour of the Day",
    min_value=0,
    max_value=23,
    value=10
)

temp_c = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=41.0,
    value=25.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

windspeed_kmh = st.number_input(
    "Wind Speed (km/h)",
    min_value=0.0,
    max_value=67.0,
    value=10.0
)

season = st.selectbox(
    "Season",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Spring",
        2: "Summer",
        3: "Fall",
        4: "Winter"
    }[x]
)

weathersit = st.selectbox(
    "Weather Situation",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Clear / Partly Cloudy",
        2: "Mist / Cloudy",
        3: "Light Rain / Snow",
        4: "Heavy Rain / Snow"
    }[x]
)

workingday = st.selectbox(
    "Working Day",
    [0, 1],
    format_func=lambda x: {
        0: "No",
        1: "Yes"
    }[x]
)


# ==========================================
# Prediction
# ==========================================

if st.button("🚲 Predict Bike Rentals"):

    # Convert real-world values
    # into the normalized format used by the dataset

    temp = temp_c / 41.0
    hum = humidity / 100.0
    windspeed = windspeed_kmh / 67.0

    # Create input DataFrame

    input_data = pd.DataFrame({
        "hr": [hr],
        "temp": [temp],
        "hum": [hum],
        "windspeed": [windspeed],
        "season": [season],
        "weathersit": [weathersit],
        "workingday": [workingday]
    })

    # Arrange columns in training order

    input_data = input_data[
        [
            "hr",
            "temp",
            "hum",
            "windspeed",
            "season",
            "weathersit",
            "workingday"
        ]
    ]

    # Apply the same scaler used during training

    input_scaled = scaler.transform(input_data)

    # Make prediction

    prediction = model.predict(
        input_scaled,
        verbose=0
    )

    predicted_rentals = max(
        0,
        round(float(prediction[0][0]))
    )

    # Display result

    st.success(
        f"🚲 Predicted Bike Rentals: {predicted_rentals}"
    )