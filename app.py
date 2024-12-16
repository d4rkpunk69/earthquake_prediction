from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import requests
from calendar import monthrange

# Load the trained LSTM model
model = load_model('model/best_lstm_model_final.h5', compile=False)

# Min and max values used in MinMaxScaler during preprocessing
latitude_min, latitude_max = 10.0, 20.0
longitude_min, longitude_max = 120.0, 130.0
depth_min, depth_max = 0.0, 50.0
magnitude_min, magnitude_max = 3.0, 8.0

# Threshold for significant earthquake
MAGNITUDE_THRESHOLD = 5.0
CONSECUTIVE_DAYS_THRESHOLD = 3  # Minimum consecutive days for a true earthquake

# Geocoding API configuration
GEOCODING_API_KEY = "a60803bd9c024c418324bcb0155f9b57"
GEOCODING_API_URL = "https://api.opencagedata.com/geocode/v1/json"

# Load sun and moon distances from the file
sun_moon_data = pd.read_csv('distances/SUN_MOON-2025.csv', parse_dates=['Date'])
sun_moon_data['Date'] = sun_moon_data['Date'].apply(lambda x: x.strftime('%Y-%m-%d') + " 00:00:00")
sun_moon_data['Date'] = pd.to_datetime(sun_moon_data['Date'])

app = Flask(__name__)

def geocode_place(place):
    """Convert a place name to latitude and longitude using a geocoding API."""
    params = {"q": place, "key": GEOCODING_API_KEY}
    response = requests.get(GEOCODING_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            lat = data['results'][0]['geometry']['lat']
            lon = data['results'][0]['geometry']['lng']
            return lat, lon
        else:
            raise ValueError(f"Place '{place}' not found.")
    else:
        raise ValueError("Error in geocoding API request.")

def minmax_scale(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def minmax_inverse_scale(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

def get_sun_moon_distances(date_time):
    """Retrieve sun and moon distances for the specified date and time."""
    filtered_data = sun_moon_data[sun_moon_data['Date'].dt.date == date_time.date()]
    if not filtered_data.empty:
        return filtered_data.iloc[0]['sun_distance'], filtered_data.iloc[0]['moon_distance']
    else:
        raise ValueError("Sun and moon distances not found for the specified date and time.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract input values from the request
        place = data['place']
        start_month = int(data['start_month'])
        end_month = int(data['end_month'])

        # Geocode the place to get latitude and longitude
        lat, lon = geocode_place(place)

        # Ensure the months are within range
        if not (1 <= start_month <= 12) or not (1 <= end_month <= 12):
            return jsonify({"error": "Invalid month range. Only months between 1 and 12 are allowed."}), 400

        # Prepare results for each month
        predictions = []

        # Loop through the selected months (from start to end month)
        for month in range(start_month, end_month + 1):
            # Get the number of days in the month
            _, num_days = monthrange(2025, month)

            # Loop through each day in the month
            for day in range(1, num_days + 1):
                date_time = datetime(2025, month, day, 0, 0)

                try:
                    # Get sun and moon distances
                    sun_distance, moon_distance = get_sun_moon_distances(date_time)

                    # Scale inputs
                    latitude_scaled = minmax_scale(lat, latitude_min, latitude_max)
                    longitude_scaled = minmax_scale(lon, longitude_min, longitude_max)
                    depth_scaled = minmax_scale(10, depth_min, depth_max)  # Example depth at 10 km
                    sun_distance_scaled = minmax_scale(sun_distance, sun_moon_data['sun_distance'].min(), sun_moon_data['sun_distance'].max())
                    moon_distance_scaled = minmax_scale(moon_distance, sun_moon_data['moon_distance'].min(), sun_moon_data['moon_distance'].max())
                    year_scaled = minmax_scale(date_time.year, 2016, 2025)
                    month_scaled = minmax_scale(month, 1, 12)
                    day_scaled = minmax_scale(day, 1, num_days)

                    # Prepare the input array (ensure the shape is correct for LSTM)
                    inputs = np.array([[latitude_scaled, longitude_scaled, depth_scaled, year_scaled, month_scaled, day_scaled, 0, sun_distance_scaled, moon_distance_scaled]])
                    inputs = np.expand_dims(inputs, axis=0)  # Add batch dimension

                    # Make prediction
                    normalized_prediction = model.predict(inputs)[0]
                    actual_magnitude = float(minmax_inverse_scale(normalized_prediction, magnitude_min, magnitude_max))

                    # Only consider significant predictions
                    if actual_magnitude >= MAGNITUDE_THRESHOLD:
                        predictions.append({
                            "date": date_time,
                            "predicted_magnitude": actual_magnitude
                        })

                except ValueError:
                    continue

        # Analyze predictions for consecutive significant earthquakes
        if predictions:
            predictions = sorted(predictions, key=lambda x: x['date'])

            consecutive_quakes = []
            current_streak = [predictions[0]]

            for i in range(1, len(predictions)):
                if (predictions[i]['date'] - predictions[i - 1]['date']).days <= 1:
                    current_streak.append(predictions[i])
                else:
                    if len(current_streak) >= CONSECUTIVE_DAYS_THRESHOLD:
                        consecutive_quakes.append(current_streak)
                    current_streak = [predictions[i]]

            if len(current_streak) >= CONSECUTIVE_DAYS_THRESHOLD:
                consecutive_quakes.append(current_streak)

            if consecutive_quakes:
                most_likely_quake = max(consecutive_quakes, key=lambda x: max(q['predicted_magnitude'] for q in x))
                start_date = most_likely_quake[0]['date']
                end_date = most_likely_quake[-1]['date']
                return jsonify({
                    "place": place,
                    "message": f"A significant earthquake is likely to occur between {start_date.strftime('%B %d')} and {end_date.strftime('%B %d')}.",
                    "details": most_likely_quake
                })

        return jsonify({"place": place, "message": "It is unlikely for an earthquake to occur this month."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
