from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
import pandas as pd  # For reading the SUN_MOON-2025 file
import requests

# Load the trained LSTM model
model = load_model('model/best_lstm_model_final.h5', compile=False)

# Min and max values used in MinMaxScaler during preprocessing
latitude_min, latitude_max = 10.0, 20.0
longitude_min, longitude_max = 120.0, 130.0
depth_min, depth_max = 0.0, 50.0
magnitude_min, magnitude_max = 3.0, 8.0

# Valid range for the Philippines
philippines_lat_min, philippines_lat_max = 4.0, 21.0
philippines_lon_min, philippines_lon_max = 116.0, 127.0

# Year range for scaling
year_min, year_max = 2016, 2024
month_min, month_max = 1, 12
day_min, day_max = 1, 31
hour_min, hour_max = 0, 23

# Geocoding API configuration
GEOCODING_API_KEY = "a60803bd9c024c418324bcb0155f9b57"
GEOCODING_API_URL = "https://api.opencagedata.com/geocode/v1/json"

# Load sun and moon distances from the file
sun_moon_data = pd.read_csv('distances/SUN_MOON-2025.csv', parse_dates=['Date'])

# Autofill missing times in the dataset with 00:00:00
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

def validate_inputs(lat, lon, depth, date_time):
    errors = []
    if not (philippines_lat_min <= lat <= philippines_lat_max):
        errors.append(f"Latitude {lat} is outside the Philippines' range.")
    if not (philippines_lon_min <= lon <= philippines_lon_max):
        errors.append(f"Longitude {lon} is outside the Philippines' range.")
    if not (depth_min <= depth <= depth_max):
        errors.append(f"Depth {depth} is outside the valid range ({depth_min}-{depth_max}).")
    if date_time <= datetime.now():
        errors.append("Date and time must be in the future.")
    return errors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract input values from the request
        place = data['place']
        depth = float(data['depth'])
        date_str = data['datetime']

        # Autofill time if missing and convert to datetime
        date_time = datetime.strptime(date_str, "%Y-%m-%dT%H:%M")

        # Geocode the place to get latitude and longitude
        lat, lon = geocode_place(place)

        # Get sun and moon distance data
        sun_distance, moon_distance = get_sun_moon_distances(date_time)

        # Validate inputs
        errors = validate_inputs(lat, lon, depth, date_time)
        if errors:
            return jsonify({"errors": errors}), 400

        # Scale inputs
        latitude_scaled = minmax_scale(lat, latitude_min, latitude_max)
        longitude_scaled = minmax_scale(lon, longitude_min, longitude_max)
        depth_scaled = minmax_scale(depth, depth_min, depth_max)
        sun_distance_scaled = minmax_scale(sun_distance, sun_moon_data['sun_distance'].min(), sun_moon_data['sun_distance'].max())
        moon_distance_scaled = minmax_scale(moon_distance, sun_moon_data['moon_distance'].min(), sun_moon_data['moon_distance'].max())
        year_scaled = minmax_scale(date_time.year, year_min, year_max)
        month_scaled = minmax_scale(date_time.month, month_min, month_max)
        day_scaled = minmax_scale(date_time.day, day_min, day_max)
        hour_scaled = minmax_scale(date_time.hour, hour_min, hour_max)

        # Prepare the input array (ensure the shape is correct for LSTM)
        inputs = np.array([[latitude_scaled, longitude_scaled, depth_scaled, year_scaled, month_scaled, day_scaled, hour_scaled, sun_distance_scaled, moon_distance_scaled]])
        inputs = np.expand_dims(inputs, axis=0)  # Ensure batch size dimension is added

        # Make prediction
        normalized_prediction = model.predict(inputs)[0]
        actual_magnitude = float(minmax_inverse_scale(normalized_prediction, magnitude_min, magnitude_max))

        return jsonify({
            "place": place,
            "latitude": lat,
            "longitude": lon,
            "depth": depth,
            "sun_distance": float(sun_distance),
            "moon_distance": float(moon_distance),
            "predicted_magnitude": round(actual_magnitude, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
