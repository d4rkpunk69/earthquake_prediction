from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
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

# Valid range for the Philippines
philippines_lat_min, philippines_lat_max = 4.0, 21.0
philippines_lon_min, philippines_lon_max = 116.0, 127.0

# Year range for scaling
year_min, year_max = 2016, 2024
month_min, month_max = 1, 12

month_names = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

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

def decimal_to_dms(decimal_value):
    """Convert a decimal degree value to degrees, minutes, and seconds (DMS)."""
    is_negative = decimal_value < 0
    decimal_value = abs(decimal_value)
    degrees = int(decimal_value)
    minutes = int((decimal_value - degrees) * 60)
    seconds = round((decimal_value - degrees - minutes / 60) * 3600, 2)
    if is_negative:
        degrees = -degrees
    return degrees, minutes, seconds


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
            _, num_days = monthrange(2025, month)

            # Loop through each day in the month
            for day in range(1, num_days + 1):
                for depth in range(1, 32):  # Loop from 1 km to 31 km (inclusive)
                    date_time = datetime(2025, month, day, 0, 0)

                    # Get sun and moon distances
                    sun_distance, moon_distance = get_sun_moon_distances(date_time)

                    # Scale inputs
                    latitude_scaled = minmax_scale(lat, latitude_min, latitude_max)
                    longitude_scaled = minmax_scale(lon, longitude_min, longitude_max)
                    depth_scaled = minmax_scale(depth, 1.0, 31.0)
                    sun_distance_scaled = minmax_scale(sun_distance, sun_moon_data['sun_distance'].min(), sun_moon_data['sun_distance'].max())
                    moon_distance_scaled = minmax_scale(moon_distance, sun_moon_data['moon_distance'].min(), sun_moon_data['moon_distance'].max())
                    year_scaled = minmax_scale(date_time.year, year_min, year_max)
                    month_scaled = minmax_scale(date_time.month, month_min, month_max)
                    day_scaled = minmax_scale(day, 1, num_days)

                    # Prepare input array
                    inputs = np.array([[latitude_scaled, longitude_scaled, depth_scaled, year_scaled, month_scaled, day_scaled, 0, sun_distance_scaled, moon_distance_scaled]])
                    inputs = np.expand_dims(inputs, axis=0)

                    # Make prediction
                    normalized_prediction = model.predict(inputs)[0]
                    actual_magnitude = float(minmax_inverse_scale(normalized_prediction, magnitude_min, magnitude_max))

                    predictions.append({
                        "month": month,
                        "week": (day - 1) // 7 + 1,  # Determine the week of the month
                        "depth": depth,
                        "predicted_magnitude": round(actual_magnitude, 2),
                        "sun_distance": float(sun_distance),
                        "moon_distance": float(moon_distance)
                    })

        # Group predictions by week
        weekly_predictions = {}
        for prediction in predictions:
            week_key = (prediction["month"], prediction["week"])
            if week_key not in weekly_predictions:
                weekly_predictions[week_key] = []
            weekly_predictions[week_key].append(prediction["predicted_magnitude"])

        # Calculate average magnitude and standard deviation per week
        weekly_summary = []
        for week_key, magnitudes in weekly_predictions.items():
            avg_magnitude = np.mean(magnitudes)
            std_dev = np.std(magnitudes)
            weekly_summary.append({
                "month": week_key[0],
                "week": week_key[1],
                "avg_magnitude": round(avg_magnitude, 2),
                "std_dev": round(std_dev, 2)
            })

        lat_dms = decimal_to_dms(lat)
        lon_dms = decimal_to_dms(lon)

        # Determine high or low likelihood
        high_likelihood_weeks = [week for week in weekly_summary if week["avg_magnitude"] > 4.0 and week["std_dev"] < 0.5]

        if high_likelihood_weeks:
            best_week = max(high_likelihood_weeks, key=lambda x: x["avg_magnitude"])
            message = (f"High likelihood of an earthquake in {place} during the {best_week['week']} week of "
                       f"{month_names[best_week['month'] - 1]} with an average magnitude of {best_week['avg_magnitude']}.")
        else:
            message = f"No significant likelihood of an earthquake in {place} during the specified months."

       
        # Return the results
        return jsonify({
            "place": place,
            "latitude": lat,
            "latitude_dms": f"{abs(lat_dms[0])}°{lat_dms[1]}'{lat_dms[2]}\" {'S' if lat_dms[0] < 0 else 'N'}",
            "longitude": lon,
            "longitude_dms": f"{abs(lon_dms[0])}°{lon_dms[1]}'{lon_dms[2]}\" {'W' if lon_dms[0] < 0 else 'E'}",
            "summary": weekly_summary,  # Example placeholder
            "message": message  # Example placeholder
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
