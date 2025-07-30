# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset (replace with real data later)
data = {
    'electricity_kwh': [200, 350, 150, 400, 250],
    'vehicle_km': [50, 100, 30, 150, 70],
    'carbon_footprint': [120, 250, 90, 310, 180]
}

df = pd.DataFrame(data)

# Features and target
X = df[['electricity_kwh', 'vehicle_km']]
y = df['carbon_footprint']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'carbon_model.pkl')

print("✅ Model trained and saved as carbon_model.pkl")
