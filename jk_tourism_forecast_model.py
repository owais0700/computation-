import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load your existing dataset
df = pd.read_csv("jk_tourism_data_2020_2024.csv")

# Label Encoding
le_month = LabelEncoder()
le_place = LabelEncoder()
le_weather = LabelEncoder()
le_festival = LabelEncoder()

df['Month'] = le_month.fit_transform(df['Month'])
df['Place'] = le_place.fit_transform(df['Place'])
df['Weather'] = le_weather.fit_transform(df['Weather'])
df['Festival'] = df['Festival'].fillna('None')
df['Festival'] = le_festival.fit_transform(df['Festival'])

# Features and Target
X = df[['Year', 'Month', 'Place', 'Weather', 'Festival', 'Holiday_Flag']]
y = df['Tourist_Count']

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Create Future Data (2025–2028) ---
future_years = range(2025, 2029)
months = list(range(12))  # 0-11
places = df['Place'].unique()
weathers = df['Weather'].unique()
festivals = df['Festival'].unique()

future_rows = []
for year in future_years:
    for month in months:
        for place in places:
            weather = random.choice(weathers)
            festival = random.choice(festivals)
            holiday = 1 if festival != le_festival.transform(['None'])[0] else 0
            future_rows.append([year, month, place, weather, festival, holiday])

future_df = pd.DataFrame(future_rows, columns=['Year', 'Month', 'Place', 'Weather', 'Festival', 'Holiday_Flag'])

# Predict
future_df['Predicted_Tourist_Count'] = model.predict(future_df)

# Decode for readability
future_df['Month'] = le_month.inverse_transform(future_df['Month'])
future_df['Place'] = le_place.inverse_transform(future_df['Place'])
future_df['Weather'] = le_weather.inverse_transform(future_df['Weather'])
future_df['Festival'] = le_festival.inverse_transform(future_df['Festival'])

# Save to CSV
future_df.to_csv("jk_tourism_predictions_2025_2028.csv", index=False)
print("Saved predictions to jk_tourism_predictions_2025_2028.csv")
