import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
df = pd.read_csv('backend/state_data.csv')

# Encode categorical variables
le = LabelEncoder()
df['State Name'] = le.fit_transform(df['State Name'])

# Split into features and target
X = df.drop('Total Water Demand', axis=1)  # Drop target column
y = df['Total Water Demand']

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Save the model and preprocessing objects
joblib.dump(model, 'backend/water_stress_model.pkl')
joblib.dump(scaler, 'backend/scaler.pkl')
joblib.dump(le, 'backend/label_encoder.pkl')

print("Model and preprocessing objects saved successfully!")