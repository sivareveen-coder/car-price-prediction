import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# SAVE DIRECTORY
# -------------------------------
SAVE_DIR = r"C:\Users\User\OneDrive\Desktop\Usedcar_Price_Prediction"

# -------------------------------
# Load Data
# -------------------------------
csv_path = os.path.join(SAVE_DIR, "used_cars.csv")
df = pd.read_csv(csv_path)
print("✅ Dataset loaded:", df.shape)

# -------------------------------
# Clean Price
# -------------------------------
df['price'] = df['price'].str.replace('$', '', regex=False)
df['price'] = df['price'].str.replace(',', '', regex=False)
df['price'] = df['price'].astype(float)
df['price'] = (df['price'] * 83).astype(int)  # USD → INR

# -------------------------------
# Clean Mileage
# -------------------------------
df['milage'] = df['milage'].str.replace(',', '', regex=False)
df['milage'] = df['milage'].str.replace(' mi.', '', regex=False)
df['milage'] = df['milage'].astype(float)
df['milage'] = (df['milage'] * 1.60934).astype(int)  # miles → km
df.rename(columns={'milage': 'kilometers_driven'}, inplace=True)

# -------------------------------
# Rename model_year → year
# -------------------------------
df.rename(columns={'model_year': 'year'}, inplace=True)

# -------------------------------
# Fill Missing Values
# -------------------------------
df['fuel_type']   = df['fuel_type'].fillna(df['fuel_type'].mode()[0])
df['accident']    = df['accident'].fillna(df['accident'].mode()[0])
df['clean_title'] = df['clean_title'].fillna(df['clean_title'].mode()[0])

# -------------------------------
# Keep only relevant columns for the app
# -------------------------------
df = df[['year', 'kilometers_driven', 'fuel_type', 'accident', 'clean_title', 'price']]
print("✅ Columns kept:", df.columns.tolist())
print("✅ Sample data:")
print(df.head(3))

# -------------------------------
# One-Hot Encode
# -------------------------------
df = pd.get_dummies(df, columns=['fuel_type', 'accident', 'clean_title'], drop_first=False)
df = df.astype(int)
print("\n✅ After encoding, shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------------
# Remove duplicates
# -------------------------------
df.drop_duplicates(inplace=True)

# -------------------------------
# Train / Test Split
# -------------------------------
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n✅ Train: {X_train.shape}, Test: {X_test.shape}")

# -------------------------------
# Train Model (GridSearchCV)
# -------------------------------
print("\n⏳ Running GridSearchCV — this may take a few minutes...")
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("✅ Best Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_

# -------------------------------
# Evaluate
# -------------------------------
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ MAE: ₹{int(mae):,}")
print(f"✅ R2 Score: {r2:.4f}")

# -------------------------------
# Save model & features to project folder
# -------------------------------
model_save_path    = os.path.join(SAVE_DIR, "car_price_prediction_model.pkl")
features_save_path = os.path.join(SAVE_DIR, "model_features.pkl")

with open(model_save_path, "wb") as f:
    pickle.dump(best_model, f)

with open(features_save_path, "wb") as f:
    pickle.dump(list(X.columns), f)

print(f"\n✅ Model saved to:    {model_save_path}")
print(f"✅ Features saved to: {features_save_path}")
print("\n🎉 Done! You can now run your Streamlit app.")