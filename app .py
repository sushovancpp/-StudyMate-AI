import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

# Load dataset
file_path = "student_data.csv"  
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file '{file_path}' not found.")

df = pd.read_csv(file_path)

# Encode categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'Country', 'State', 'City', 'Parent Occupation', 'Earning Class',
                       'Level of Student', 'Level of Course', 'Course Name']

for col in categorical_columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Feature selection
X = df[['Age', 'Gender', 'Country', 'State', 'City', 'Parent Occupation', 'Earning Class',
        'Level of Student', 'Level of Course', 'Course Name', 'Study Time Per Day', 'IQ of Student']]
y = df['Assessment Score']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

df['Predicted Score'] = model.predict(X)

# Dynamic Promotion Criteria (mean-based threshold)
promotion_threshold = df['Assessment Score'].mean()
df['Promotion Status'] = df['Predicted Score'].apply(lambda x: 'Promoted' if x >= promotion_threshold else 'Not Promoted')

# Improved Study Material Recommendation
def recommend_material(level):
    if level < 1:
        return "Basic Study Materials"
    elif 1 <= level <= 2:
        return "Intermediate Study Materials"
    else:
        return "Advanced Study Materials"

df['Recommended Material'] = df['Level of Student'].apply(recommend_material)

# Save outputs in different formats
output_excel = "Final_predictions.xlsx"
output_csv = "Final_predictions.csv"
output_json = "Final_predictions.json"

df.to_excel(output_excel, index=False)
df.to_csv(output_csv, index=False)
df.to_json(output_json, orient='records', indent=4)

# Print results in a structured format
print("\n========== Student Assessment Results ==========")
print(df[['Name', 'Assessment Score', 'Predicted Score', 'Promotion Status', 'Recommended Material']].head().to_string(index=False))
print("\n===============================================")
print(f"\nResults saved as: {output_excel}, {output_csv}, {output_json}\n")
