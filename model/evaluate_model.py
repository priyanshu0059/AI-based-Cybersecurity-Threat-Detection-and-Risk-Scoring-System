import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import json

# Load data
df = pd.read_csv('./data/processed/processed_dataset.csv')
X = df.drop('Label', axis=1)
y = df['Label']

# Split (same as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Load model
model = joblib.load('./model/model.pkl')

# Predict
y_pred = model.predict(X_test)

# Evaluation
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Save report
with open('./model/report.json', 'w') as f:
    json.dump(report, f, indent=4)

print("📊 Classification Report:")
print(json.dumps(report, indent=2))

print("\n🧩 Confusion Matrix:")
print(cm)
