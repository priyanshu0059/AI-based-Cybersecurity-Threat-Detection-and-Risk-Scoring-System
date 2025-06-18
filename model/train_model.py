# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import joblib
# import os
# from datetime import datetime
# import json
# import warnings

# # 📌 Ignore warnings for cleaner output
# warnings.filterwarnings('ignore')

# # Step 0: Fix dataset header if missing
# columns = [
#     'stime', 'proto', 'srcip', 'sport', 'dstip', 'dsport', 'state', 'dur',
#     'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload',
#     'dload', 'spkts', 'dpkts', 'swins', 'dwins', 'stcpb', 'dtcpb', 'smeansz',
#     'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'tcprtt',
#     'synack', 'ackdat', 'is_sm_ips_ports', 'label'
# ]
# DATA_PATH = './data/processed/processed_dataset.csv'
# df = pd.read_csv(DATA_PATH, header=None, names=columns)
# df.to_csv(DATA_PATH, index=False)
# print("✅ Dataset fixed and saved with headers.")

# # Constants
# MODEL_DIR = './model'
# MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
# METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')
# FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, 'feature_importance.png')

# # Step 1: Reload cleaned dataset
# df = pd.read_csv(DATA_PATH)
# print(f"\n📁 Loaded dataset with shape: {df.shape}")

# # Step 2: Normalize column names
# df.columns = df.columns.str.strip()
# print(f"🧾 Columns in dataset: {list(df.columns)}")

# # Step 3: Find actual label column
# label_col = None
# for col in df.columns:
#     if col.strip().lower() == 'label':
#         label_col = col
#         break

# if not label_col:
#     raise ValueError("❌ 'Label' column not found in dataset.")
# print(f"✅ Found label column: '{label_col}'")

# # Step 4: Keep only top 50 most common labels
# top_labels = df[label_col].value_counts().nlargest(50).index
# print(f"✅ Keeping top {len(top_labels)} most frequent labels for stratified split.")
# df = df[df[label_col].isin(top_labels)]

# # Step 5: Prepare features and target
# X = df.drop(label_col, axis=1)
# y = df[label_col]

# # Encode categorical columns
# categorical_cols = X.select_dtypes(include=['object']).columns
# if len(categorical_cols) > 0:
#     print(f"🔤 Encoding categorical columns: {list(categorical_cols)}")
#     X = pd.get_dummies(X, columns=categorical_cols)

# # Step 6: Train/test split (with stratify)
# if len(y.unique()) > 1:
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
# else:
#     raise ValueError("❌ Cannot stratify. Only one unique class found in target.")

# # Step 7: Train model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Step 8: Evaluate model
# y_pred = model.predict(X_test)
# print("\n📊 Classification Report:")
# print(classification_report(y_test, y_pred))
# print("\n🧩 Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# # Step 9: Cross-validation
# cv_scores = cross_val_score(model, X, y, cv=5)
# print(f"\n📈 Mean CV Accuracy: {np.mean(cv_scores):.4f}")

# # Step 10: Save model
# os.makedirs(MODEL_DIR, exist_ok=True)
# joblib.dump(model, MODEL_PATH)
# print(f"\n✅ Model saved to '{MODEL_PATH}'")

# # Step 11: Save metadata
# metadata = {
#     "model_name": "RandomForestClassifier",
#     "model_version": "1.0.0",
#     "trained_on": datetime.utcnow().isoformat(),
#     "features": list(X.columns),
#     "target_label": label_col,
#     "cv_accuracy": float(np.mean(cv_scores)),
#     "test_report": classification_report(y_test, y_pred, output_dict=True),
#     "class_names": list(np.unique(y)),
#     "preprocessing": {
#         "scaling": False,
#         "encoding": False
#     }
# }
# with open(METADATA_PATH, 'w') as f:
#     json.dump(metadata, f, indent=4)
# print(f"🗂️ Metadata saved to '{METADATA_PATH}'")

# # Step 12: Feature Importance Plot
# plt.figure(figsize=(10, 6))
# importances = model.feature_importances_
# indices = np.argsort(importances)[::-1]
# plt.title("Feature Importance")
# plt.bar(range(X.shape[1]), importances[indices], align="center")
# plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
# plt.tight_layout()
# plt.savefig(FEATURE_IMPORTANCE_PATH)
# print(f"📊 Feature importance plot saved to '{FEATURE_IMPORTANCE_PATH}'")

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

# === Constants ===
DATA_PATH = './data/processed/processed_dataset.csv'
MODEL_DIR = './model'
MODEL_PATH = os.path.join(MODEL_DIR, 'full_pipeline.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, 'feature_importance.png')

# === Step 0: Fix headers if missing ===
columns = [
    'stime', 'proto', 'srcip', 'sport', 'dstip', 'dsport', 'state', 'dur',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload',
    'dload', 'spkts', 'dpkts', 'swins', 'dwins', 'stcpb', 'dtcpb', 'smeansz',
    'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'tcprtt',
    'synack', 'ackdat', 'is_sm_ips_ports', 'label'
]
df = pd.read_csv(DATA_PATH, header=None, names=columns)
df.to_csv(DATA_PATH, index=False)
print("✅ Dataset fixed and saved with headers.")

# === Step 1: Load dataset ===
df = pd.read_csv(DATA_PATH)
print(f"\n📁 Loaded dataset with shape: {df.shape}")
df.columns = df.columns.str.strip()

# === Step 2: Identify label column ===
label_col = 'label'
if label_col not in df.columns:
    raise ValueError("❌ 'label' column not found.")

# === Step 3: Filter top labels ===
top_labels = df[label_col].value_counts().nlargest(50).index
df = df[df[label_col].isin(top_labels)]
print(f"✅ Kept top {len(top_labels)} labels.")

# === Step 4: Split features/labels ===
X = df.drop(label_col, axis=1)
y = df[label_col]

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()

# === Step 5: Preprocessing and pipeline ===
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
], remainder='passthrough')  # keep numeric columns as-is

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# === Step 6: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 7: Fit model ===
pipeline.fit(X_train, y_train)

# === Step 8: Evaluation ===
y_pred = pipeline.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Step 9: Cross-validation ===
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f"\n📈 Mean CV Accuracy: {np.mean(cv_scores):.4f}")

# === Step 10: Save model ===
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Full pipeline saved to '{MODEL_PATH}'")

# === Step 11: Save metadata ===
metadata = {
    "model_name": "RandomForestClassifier",
    "model_version": "1.0.0",
    "trained_on": datetime.utcnow().isoformat(),
    "categorical_features": categorical_cols,
    "numeric_features": numeric_cols,
    "target_label": label_col,
    "cv_accuracy": float(np.mean(cv_scores)),
    "test_report": classification_report(y_test, y_pred, output_dict=True),
    "class_names": list(np.unique(y)),
    "pipeline_model_path": MODEL_PATH,
    "preprocessing": {
        "pipeline": True,
        "encoding": "onehot"
    },
    "features": list(X.columns)  # X after all preprocessing/encoding
}
with open(METADATA_PATH, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"🗂️ Metadata saved to '{METADATA_PATH}'")

# === Step 12: Feature Importance Plot ===
# Use trained classifier from pipeline
classifier = pipeline.named_steps['classifier']
importances = classifier.feature_importances_

# Get transformed feature names
onehot = pipeline.named_steps['preprocessor'].named_transformers_['cat']
encoded_cols = onehot.get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([encoded_cols, numeric_cols])

# Plot
plt.figure(figsize=(10, 6))
indices = np.argsort(importances)[::-1]
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), all_feature_names[indices], rotation=90)
plt.tight_layout()
plt.savefig(FEATURE_IMPORTANCE_PATH)
print(f"📊 Feature importance plot saved to '{FEATURE_IMPORTANCE_PATH}'")
