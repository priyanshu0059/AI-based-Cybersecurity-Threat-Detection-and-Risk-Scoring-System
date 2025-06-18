# 🛡️ AI-based Cybersecurity Threat Detection and Risk Scoring System

This project is an intelligent system designed to automatically detect cybersecurity threats and assign risk scores to network traffic. It uses machine learning models to analyze traffic features and classify them as benign or malicious with high accuracy.

---

## 📌 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Features

- Detects cyber threats in real-time
- Assigns risk scores to each input
- Preprocessing pipeline for categorical and numeric features
- High model accuracy with cross-validation
- Easy integration via API

---

## 🛠️ Technologies Used

- Python 3.9+
- Scikit-learn
- Pandas, NumPy
- Flask / FastAPI (for API serving)
- Git / GitHub
- Jupyter Notebook (for EDA)
- VS Code

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/AI-based-Cybersecurity-Threat-Detection-and-Risk-Scoring-System.git
cd AI-based-Cybersecurity-Threat-Detection-and-Risk-Scoring-System

Create and activate virtual environment   python -m venv venv
source venv/bin/activate      # For Linux/macOS
venv\Scripts\activate         # For Windows  Install dependencies  pip install -r requirements.txt


##  📊 **Model Details
Model Used: RandomForestClassifier

Preprocessing: One-hot encoding with pipeline

Target Label: label

Accuracy: 99.91% (Cross-validation)

Classes: 50+ attack/normal types

Input Features: Network traffic attributes like srcip, proto, sport, dur, sbytes, dbytes, etc.

Model metadata:

Stored in model_metadata.json

Preprocessing pipeline: full_pipeline.pkl

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
