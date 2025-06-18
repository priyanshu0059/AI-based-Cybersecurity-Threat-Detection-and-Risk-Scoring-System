from pymongo import MongoClient
from datetime import datetime

# MongoDB connection (default local)
client = MongoClient("mongodb://localhost:27017/")
db = client["cybersecurity_ai"]
collection = db["predictions"]

def log_prediction(input_data, prediction, probability, risk_score):
    record = {
        "timestamp": datetime.utcnow(),
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
        "risk_score": risk_score
    }
    collection.insert_one(record)
    print("✅ Prediction logged to MongoDB")

# Example usage
if __name__ == "__main__":
    example_input = {"Flow Duration": 12345, "Flow Bytes/s": 98.6}
    log_prediction(example_input, "THREAT", 0.86, 0.92)
