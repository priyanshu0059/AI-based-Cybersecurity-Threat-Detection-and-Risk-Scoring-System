CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_data JSON,
    prediction VARCHAR(10),
    probability FLOAT,
    risk_score FLOAT
);
