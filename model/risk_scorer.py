def calculate_risk_score(probability, predicted_label, label_name=None):
    print(f"[DEBUG] probability={probability} ({type(probability)}), predicted_label={predicted_label} ({type(predicted_label)}), label_name={label_name} ({type(label_name)})")
    """
    Calculate risk score based on model confidence and label type
    :param probability: Float between 0-1
    :param predicted_label: 0 or 1 (benign/threat)
    :param label_name: Optional (e.g., DoS, Bot, etc.)
    :return: Risk score (0–100)
    """

    # Ensure correct types before risk scoring
    try:
        probability = float(probability)
    except Exception:
        probability = 0.0

    try:
        predicted_label = int(float(predicted_label))
    except Exception:
        predicted_label = 0

    # Defensive: if probability is not in [0, 1], clamp it
    if probability < 0.0:
        probability = 0.0
    if probability > 1.0:
        probability = 1.0

    # Base scores
    if predicted_label == 0:
        return 0  # benign
    else:
        base_score = probability * 100

        # Optional multiplier for specific attack types
        attack_weights = {
            "DDoS": 1.2,
            "Bot": 1.1,
            "DoS": 1.3,
            "PortScan": 1.0,
            "BruteForce": 1.4,
            "Infiltration": 1.5
        }

        multiplier = attack_weights.get(str(label_name), 1.0)
        try:
            multiplier = float(multiplier)
        except Exception:
            multiplier = 1.0

        risk_score = base_score * multiplier

        try:
            return min(round(float(risk_score), 2), 100)  # Cap at 100
        except Exception:
            return 0.0
