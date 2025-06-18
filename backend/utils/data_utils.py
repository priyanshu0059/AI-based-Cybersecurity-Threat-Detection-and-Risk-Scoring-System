import pandas as pd

# You can load this from a config file or manually define
EXPECTED_FEATURES = [
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Flow Duration", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Min", "Bwd IAT Min",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s"
    # Add all features your model expects (from training phase)
]

def preprocess_input(data_dict):
    """
    Ensure input dict matches expected features in order.
    :param data_dict: JSON-like input from API
    :return: pd.DataFrame ready for model
    """
    try:
        # Ensure feature order and missing ones set to 0
        ordered_data = {feat: data_dict.get(feat, 0) for feat in EXPECTED_FEATURES}
        df = pd.DataFrame([ordered_data])
        return df
    except Exception as e:
        raise ValueError(f"Input preprocessing error: {str(e)}")
