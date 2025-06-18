import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np

API_URL = "http://localhost:5000/predict"

# ----------------------------
# ⚙️ Page Config
# ----------------------------
st.set_page_config(page_title="Cybersecurity AI - Threat Detector", layout="wide")
st.title("🛡️ AI Cybersecurity Threat Detection & Risk Scoring")

# ----------------------------
# 🎨 Theme Toggle
# ----------------------------
theme_mode = st.sidebar.radio("🌓 Theme Mode", ["Light", "Dark"])

if theme_mode == "Dark":
    st.markdown("""
        <style>
            body, .stApp { background-color: #0e1117; color: #ffffff; }
            .stMetric label, .stMetric div { color: white; }
            .stButton>button { background-color: #333; color: white; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            .stButton>button { background-color: #f0f2f6; color: black; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

# ----------------------------
# 📊 Risk Level Logic
# ----------------------------
def get_risk_level(score):
    if score < 0.3:
        return "🔵 Low"
    elif score < 0.7:
        return "🟡 Medium"
    else:
        return "🔴 High"

# ----------------------------
# 📁 Input Options
# ----------------------------
st.sidebar.header("📁 Input Options")
input_mode = st.sidebar.radio("Choose Input Method", ["Manual Form", "Upload CSV"])

# ----------------------------
# 🔘 Manual Input Form
# ----------------------------
if input_mode == "Manual Form":
    with st.form("input_form"):
        st.subheader("📥 Enter Network Flow Features")
        col1, col2 = st.columns(2)

        with col1:
            fwd_packet_len_max = st.number_input("Fwd Packet Length Max", value=0.0)
            flow_duration = st.number_input("Flow Duration", value=0.0)
            total_fwd_packets = st.number_input("Total Length of Fwd Packets", value=0.0)
            flow_bytes_sec = st.number_input("Flow Bytes/s", value=0.0)
            flow_iat_mean = st.number_input("Flow IAT Mean", value=0.0)
            fwd_iat_total = st.number_input("Fwd IAT Total", value=0.0)
            fwd_header_len = st.number_input("Fwd Header Length", value=0.0)
            fwd_packets_sec = st.number_input("Fwd Packets/s", value=0.0)

        with col2:
            fwd_packet_len_min = st.number_input("Fwd Packet Length Min", value=0.0)
            total_bwd_packets = st.number_input("Total Length of Bwd Packets", value=0.0)
            flow_packets_sec = st.number_input("Flow Packets/s", value=0.0)
            flow_iat_max = st.number_input("Flow IAT Max", value=0.0)
            flow_iat_min = st.number_input("Flow IAT Min", value=0.0)
            fwd_iat_min = st.number_input("Fwd IAT Min", value=0.0)
            bwd_iat_min = st.number_input("Bwd IAT Min", value=0.0)
            bwd_header_len = st.number_input("Bwd Header Length", value=0.0)
            bwd_packets_sec = st.number_input("Bwd Packets/s", value=0.0)

        submitted = st.form_submit_button("🔍 Detect Threat")

    if submitted:
        payload = {
            "Fwd Packet Length Max": fwd_packet_len_max,
            "Fwd Packet Length Min": fwd_packet_len_min,
            "Flow Duration": flow_duration,
            "Total Length of Fwd Packets": total_fwd_packets,
            "Total Length of Bwd Packets": total_bwd_packets,
            "Flow Bytes/s": flow_bytes_sec,
            "Flow Packets/s": flow_packets_sec,
            "Flow IAT Mean": flow_iat_mean,
            "Flow IAT Max": flow_iat_max,
            "Flow IAT Min": flow_iat_min,
            "Fwd IAT Total": fwd_iat_total,
            "Fwd IAT Min": fwd_iat_min,
            "Bwd IAT Min": bwd_iat_min,
            "Fwd Header Length": fwd_header_len,
            "Bwd Header Length": bwd_header_len,
            "Fwd Packets/s": fwd_packets_sec,
            "Bwd Packets/s": bwd_packets_sec
        }

        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            if response.status_code == 200:
                st.success("✅ Prediction received")
                st.metric("Prediction", result["prediction"])
                st.metric("Probability", f"{result['probability'] * 100:.2f}%")
                st.metric("Risk Score", result["risk_score"])
                st.metric("Risk Level", get_risk_level(result["risk_score"]))

                if result["prediction"] == "THREAT":
                    st.error("⚠️ THREAT DETECTED!")
                else:
                    st.info("✅ Safe traffic")
            else:
                st.error("❌ Server error")
                st.json(result)

        except Exception as e:
            st.error(f"Request failed: {e}")

# ----------------------------
# 📤 CSV Upload Mode
# ----------------------------
elif input_mode == "Upload CSV":
    st.subheader("📤 Upload Network Flow CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Clean upfront
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        st.write("📄 Uploaded Data Preview", df.head())

        if st.button("🚀 Run Batch Prediction"):
            results = []
            with st.spinner("Predicting threats for each row..."):
                for i, row in df.iterrows():
                    clean_row = row.replace([np.inf, -np.inf], np.nan).fillna(0)
                    payload = clean_row.to_dict()

                    try:
                        response = requests.post(API_URL, json=payload)
                        if response.status_code == 200:
                            result = response.json()
                            result_row = {
                                **clean_row,
                                "Prediction": result["prediction"],
                                "Probability": result["probability"],
                                "Risk Score": result["risk_score"],
                                "Risk Level": get_risk_level(result["risk_score"]),
                                "Confidence Interval (±5%)": f"{max(result['probability'] - 0.05, 0):.2f} - {min(result['probability'] + 0.05, 1):.2f}"
                            }
                            results.append(result_row)
                        else:
                            st.warning(f"Row {i} failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error at row {i}: {e}")

            if results:
                results_df = pd.DataFrame(results)
                st.success("✅ All predictions completed.")
                st.dataframe(results_df)

                # 📊 Dashboard Summary
                st.subheader("📊 Dashboard Summary")

                # Pie Chart
                pie = px.pie(
                    names=results_df["Prediction"].value_counts().index,
                    values=results_df["Prediction"].value_counts().values,
                    title="Threat Distribution",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(pie, use_container_width=True)

                # Risk Score Line Chart
                st.line_chart(results_df["Risk Score"])

                # Heatmap
                numeric_cols = results_df.select_dtypes(include=["float64", "int"]).drop(columns=["Probability", "Risk Score"], errors='ignore')
                if not numeric_cols.empty:
                    st.subheader("🔥 Feature Correlation Heatmap")
                    corr = numeric_cols.corr()
                    heatmap = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu",
                        title="Feature Correlation"
                    )
                    st.plotly_chart(heatmap, use_container_width=True)

                # Download Results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="predicted_threats.csv",
                    mime="text/csv"
                )
