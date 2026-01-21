import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load separate models for each parameter
models = {
    "pH": joblib.load("IF_pH_level.joblib"),
    "Flowrate": joblib.load("IF_flow_rate.joblib"),
    "WaterLevel": joblib.load("IF_water_level.joblib"),
    "Turbidity": joblib.load("IF_turbidity.joblib"),
    "WaterTemperature": joblib.load("IF_temperature.joblib"),
}

# App title and description
st.title("💧 Water Quality Anomaly Detection")
st.markdown("""
### A Streamlit Edition of Our Thesis
This app uses **separate Isolation Forest models** for each water quality parameter:  
**pH, Flowrate, Water Level, Turbidity, and Water Temperature.**  
Upload your sensor dataset and the app will detect anomalies per parameter.
""")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file with sensor data", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

    required_cols = list(models.keys())

    if all(col in data.columns for col in required_cols):
        if st.button("Run Anomaly Detection"):
            # Run each model separately
            for col in required_cols:
                predictions = models[col].predict(data[[col]])
                data[f"{col}_Anomaly"] = predictions

                # Visualization
                st.write(f"### {col} with Anomaly Detection")
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(data.index, data[col], label=col, color="blue")

                anomaly_points = data[data[f"{col}_Anomaly"] == -1]
                ax.scatter(anomaly_points.index, anomaly_points[col],
                           color="red", label="Anomaly", marker="x", s=100)

                ax.set_xlabel("Sample Index")
                ax.set_ylabel(col)
                ax.set_title(f"{col} with Anomaly Detection")
                ax.legend()

                st.pyplot(fig)

            st.write("### Results with Anomaly Labels")
            st.dataframe(data)

    else:
        st.error(f"Uploaded file must contain columns: {required_cols}")
