import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load your trained model
model = joblib.load("your_model.joblib")

# App title and description
st.title("💧 Water Quality Anomaly Detection")
st.markdown("""
### A Streamlit Edition of Our Thesis
Upload sensor data files containing **pH, flowrate, water level, turbidity, and water temperature**.  
Our trained machine learning model will analyze the data and flag anomalies that may indicate contamination or irregularities.
""")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file with sensor data", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV
    data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

    # Required columns
    required_cols = ["pH", "Flowrate", "WaterLevel", "Turbidity", "WaterTemperature"]

    if all(col in data.columns for col in required_cols):
        if st.button("Run Anomaly Detection"):
            predictions = model.predict(data[required_cols])
            data["Anomaly"] = predictions

            st.write("### Results with Anomaly Labels")
            st.dataframe(data)

            # Count anomalies
            anomaly_count = (predictions == -1).sum()
            st.warning(f"⚠️ Detected {anomaly_count} anomalies in the dataset.")

            # Visualization for each parameter
            st.write("### Sensor Data Visualizations with Anomalies")

            for col in required_cols:
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(data.index, data[col], label=col, color="blue")

                # Highlight anomalies
                anomaly_points = data[data["Anomaly"] == -1]
                ax.scatter(anomaly_points.index, anomaly_points[col],
                           color="red", label="Anomaly", marker="x", s=100)

                ax.set_xlabel("Sample Index")
                ax.set_ylabel(col)
                ax.set_title(f"{col} with Anomaly Detection")
                ax.legend()

                st.pyplot(fig)

    else:
        st.error(f"Uploaded file must contain columns: {required_cols}")
