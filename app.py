import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load separate models for each parameter
models = {
    "pH": joblib.load("model_ph.joblib"),
    "Flowrate": joblib.load("model_flowrate.joblib"),
    "WaterLevel": joblib.load("model_waterlevel.joblib"),
    "Turbidity": joblib.load("model_turbidity.joblib"),
    "WaterTemperature": joblib.load("model_temperature.joblib"),
}

# App title and description
st.title("💧 Water Quality Anomaly Detection")
st.markdown("""
### A Streamlit Edition of Our Thesis  
This app uses **separate Isolation Forest models** for each water quality parameter:  
**pH, Flowrate, Water Level, Turbidity, and Water Temperature.**  
Upload your sensor dataset and the app will detect anomalies per parameter and visualize them.
""")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV file with sensor data", type="csv")

if uploaded_file is not None:
    # Read and normalize column names
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip().str.lower()

    # Map actual column names to expected ones
    column_map = {
        "ph_level": "pH",
        "flow_rate": "Flowrate",
        "water_level": "WaterLevel",
        "turbidity": "Turbidity",
        "temperature": "WaterTemperature"
    }
    data.rename(columns=column_map, inplace=True)

    required_cols = list(models.keys())

    # Check if all required columns are present
    if all(col in data.columns for col in required_cols):
        st.write("### ✅ Uploaded Data Preview")
        st.dataframe(data.head())

        if st.button("🚀 Run Anomaly Detection"):
            # Run each model and visualize
            for col in required_cols:
                predictions = models[col].predict(data[[col]])
                data[f"{col}_Anomaly"] = predictions

                st.write(f"### 📊 {col} with Anomaly Detection")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data.index, data[col], label=col, color="blue")

                anomaly_points = data[data[f"{col}_Anomaly"] == -1]
                ax.scatter(anomaly_points.index, anomaly_points[col],
                           color="red", label="Anomaly", marker="x", s=100)

                ax.set_xlabel("Sample Index")
                ax.set_ylabel(col)
                ax.set_title(f"{col} with Anomaly Detection")
                ax.legend()
                st.pyplot(fig)

            # Show final results
            st.write("### 🧾 Results with Anomaly Labels")
            st.dataframe(data)

            # Optional: summary bar chart
            st.write("### 📈 Anomaly Summary")
            anomaly_counts = {col: (data[f"{col}_Anomaly"] == -1).sum() for col in required_cols}
            summary_df = pd.DataFrame.from_dict(anomaly_counts, orient='index', columns=['Anomaly Count'])
            st.bar_chart(summary_df)

    else:
        st.error(f"❌ Uploaded file must contain columns: {list(column_map.keys())}")
