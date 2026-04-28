import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

# ===========================================================================
# 1. PAGE CONFIGURATION & BRANDING
# ===========================================================================
st.set_page_config(
    page_title="UrbanPulse Analytics | Nova Haven",
    page_icon="🏙️",
    layout="wide",
)
st.markdown("""
    <style>
    /* Use Streamlit's native background and text variables */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Sidebar - uses a slightly offset transparency to look good on both */
    section[data-testid="stSidebar"] {
        background-color: rgba(151, 166, 195, 0.1) !important;
    }

    /* Metric Cards - adapt border and background to theme */
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }

    /* Labels - Use secondary text color for a subtle look */
    [data-testid="stMetricLabel"] p {
        color: var(--text-color) !important;
        opacity: 0.8;
        font-size: 1rem !important;
    }

    /* Values - Use Primary accent color for consistency */
    [data-testid="stMetricValue"] div {
        color: var(--primary-color) !important;
        font-weight: bold !important;
    }

    /* Headers - Use Primary accent color */
    h1, h2, h3 {
        color: var(--primary-color) !important;
    }

    /* Buttons - Use Primary color and automatically adjust text contrast */
    .stButton>button {
        background-color: var(--primary-color);
        color: white; /* Streamlit buttons usually handle contrast automatically, but white is safe for primary */
        border: none;
        width: 100%;
    }
    
    /* Optional: Hover effect for buttons using the primary color with filter */
    .stButton>button:hover {
        filter: brightness(0.9);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# Sidebar Branding
st.sidebar.title("🏙️ UrbanPulse")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Select Intelligence Module",
    [
        "Home", 
        "Model 1: Traffic Severity (ML)", 
        "Model 2: Resource Allocation (DNN)", 
        "Model 3: Road Inspection (CNN)", 
        "Model 4: 311 Classifier (NLP)", 
        "Model 5: Innovation Module"
    ]
)

# ===========================================================================
# 2. CACHED MODEL LOADERS
# ===========================================================================
@st.cache_resource
def load_ml_model():
    import joblib
    model        = joblib.load("models/model1_traditional_ml/saved_model/model.joblib")
    scaler       = joblib.load("models/model1_traditional_ml/saved_model/scaler.joblib")
    le           = joblib.load("models/model1_traditional_ml/saved_model/label_encoder.joblib")
    feature_cols = joblib.load("models/model1_traditional_ml/saved_model/feature_columns.joblib")
    return model, scaler, le, feature_cols

@st.cache_resource
def load_dnn_model():
    import tensorflow as tf
    return tf.keras.models.load_model("models/model2_deep_learning/saved_model/model.keras")

@st.cache_resource
def load_cnn_model():
    import tensorflow as tf
    return tf.keras.models.load_model("models/model3_cnn/saved_model/model.keras")

@st.cache_resource
def load_nlp_assets():
    import joblib
    model = joblib.load("models/model4_nlp_classification/saved_model/model.joblib")
    vectorizer = joblib.load("models/model4_nlp_classification/saved_model/vectorizer.joblib")
    return model, vectorizer

# ===========================================================================
# 3. ROUTING & PAGES
# ===========================================================================

# --- HOME PAGE ---
if model_choice == "Home":
    st.title("🏙️ UrbanPulse Analytics Dashboard")
    st.subheader("Nova Haven Smart City Initiative")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Population", "8.3M")
    col2.metric("Road Miles", "19,000")
    col3.metric("Annual 311s", "3.4M")
    col4.metric("Efficiency Goal", "15% +")

    st.markdown("""
    ---
    ### Strategic AI Solutions
    This dashboard provides a centralized interface for Nova Haven's AI-driven urban operations.
    
    * **Traffic Safety:** Predicting accident severity to optimize emergency response.
    * **Infrastructure:** Automating pothole detection using computer vision.
    * **Resident Services:** Intelligent routing of 311 service requests.
    * **Innovation:** Advanced analytics for city policy and resource equity.
    """)

# --- MODEL 1: TRADITIONAL ML ---
elif model_choice == "Model 1: Traffic Severity (ML)":
    st.header("🚦 Traffic Accident Severity Prediction")
    st.write("Analyze environmental factors to predict accident impact.")

    col1, col2 = st.columns(2)
    with col1:
        road_type = st.selectbox("Road Category", ["Local", "Arterial", "Highway"])
        weather = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Fog"])
    with col2:
        speed_limit = st.slider("Speed Limit (mph)", 25, 75, 45)
        time_of_day = st.selectbox("Time Window", ["Morning", "Afternoon", "Evening", "Night"])

    if st.button("Calculate Severity Risk"):
        try:
            model, scaler, le, feature_cols = load_ml_model()

            hour = {"Morning": 8, "Afternoon": 14, "Evening": 17, "Night": 22}[time_of_day]
            is_morning_rush = int(7 <= hour <= 9)
            is_evening_rush = int(16 <= hour <= 19)

            overrides = {
                "Distance(mi)":                   speed_limit / 50.0,
                "is_morning_rush":                is_morning_rush,
                "is_evening_rush":                is_evening_rush,
                "is_rush_hour":                   int(is_morning_rush or is_evening_rush),
                "is_freezing":                    int(weather == "Snow"),
                "low_visibility_severity":        int(weather == "Fog"),
                "has_precipitation":              int(weather in ("Rain", "Snow")),
                "weather_cluster_clear":          int(weather == "Clear"),
                "weather_cluster_rain":           int(weather == "Rain"),
                "weather_cluster_snow_ice":       int(weather == "Snow"),
                "weather_cluster_low_visibility": int(weather == "Fog"),
                "n_road_features":                {"Local": 1, "Arterial": 3, "Highway": 2}[road_type],
                "has_traffic_control":            {"Local": 0, "Arterial": 1, "Highway": 0}[road_type],
                "DangerousScore":                 (2 if weather == "Snow" else 1 if weather in ("Rain", "Fog") else 0)
                                                  + int(is_morning_rush or is_evening_rush)
                                                  + int(speed_limit >= 65),
            }

            row = {col: overrides.get(col, 0) for col in feature_cols}
            X = pd.DataFrame([row])[feature_cols]
            X_scaled = scaler.transform(X)

            proba      = model.predict_proba(X_scaled)[0]
            pred_enc   = np.argmax(proba)
            prediction = le.inverse_transform([pred_enc])[0]
            conf       = float(proba[pred_enc])

            st.divider()
            st.metric("Risk Level", f"Severity {prediction}", delta=f"{conf:.1%} Confidence")
        except Exception as e:
            st.error(f"Error loading Model 1: {e}")

# --- MODEL 2: DEEP LEARNING ---
elif model_choice == "Model 2: Resource Allocation (DNN)":
    st.header("⚙️ Deep Learning Resource Allocation ")
    
    # 1. Expand inputs to match your actual dataset features
    col1, col2 = st.columns(2)
    with col1:
        volume = st.number_input("Historical Volume Index", 0.0, 100.0, 50.0)
    with col2:
        staffing = st.number_input("Current Staffing Level", 1, 50, 10)

    if st.button("Predict Allocation Score"):
        try:
            model = load_dnn_model()
            # 2. Make sure the input shape matches what you trained (e.g., 2 features)
            input_data = np.array([[volume, staffing]]) 
            prediction = model.predict(input_data)
            
            # 3. Display the results clearly
            st.subheader("Model Output")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", f"{prediction[0][0]:.2f}")
            c2.metric("Probability", "0.85") # Placeholder for your model's prob
            c3.metric("Confidence", "High")
            
        except Exception as e:
            st.error(f"Error: {e}. Ensure model is in models/model2_deep_learning/saved_model/")

# --- MODEL 3: CNN ---
elif model_choice == "Model 3: Road Inspection (CNN)":
    st.header("🧱 Vision AI: Infrastructure Inspection")
    uploaded_file = st.file_uploader("Upload road surface photo...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Inspection Capture", width=500)
        
        if st.button("Scan for Damage"):
            try:
                model = load_cnn_model()
                # Preprocessing: match your training size (e.g., 224x224)
                img_proc = img.resize((224, 224))
                img_array = np.array(img_proc) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)
                
                prediction = model.predict(img_batch)
                label = "Pothole Detected" if prediction[0][0] > 0.5 else "Road Clear"
                
                st.subheader(f"Analysis: {label}")
                st.progress(float(prediction[0][0]))
            except Exception as e:
                st.error("CNN Model file missing from models/model3_cnn/saved_model/")

# --- MODEL 4: NLP ---
elif model_choice == "Model 4: 311 Classifier (NLP)":
    st.header("🗣️ 311 Service Request Routing")
    complaint_text = st.text_area("Resident Complaint Text:", placeholder="Describe the issue...")

    if st.button("Route to Agency") and complaint_text:
        try:
            model, vectorizer = load_nlp_assets()
            vec_text = vectorizer.transform([complaint_text])
            prediction = model.predict(vec_text)[0]
            
            st.success(f"Recommended Routing: **{prediction}**")
            st.info("Text successfully classified using Nova Haven's NLP pipeline.")
        except Exception as e:
            st.error("NLP assets (model.joblib or vectorizer.joblib) not found.")

# --- MODEL 5: INNOVATION ---
elif model_choice == "Model 5: Innovation Module":
    st.header("💡 Vanguard Innovation Lab")
    st.write("This module explores advanced predictive maintenance and city planning scenarios.")
    # Add your unique Vanguard Systems AI feature here
    st.warning("Vanguard Module: Training data integration in progress.")

# ===========================================================================
# 4. FOOTER
# ===========================================================================
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 UrbanPulse Analytics | E2WS AI Topia")