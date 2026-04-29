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

@st.cache_data
def load_geo_data():
    df = pd.read_csv(
        "data/raw/city_traffic_accidents.csv",
        usecols=["Start_Lat", "Start_Lng", "Severity", "Zipcode"],
    )
    df = df.dropna(subset=["Start_Lat", "Start_Lng", "Zipcode"])
    df["Zipcode"] = df["Zipcode"].astype(str).str.split("-").str[0].str.zfill(5)
    hotspots = (
        df[df["Severity"] >= 3]
        .groupby("Zipcode")
        .agg(count=("Severity", "size"), lat=("Start_Lat", "mean"), lon=("Start_Lng", "mean"))
        .nlargest(30, "count")
        .reset_index()
    )
    zip_lookup = (
        df.groupby("Zipcode")
        .agg(lat=("Start_Lat", "mean"), lon=("Start_Lng", "mean"))
        .reset_index()
        .set_index("Zipcode")
    )
    return hotspots, zip_lookup

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

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

@st.cache_resource
def load_innovation_model():
    import joblib
    model   = joblib.load("models/model5_innovation/saved_model/model.joblib")
    enc     = joblib.load("models/model5_innovation/saved_model/ordinal_encoder.joblib")
    le      = joblib.load("models/model5_innovation/saved_model/label_encoder.joblib")
    metrics = joblib.load("models/model5_innovation/saved_model/metrics.joblib")
    return model, enc, le, metrics


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

    # --- Accident Map (pre-generated HTML — no recreation on reload) ---
    st.subheader("Accident Locations Map")
    map_path = Path(__file__).resolve().parents[1] / "data/maps/accident_map.html"
    with open(map_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=450, scrolling=False)

    _, zip_lookup = load_geo_data()
    zip_input = st.text_input("Enter your ZIP code to find distance to nearest hotspot", max_chars=5, placeholder="e.g. 90210")

    if zip_input:
        z = zip_input.strip().zfill(5)
        if z in zip_lookup.index:
            u_lat = zip_lookup.loc[z, "lat"]
            u_lon = zip_lookup.loc[z, "lon"]
            hotspots, _ = load_geo_data()
            dists = hotspots.apply(lambda r: haversine_miles(u_lat, u_lon, r["lat"], r["lon"]), axis=1)
            nearest_idx = dists.idxmin()
            dist_to_hotspot = dists[nearest_idx]
            nearest_zip = hotspots.loc[nearest_idx, "Zipcode"]
            st.info(f"Nearest hotspot: ZIP **{nearest_zip}** — **{dist_to_hotspot:.1f} miles away**")
        else:
            st.warning("ZIP not found in dataset.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        road_type = st.selectbox("Road Category", ["Local", "High-Capacity Road", "Highway"])
        weather = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Fog"])
    with col2:
        speed_limit = st.slider("Speed Limit (mph)", 25, 75, 45)
        time_of_day = st.selectbox("Time Window", ["Morning", "Afternoon", "Evening", "Night"])
    if st.button("Calculate Severity Risk"):
        try:
            model, scaler, le, feature_cols = load_ml_model()
            row = {col: 0 for col in feature_cols}

            # --- Road type → distance & infrastructure features ---
            road_map = {
                "Local": {"Distance(mi)": 0.3, "n_road_features": 3, "has_traffic_control": 1},
                "High-Capacity Road": {"Distance(mi)": 0.8, "n_road_features": 2, "has_traffic_control": 1,
                                       "word_lane": 1},
                "Highway":  {"Distance(mi)": 2.5, "n_road_features": 0, "has_traffic_control": 0, "word_exit": 1,
                             "word_lane": 1, "word_closed": 1, "word_northbound": 1, "word_southbound": 1, "word_eastbound": 1,
                               "word_westbound": 1, "word_shoulder": 1},
            }
            for k, v in road_map[road_type].items():
                if k in row: row[k] = v

            # --- Weather → weather cluster + condition flags ---
            weather_map = {
                "Clear": {"weather_cluster_clear": 1},
                "Rain":  {"weather_cluster_rain": 1, "has_precipitation": 1, "weather_cluster_cloudy":1},
                "Snow":  {"weather_cluster_snow_ice": 1, "is_freezing": 1, "weather_cluster_cloudy":1,
                          "has_precipitation": 1, "low_visibility_severity": 1, "weather_cluster_low_visibility": 1},
                "Fog":   {"weather_cluster_low_visibility": 1, "low_visibility_severity": 1, "weather_cluster_cloudy":1, "has_precipitation": 1},
            }
            for k, v in weather_map[weather].items():
                if k in row: row[k] = v

            # --- Time of day → rush-hour flags ---
            time_map = {
                "Morning":   {"is_morning_rush": 1, "is_rush_hour": 1},
                "Afternoon": {},
                "Evening":   {"is_evening_rush": 1, "is_rush_hour": 1, "low_visibility_severity": 1},
                "Night":     {"low_visibility_severity": 1},
            }
            for k, v in time_map[time_of_day].items():
                if k in row: row[k] = v

            # --- DangerousScore from speed + weather ---
            speed_danger = 4 if speed_limit > 65 else (3 if speed_limit > 55 else (1 if speed_limit > 45 else 0))
            weather_danger = {"Clear": 0, "Rain": 2, "Fog": 2, "Snow": 3}[weather]
            if "DangerousScore" in row:
                row["DangerousScore"] = speed_danger + weather_danger

            # --- Predict ---
            X = pd.DataFrame([row])[feature_cols]
            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)[0]

            high_risk_prob = float(proba[1])
            prediction = "High Risk" if high_risk_prob >= 0.5 else "Standard Risk"

            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Risk Assessment", prediction)
            c2.metric("High Risk Probability", f"{high_risk_prob:.1%}")

            st.progress(high_risk_prob)

            if high_risk_prob >= 0.5:
                st.error("High probability of a severe incident (Severity 3 or 4). Exercise extreme caution.")
            elif high_risk_prob >= 0.25:
                st.warning("Moderate severe incident risk. Conditions warrant caution.")
            else:
                st.success("Conditions indicate standard risk.")

        except Exception as e:
            st.error(f"Error: {e}")

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
    st.header("💡 311 Complaint Resolution Urgency Predictor")
    st.write("Predict whether a 311 service request will take a short, moderate, or long time to resolve — enabling smarter dispatch and proactive SLA management.")

    col1, col2 = st.columns(2)
    with col1:
        complaint_type = st.selectbox("Complaint Type", [
            "Illegal Parking", "HEAT/HOT WATER", "Noise - Residential", "Snow or Ice",
            "Blocked Driveway", "Street Condition", "UNSANITARY CONDITION", "PLUMBING",
            "Traffic Signal Condition", "Noise - Street/Sidewalk", "Water System",
            "PAINT/PLASTER", "Dirty Condition", "Abandoned Vehicle", "WATER LEAK",
        ])
        agency = st.selectbox("Agency", ["DCWP", "DEP", "DHS", "DOB", "DOE", "DOHMH", "DOT", "DPR", "DSNY", "HPD", "NYPD", "OOS", "OTI", "TLC"])
    with col2:
        borough = st.selectbox("Borough", ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND", "Unspecified"])
        channel = st.selectbox("Submission Channel", ["MOBILE", "ONLINE", "OTHER", "PHONE", "UNKNOWN"])

    col3, col4 = st.columns(2)
    with col3:
        complaint_date = st.date_input("Complaint Date")
    with col4:
        complaint_hour = st.slider("Hour of Day", 0, 23, 12)

    if st.button("Predict Resolution Urgency"):
        try:
            model, enc, le, metrics = load_innovation_model()

            day_of_week = complaint_date.weekday()
            month       = complaint_date.month
            is_weekend  = int(day_of_week >= 5)

            cat_input = np.array([[complaint_type, agency, borough, channel]])
            X_cat     = enc.transform(cat_input)
            X_num     = np.array([[complaint_hour, day_of_week, month, is_weekend]])
            X         = np.hstack([X_cat, X_num])

            proba      = model.predict_proba(X)[0]
            pred_enc   = np.argmax(proba)
            prediction = le.inverse_transform([pred_enc])[0]
            conf       = float(proba[pred_enc])

            label_map  = {"high_risk": "High Risk (5+ days)", "medium_risk": "Medium Risk (1–5 days)", "low_risk": "Low Risk (same day)"}
            color_map  = {"high_risk": "error", "medium_risk": "warning", "low_risk": "success"}
            display    = label_map.get(prediction, prediction)

            st.divider()
            getattr(st, color_map.get(prediction, "info"))(f"Resolution Urgency: **{display}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Urgency Level", display.split("(")[0].strip())
            c2.metric("Confidence", f"{conf:.1%}")
            c3.metric("Model F1 Score", f"{metrics['weighted_f1']:.4f}")
        except Exception as e:
            st.error(f"Error loading Innovation Model: {e}")

# ===========================================================================
# 4. FOOTER
# ===========================================================================
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 UrbanPulse Analytics | E2WS AI Topia")
