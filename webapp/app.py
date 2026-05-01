import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


# Must be defined at module level so joblib can unpickle model4 pipelines
# (pickled as __main__.ExtraTextFeatures when train.py ran as __main__)
class ExtraTextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X).fillna("").astype(str).str.lower()
        extra = pd.DataFrame({
            "is_driveway": s.str.contains(r"\bdriveway\b", regex=True).astype(int),
            "is_parking":  s.str.contains(r"\bparking\b|\bparked\b|\bvehicle\b|\bcar\b", regex=True).astype(int),
            "is_blocked":  s.str.contains(r"\bblocked\b|\bblocking\b", regex=True).astype(int),
            "is_noise":    s.str.contains(r"\bbanging\b|\bpounding\b|\bloud\b|\bmusic\b", regex=True).astype(int),
        })
        return csr_matrix(extra.values)

# Default ZIP code for Nova Haven city center — change this to set the map default
CITY_CENTER_ZIP = "95336"

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
def load_hourly_points():
    df = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data/raw/city_traffic_accidents.csv",
        usecols=["Start_Lat", "Start_Lng", "Start_Time"],
    )
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["hour"] = df["Start_Time"].dt.hour
    df = df.dropna(subset=["Start_Lat", "Start_Lng", "hour"])
    return [df[df["hour"] == h][["Start_Lat", "Start_Lng"]].values.tolist() for h in range(24)]

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
    import joblib
    import tensorflow as tf
    model         = tf.keras.models.load_model("models/model2_deep_learning/saved_model/model.keras")
    scaler        = joblib.load("models/model2_deep_learning/saved_model/scaler.joblib")
    label_encoder = joblib.load("models/model2_deep_learning/saved_model/label_encoder.joblib")
    feature_cols  = joblib.load("models/model2_deep_learning/saved_model/feature_columns.joblib")
    metrics       = joblib.load("models/model2_deep_learning/saved_model/metrics.joblib")
    return model, scaler, label_encoder, feature_cols, metrics

@st.cache_resource
def load_cnn_model():
    import joblib
    import tensorflow as tf
    model     = tf.keras.models.load_model("models/model3_cnn/saved_model/model.keras")
    threshold = joblib.load("models/model3_cnn/saved_model/threshold.joblib")
    metrics   = joblib.load("models/model3_cnn/saved_model/metrics.joblib")
    return model, threshold, metrics

@st.cache_resource
def load_nlp_assets():
    import joblib
    routing_pipeline = joblib.load("models/model4_nlp_classification/saved_model/model4_routing_classifier_char_tfidf_SGD.pkl")
    routing_le       = joblib.load("models/model4_nlp_classification/saved_model/model4_routing_label_encoder.pkl")
    category_pipeline = joblib.load("models/model4_nlp_classification/saved_model/model4_category_classifier_char_tfidf_SGD.pkl")
    category_le       = joblib.load("models/model4_nlp_classification/saved_model/model4_category_label_encoder.pkl")
    return routing_pipeline, routing_le, category_pipeline, category_le

@st.cache_resource
def load_innovation_model():
    import joblib
    s = "models/model5_innovation/saved_model/"
    return (
        joblib.load(s + "outcome_clf.joblib"),
        joblib.load(s + "time_clf.joblib"),
        joblib.load(s + "tfidf.joblib"),
        joblib.load(s + "ord_enc.joblib"),
        joblib.load(s + "outcome_le.joblib"),
        joblib.load(s + "time_le.joblib"),
        joblib.load(s + "metrics.joblib"),
    )


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

    # --- Hourly Accident Heatmap (cached data, dynamic center based on ZIP) ---
    import folium as _folium
    from folium.plugins import HeatMapWithTime

    st.subheader("Accident Hotspot Heatmap by Hour")

    hotspots, zip_lookup = load_geo_data()

    def compute_hotspot_distance(z, hotspots, zip_lookup):
        z = z.strip().zfill(5)
        if z not in zip_lookup.index:
            return None, None, None
        u_lat = zip_lookup.loc[z, "lat"]
        u_lon = zip_lookup.loc[z, "lon"]
        dists = hotspots.apply(lambda r: haversine_miles(u_lat, u_lon, r["lat"], r["lon"]), axis=1)
        nearest_idx = dists.idxmin()
        return float(dists[nearest_idx]), hotspots.loc[nearest_idx, "Zipcode"], z

    # Read zip input from session state BEFORE building the map so the center is always current
    raw_zip = st.session_state.get("m1_zip_input", "")
    active_input = raw_zip.strip() if raw_zip.strip() else CITY_CENTER_ZIP
    dist, nearest_zip, resolved_z = compute_hotspot_distance(active_input, hotspots, zip_lookup)

    if dist is not None:
        st.session_state.zip_dist    = dist
        st.session_state.zip_source  = resolved_z
        st.session_state.nearest_zip = nearest_zip
        active_z = resolved_z
    else:
        active_z = CITY_CENTER_ZIP.zfill(5)

    if active_z in zip_lookup.index:
        center = [zip_lookup.loc[active_z, "lat"], zip_lookup.loc[active_z, "lon"]]
        zoom   = 11
    else:
        center = [37.0902, -95.7129]
        zoom   = 4

    hourly_points = load_hourly_points()
    m = _folium.Map(location=center, zoom_start=zoom)
    HeatMapWithTime(
        hourly_points,
        index=[f"{h}:00" for h in range(24)],
        radius=10,
        auto_play=True,
        max_opacity=0.8,
    ).add_to(m)
    _folium.Marker(
        center,
        popup=f"ZIP: {active_z}",
        icon=_folium.Icon(color="blue", icon="home"),
    ).add_to(m)
    st.components.v1.html(m.get_root().render(), height=450, scrolling=False)

    # ZIP input below the map — key persists value into session state for next render
    st.text_input(
        "Enter your ZIP code to re-center map and use distance as a feature",
        max_chars=5,
        placeholder=f"Default: {CITY_CENTER_ZIP}",
        key="m1_zip_input",
    )
    if raw_zip and dist is None:
        st.warning("ZIP not found in dataset — using city center distance.")

    if st.session_state.get("zip_dist") is not None:
        st.info(
            f"ZIP **{st.session_state.zip_source}** → nearest hotspot ZIP "
            f"**{st.session_state.nearest_zip}** — "
            f"**{st.session_state.zip_dist:.1f} miles** (used as Distance feature)"
        )
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

            # ZIP distance overrides the road-type default when available
            if st.session_state.zip_dist is not None and "Distance(mi)" in row:
                row["Distance(mi)"] = st.session_state.zip_dist

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
    st.header("🧠 Traffic Accident Severity — Deep Neural Network")


    # --- Static hotspot map (no time animation, no ZIP input) ---
    import folium as _folium

    st.subheader("Top Accident Hotspots")

    hotspots, zip_lookup = load_geo_data()

    z = CITY_CENTER_ZIP.zfill(5)
    if z in zip_lookup.index:
        center = [zip_lookup.loc[z, "lat"], zip_lookup.loc[z, "lon"]]
        zoom   = 11
    else:
        center = [hotspots["lat"].mean(), hotspots["lon"].mean()]
        zoom   = 5
    m2 = _folium.Map(location=center, zoom_start=zoom)

    for _, row in hotspots.iterrows():
        _folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6 + int(row["count"] / hotspots["count"].max() * 10),
            color="#dc3545",
            fill=True,
            fill_color="#dc3545",
            fill_opacity=0.6,
            popup=f"ZIP {row['Zipcode']} — {int(row['count'])} severe accidents",
        ).add_to(m2)

    st.components.v1.html(m2.get_root().render(), height=450, scrolling=False)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        road_type_dnn = st.selectbox("Road Category", ["Local", "High-Capacity Road", "Highway"], key="dnn_road")
        weather_dnn   = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Fog"], key="dnn_weather")
    with col2:
        speed_limit_dnn = st.slider("Speed Limit (mph)", 25, 75, 45, key="dnn_speed")
        time_of_day_dnn = st.selectbox("Time Window", ["Morning", "Afternoon", "Evening", "Night"], key="dnn_time")

    if st.button("Predict Severity (DNN)"):
        try:
            model, scaler, label_encoder, feature_cols, dnn_metrics = load_dnn_model()

            row = {col: 0 for col in feature_cols}

            road_map = {
                "Local":              {"Distance(mi)": 0.3, "n_road_features": 3, "has_traffic_control": 1},
                "High-Capacity Road": {"Distance(mi)": 0.8, "n_road_features": 2, "has_traffic_control": 1},
                "Highway":            {"Distance(mi)": 2.5, "n_road_features": 0, "has_traffic_control": 0},
            }
            for k, v in road_map[road_type_dnn].items():
                if k in row: row[k] = v

            weather_map = {
                "Clear": {"weather_cluster_clear": 1},
                "Rain":  {"weather_cluster_rain": 1, "has_precipitation": 1},
                "Snow":  {"weather_cluster_snow_ice": 1, "is_freezing": 1,
                          "low_visibility_severity": 1, "has_precipitation": 1},
                "Fog":   {"weather_cluster_low_visibility": 1, "low_visibility_severity": 1},
            }
            for k, v in weather_map[weather_dnn].items():
                if k in row: row[k] = v

            time_map = {
                "Morning":   {"is_morning_rush": 1, "is_rush_hour": 1},
                "Afternoon": {},
                "Evening":   {"is_evening_rush": 1, "is_rush_hour": 1, "low_visibility_severity": 1},
                "Night":     {"low_visibility_severity": 1},
            }
            for k, v in time_map[time_of_day_dnn].items():
                if k in row: row[k] = v

            speed_danger = 4 if speed_limit_dnn > 65 else (3 if speed_limit_dnn > 55 else (1 if speed_limit_dnn > 45 else 0))
            weather_danger = {"Clear": 0, "Rain": 2, "Fog": 2, "Snow": 3}[weather_dnn]
            if "DangerousScore" in row:
                row["DangerousScore"] = speed_danger + weather_danger

            X = pd.DataFrame([row])[feature_cols]
            X_scaled = scaler.transform(X)
            proba = model.predict(X_scaled, verbose=0)[0]

            # Apply same custom thresholds used in training
            if proba[0] >= 0.30:
                pred_enc = 0
            elif proba[3] >= 0.20:
                pred_enc = 3
            else:
                pred_enc = int(np.argmax(proba))

            severity    = int(label_encoder.inverse_transform([pred_enc])[0])
            confidence  = float(proba[pred_enc])

            severity_meta = {
                1: ("Minor",    "#28a745"),
                2: ("Moderate", "#ffc107"),
                3: ("Serious",  "#fd7e14"),
                4: ("Severe",   "#dc3545"),
            }
            sev_name, sev_color = severity_meta[severity]

            st.divider()

            # Colored severity badge
            st.markdown(
                f"""<div style="background:{sev_color};padding:22px 16px;border-radius:12px;text-align:center;">
                <h2 style="color:white;margin:0;">SEVERITY LEVEL {severity} — {sev_name.upper()}</h2>
                <p style="color:white;margin:6px 0 0;font-size:1.1rem;">
                    DNN Confidence: <strong>{confidence:.1%}</strong> &nbsp;|&nbsp;
                    Model Weighted F1: <strong>{dnn_metrics['weighted_f1']:.4f}</strong>
                </p></div>""",
                unsafe_allow_html=True,
            )

            st.divider()

            # Probability bar chart — all 4 severity levels
            st.subheader("Severity Class Probabilities")
            prob_chart = pd.DataFrame(
                {"Probability": proba},
                index=[f"Level {c}" for c in label_encoder.classes_],
            )
            st.bar_chart(prob_chart, color=sev_color)

            st.divider()

            # DNN vs XGBoost grouped bar chart
            st.subheader("DNN vs Traditional ML (XGBoost)")
            compare_chart = pd.DataFrame({
                "DNN":     [round(dnn_metrics["accuracy"], 2), round(dnn_metrics["weighted_f1"], 2), 0.47, 0.45],
                "XGBoost": [0.84, 0.81, 0.39, 0.11],
            }, index=["Accuracy", "Weighted F1", "Class 3 Recall", "Class 4 Recall"])
            st.bar_chart(compare_chart)
            st.caption("DNN trades some overall accuracy for better detection of severe (class 3 & 4) accidents.")

        except Exception as e:
            st.error(f"Error: {e}. Run models/model2_deep_learning/train.py first.")

# --- MODEL 3: CNN ---
elif model_choice == "Model 3: Road Inspection (CNN)":
    st.header("🧱 Pothole Detection — EfficientNetB0")
    st.write("Upload a road surface photo to detect potholes. "
             "The model uses transfer learning (EfficientNetB0) with Grad-CAM visualization.")

    uploaded_file = st.file_uploader("Upload road surface photo", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=480)

        if st.button("Scan for Potholes"):
            try:
                import tensorflow as tf
                import matplotlib.pyplot as plt

                model, threshold, cnn_metrics = load_cnn_model()
                preprocess_input = tf.keras.applications.efficientnet.preprocess_input

                # --- Preprocessing (matches training pipeline) ---
                img_tf = tf.constant(np.array(img), dtype=tf.float32) / 255.0

                # Road crop: vertical 30–78%, horizontal 5–95%
                h, w  = tf.shape(img_tf)[0], tf.shape(img_tf)[1]
                y1 = tf.cast(tf.cast(h, tf.float32) * 0.30, tf.int32)
                y2 = tf.cast(tf.cast(h, tf.float32) * 0.78, tf.int32)
                x1 = tf.cast(tf.cast(w, tf.float32) * 0.05, tf.int32)
                x2 = tf.cast(tf.cast(w, tf.float32) * 0.95, tf.int32)
                img_cropped     = img_tf[y1:y2, x1:x2, :]
                img_display     = img_cropped.numpy()
                img_resized     = tf.image.resize(img_cropped, (384, 384))
                img_preprocessed = preprocess_input(img_resized * 255.0)
                img_batch       = tf.expand_dims(img_preprocessed, axis=0)

                prob  = float(model.predict(img_batch, verbose=0)[0][0])
                label = "pothole" if prob >= threshold else "no_pothole"
                conf  = prob if label == "pothole" else 1 - prob

                # --- Result badge ---
                st.divider()
                badge_color = "#dc3545" if label == "pothole" else "#28a745"
                badge_text  = "POTHOLE DETECTED" if label == "pothole" else "ROAD CLEAR"
                st.markdown(
                    f"""<div style="background:{badge_color};padding:20px;border-radius:12px;text-align:center;">
                    <h2 style="color:white;margin:0;">{badge_text}</h2>
                    <p style="color:white;margin:6px 0 0;font-size:1.1rem;">
                        Confidence: <strong>{conf:.1%}</strong> &nbsp;|&nbsp;
                        Threshold: <strong>{threshold:.2f}</strong> &nbsp;|&nbsp;
                        Model Weighted F1: <strong>{cnn_metrics['weighted_f1']:.4f}</strong>
                    </p></div>""",
                    unsafe_allow_html=True,
                )

                # --- Grad-CAM ---
                st.divider()
                st.subheader("Grad-CAM — What the Model Sees")

                effnet_base    = model.layers[2]   # EfficientNetB0 backbone
                aug_layer      = model.layers[1]
                gap_layer      = model.layers[3]
                dropout_layer  = model.layers[4]
                dense_layer    = model.layers[5]

                with tf.GradientTape() as tape:
                    x            = aug_layer(img_batch, training=False)
                    feature_maps = effnet_base(x, training=False)
                    tape.watch(feature_maps)
                    x_out   = gap_layer(feature_maps)
                    x_out   = dropout_layer(x_out, training=False)
                    preds   = dense_layer(x_out)
                    score   = preds[:, 0]

                grads        = tape.gradient(score, feature_maps)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                fm           = feature_maps[0]
                heatmap      = tf.reduce_sum(fm * pooled_grads, axis=-1)
                heatmap      = tf.maximum(heatmap, 0)
                max_val      = tf.reduce_max(heatmap)
                heatmap      = heatmap / max_val if max_val > 0 else heatmap
                heatmap_np   = heatmap.numpy()

                heatmap_resized = tf.image.resize(
                    heatmap_np[..., np.newaxis], (img_display.shape[0], img_display.shape[1])
                ).numpy().squeeze()
                cmap    = plt.get_cmap("jet")
                overlay = np.clip((1 - 0.4) * img_display + 0.4 * cmap(heatmap_resized)[..., :3], 0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                axes[0].imshow(img_display);       axes[0].set_title("Cropped Input");  axes[0].axis("off")
                axes[1].imshow(heatmap_np, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
                axes[2].imshow(overlay);           axes[2].set_title(f"Overlay — {badge_text}"); axes[2].axis("off")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Error: {e}. Run models/model3_cnn/train.py first.")

# --- MODEL 4: NLP ---
elif model_choice == "Model 4: 311 Classifier (NLP)":
    st.header("🗣️ 311 Service Request Routing")

    AGENCY_NAMES = {
        "DCWP": "Dept. of Consumer & Worker Protection",
        "DEP":  "Dept. of Environmental Protection",
        "DHS":  "Dept. of Homeless Services",
        "DOB":  "Dept. of Buildings",
        "DOE":  "Dept. of Education",
        "DOHMH":"Dept. of Health & Mental Hygiene",
        "DOT":  "Dept. of Transportation",
        "DPR":  "Dept. of Parks & Recreation",
        "DSNY": "Dept. of Sanitation",
        "HPD":  "Housing Preservation & Development",
        "NPD":  "Nova Haven Police Department",
        "OOS":  "Out of Scope",
        "OTI":  "Office of Technology & Innovation",
        "TLC":  "Taxi & Limousine Commission",
    }

    EXAMPLES = [
        "Blocked driveway - car illegally parked blocking my access.",
        "HEAT/HOT WATER - no heat or hot water in my apartment for three days.",
        "Noise - residential: loud music and banging upstairs after midnight.",
        "Snow or ice blocking the sidewalk on my street.",
        "Illegal parking - vehicle blocking the fire hydrant.",
    ]

    st.markdown("**Try an example complaint:**")
    ex_cols = st.columns(len(EXAMPLES))
    for i, (col, ex) in enumerate(zip(ex_cols, EXAMPLES)):
        if col.button(f"Example {i+1}", key=f"ex_{i}"):
            st.session_state["nlp_complaint"] = ex

    complaint_text = st.text_area(
        "Resident Complaint Text:",
        value=st.session_state.get("nlp_complaint", ""),
        placeholder="Describe the issue...",
    )

    if st.button("Route to Agency") and complaint_text:
        try:
            import re as _re
            routing_pipeline, routing_le, category_pipeline, category_le = load_nlp_assets()
            clean = lambda t: _re.sub(r"\s+", " ", _re.sub(r"[^\w\s\-/&]", " ", str(t).strip().lower())).strip()
            cleaned = clean(complaint_text)

            # Agency routing
            route_encoded = routing_pipeline.predict([cleaned])[0]
            agency_code   = routing_le.inverse_transform([route_encoded])[0]
            route_proba   = routing_pipeline.predict_proba([cleaned])[0]
            confidence    = float(route_proba.max())

            # Complaint category
            cat_encoded   = category_pipeline.predict([cleaned])[0]
            category      = category_le.inverse_transform([cat_encoded])[0].title()

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Complaint Category", category)
            c2.metric("Route to Agency", f"{agency_code} — {AGENCY_NAMES.get(agency_code, '')}")
            c3.metric("Confidence", f"{confidence:.1%}")
            st.progress(confidence)

            # Top-3 agency probabilities
            top3_idx    = route_proba.argsort()[-3:][::-1]
            top3_labels = routing_le.inverse_transform(top3_idx)
            top3_probs  = route_proba[top3_idx]
            st.markdown("**Top agency matches:**")
            for label, prob in zip(top3_labels, top3_probs):
                st.markdown(f"- **{label}** ({AGENCY_NAMES.get(label, '')}) — {prob:.1%}")
        except Exception as e:
            st.error(f"NLP prediction error: {e}")

# --- MODEL 5: INNOVATION ---
elif model_choice == "Model 5: Innovation Module":
    import re as _re

    st.header("🔍 Resolution Outcome Predictor")
    st.write(
        "Predict whether a 311 complaint is likely to be **Resolved**, **Unresolved**, "
        "or **Referred** to another department — and estimate how long it will take. "
        "Trained on resolution text from 378 K closed complaints using NLP."
    )

    _CITY_SUBS = [
        (_re.compile(r"New York City Police Department", _re.I), "Nova Haven Police Department"),
        (_re.compile(r"New York City",                   _re.I), "Nova Haven"),
        (_re.compile(r"New Yorkers?",                    _re.I), "Nova Haven residents"),
        (_re.compile(r"\bNYC\b",                         _re.I), "Nova Haven"),
        (_re.compile(r"\bNYPD\b",                        _re.I), "NHPD"),
        (_re.compile(r"\bManhattan\b",                   _re.I), "Central District"),
        (_re.compile(r"\bBrooklyn\b",                    _re.I), "East District"),
        (_re.compile(r"\bBronx\b",                       _re.I), "North District"),
        (_re.compile(r"\bQueens\b",                      _re.I), "West District"),
        (_re.compile(r"Staten Island",                   _re.I), "South District"),
    ]

    def _clean(text):
        if not text:
            return ""
        text = _re.sub(r"[^\w\s\-/&]", " ", str(text).strip().lower())
        return _re.sub(r"\s+", " ", text).strip()

    col1, col2 = st.columns(2)
    with col1:
        complaint_type = st.text_input(
            "Complaint Type",
            placeholder="e.g. Noise - Residential, Heat/Hot Water, Illegal Parking",
        )
        descriptor = st.text_input(
            "Descriptor",
            placeholder="e.g. Loud Music/Party, Entire Building, Pothole",
        )
    with col2:
        m5_agency  = st.selectbox(
            "Agency",
            ["DCWP", "DEP", "DHS", "DOB", "DOE", "DOHMH", "DOT", "DPR", "DSNY", "HPD", "NPD", "TLC"],
            key="m5_agency",
        )
        m5_channel = st.selectbox(
            "Submission Channel",
            ["MOBILE", "ONLINE", "PHONE", "UNKNOWN"],
            key="m5_channel",
        )

    col3, col4 = st.columns(2)
    with col3:
        m5_date = st.date_input("Complaint Date", key="m5_date")
    with col4:
        m5_hour = st.slider("Hour of Day", 0, 23, 12, key="m5_hour")

    if st.button("Predict Outcome"):
        if not complaint_type.strip():
            st.warning("Please enter a complaint type.")
        else:
            try:
                (outcome_clf, time_clf, tfidf, ord_enc,
                 outcome_le, time_le, metrics) = load_innovation_model()

                input_text = _clean(complaint_type.strip() + " " + descriptor.strip())
                X_tfidf    = tfidf.transform([input_text])

                cat_arr    = np.array([[m5_agency, "Unspecified", m5_channel]])
                X_cat_enc  = ord_enc.transform(cat_arr)
                X_num      = np.array([[
                    m5_hour,
                    m5_date.weekday(),
                    m5_date.month,
                ]], dtype=float)
                X_numeric  = np.hstack([X_cat_enc, X_num])

                from scipy.sparse import hstack as sp_hstack, csr_matrix
                X_full = sp_hstack([X_tfidf, csr_matrix(X_numeric)])

                # Outcome prediction
                out_proba  = outcome_clf.predict_proba(X_full)[0]
                out_idx    = int(np.argmax(out_proba))
                outcome    = outcome_le.inverse_transform([out_idx])[0]
                out_conf   = float(out_proba[out_idx])

                # Time prediction
                time_proba = time_clf.predict_proba(X_full)[0]
                time_idx   = int(np.argmax(time_proba))
                time_lbl   = time_le.inverse_transform([time_idx])[0]
                time_conf  = float(time_proba[time_idx])

                outcome_color = {
                    "Resolved":   "success",
                    "Unresolved": "error",
                    "Referred":   "warning",
                }.get(outcome, "info")

                time_color = {
                    "Same Day": "success",
                    "1–7 Days": "warning",
                    "8+ Days":  "error",
                }.get(time_lbl, "info")

                st.divider()
                getattr(st, outcome_color)(f"Predicted Outcome: **{outcome}**")
                getattr(st, time_color)(f"Estimated Resolution Time: **{time_lbl}**")

                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Outcome",          outcome)
                c2.metric("Outcome Conf.",     f"{out_conf:.1%}")
                c3.metric("Est. Time",         time_lbl)
                c4.metric("Time Conf.",        f"{time_conf:.1%}")

                st.divider()
                st.markdown("**Outcome probabilities**")
                for cls, prob in zip(
                    outcome_le.classes_,
                    outcome_clf.predict_proba(X_full)[0],
                ):
                    st.markdown(f"- **{cls}** — {prob:.1%}")

                st.markdown("**Time probabilities**")
                for cls, prob in zip(
                    time_le.classes_,
                    time_clf.predict_proba(X_full)[0],
                ):
                    st.markdown(f"- **{cls}** — {prob:.1%}")

                with st.expander("Model performance"):
                    st.markdown(
                        f"Outcome weighted F1: **{metrics['outcome_f1']:.4f}**  \n"
                        f"Time weighted F1: **{metrics['time_f1']:.4f}**  \n"
                        f"Trained on **{metrics['n_train']:,}** closed complaints"
                    )

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ===========================================================================
# 4. FOOTER
# ===========================================================================
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 UrbanPulse Analytics | E2WS AI Topia")
