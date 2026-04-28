"""
Shared Data Pipeline
====================
Shared data loading and preprocessing functions used across all models.
Put your common data cleaning, feature engineering, and splitting logic here.

Usage from any model:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.data_pipeline import load_raw_data, preprocess, split_data
"""
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter
import re

#Create interactive map
import os
import folium 
from folium.plugins import HeatMapWithTime
from folium.plugins import FastMarkerCluster

# Zipcode lookup
try:
    from uszipcode import SearchEngine
    ZIPCODE_SEARCH_AVAILABLE = True
except (ImportError, AttributeError) as e:
    ZIPCODE_SEARCH_AVAILABLE = False

#Set Regions
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Environmental data
import openmeteo_requests
from astral import Observer
from astral.sun import sun
# Add project root to sys.path
root_path = Path.cwd().parent 
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

#Import
from pipelines.data_pipeline import convert_bools_to_ints, create_temporal_features
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================================================================================
# Processing & Feature Engineering for City Traffic Accident Data
# ============================================================================================================================
def accident_engineer_features(df):
    """
    Production-ready feature engineering pipeline.
    
    USAGE:
        df = accident_engineer_features(df)
    """
    # 1. Basic Cleaning & Dropping
    df = df.drop(columns=['Country', 'ID', 'Source'], errors='ignore')

    #Cleans and fills empty columns
    df = accident_engineer_empty_columns(df)

    #Feature Engineering non-numertical columns
    df= descriptor_word_count(df)      #Engineer a feature that counts the number of words in the Description column, which may correlate with accident severity or complexity.

    #Categorize Weather Conditions
    df = process_weather_features(df)

    #one-hot encode 
    if 'Region' in df.columns:
        df = pd.concat([df.drop(columns=['Region']), pd.get_dummies(df['Region'], prefix='region', dummy_na=False, dtype=int)], axis=1)
    if 'Wind_Direction' in df.columns:
        df = pd.concat([df.drop(columns=['Wind_Direction']), pd.get_dummies(df['Wind_Direction'], prefix='wind', dummy_na=False, dtype=int)], axis=1)

    df = process_road_features(df)              #Creating aggregate features for road and traffic

    # Find top 5 zipcde in each region and group the rest into "other" category
    df = create_zipcode_features(df)

    # Group cities outside of top 20 into "Other" category to reduce cardinality
    df= encode_top_geo_features(df)

    # Drop any remaining irrelevant or redundant columns (e.g., Street, if it was too noisy and we filled geographic details from lat/lng)
    df = df.drop(columns=['State', 'Zipcode', 'City', 'County', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng'], errors='ignore')

    #logic for the matching dangerous weather patterns
    df = dangerous_conditions_score(df)

    #Engineer aggregate features for road conditions (e.g., total_road_features, has_traffic_control)
    df= engineer_road_features(df) 

    df=convert_bools_to_ints(df) #Convert boolean columns to integers for modeling
    #Retrun the processed DataFrame
    return df

# =============================================================================
# Cleans and fills empty columns for City Traffic Accident Data
# =============================================================================
def accident_engineer_empty_columns(df):
    """
    Cleans and fills empty columns using a combination of logic, 
    lookup tables, 
    and local calculations.
    
    USAGE:
        df = accident_engineer_empty_columns(df)
    """

    #Datetime Conversion
    time_cols = ['Start_Time', 'End_Time', 'Weather_Timestamp']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    #Time Imputation
    df['Start_Time'] = df['Start_Time'].fillna(df['Weather_Timestamp'])             #If Start_Time is missing, use Weather_Timestamp as a proxy (assuming weather data is timestamped at the time of the accident)
    df['End_Time'] = df['End_Time'].fillna(df['Weather_Timestamp'])                 #If End_Time is missing, use Weather_Timestamp as a proxy (assuming weather data is timestamped at the time of the accident)
    df['Weather_Timestamp'] = df['Weather_Timestamp'].fillna(df['Start_Time'])      #If Weather_Timestamp is missing, use Start_Time as a proxy (assuming weather data is timestamped at the time of the accident)

    # Temporal Features (Hour, Month, Rush Hour)
    df = create_temporal_features(df)                                               #Extract hour of day, day of week, month, and rush hour flags from Start_Time to capture time-based patterns in accidents

    #Fill Coordinates & Geographic Details
    if 'End_Lat' in df.columns:
        df['End_Lat'] = df['End_Lat'].fillna(df['Start_Lat'])
    if 'End_Lng' in df.columns:
        df['End_Lng'] = df['End_Lng'].fillna(df['Start_Lng'])
    
     #Geographic & Regional Logic
    df = fill_geographic_data(df)                                               #Use ZIP code lookup or reverse geocoding to fill missing geographic details (e.g., city, county) based on available lat/lng or zip code data
    df = df.drop(columns=['Street'], errors='ignore')                           #Drop Street column after filling geographic details, as it may be too noisy or sparse to be useful
    df = add_census_regions(df)                                                     #Add Census Region based on State to capture regional patterns in accidents
    df = create_cluster_regions(df, n_clusters=10)                                  #Create geographic clusters (e.g., using KMeans on lat/lng) to capture local accident hotspots 
    df = add_intra_region_distances(df, cluster_col='Geo_Cluster')                  #Calculate distance to cluster centers to capture how far an accident is from local hotspots
    #WEATHER & ENVIRONMENT DATA - Fast processing
    df = fast_environmental_data(df)                                            #Use local calculations (e.g., sunrise/sunset times based on lat/lng and date) to fill missing environmental data
    
    # Drop unnamed numeric-indexed columns
    df = df.loc[:, ~df.columns.astype(str).str.match(r'^\d+$')]                 # Drop unnamed numeric-indexed columns
    

    return df

# =============================================================================
# Generate a heatmap 
# =============================================================================
def generate_hourly_heatmap(data, filename=None):
    """Generate a heatmap to visualize the density of accidents over time and location

    Args:
        City Traffic Accidents

    Saves map to display in app:
        pandas DataFrame
    """
    if filename is None:
        filename = str(PROJECT_ROOT / "data" / "maps" / "interactive_traffic_map.html")
    
    #Automatically create the directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    #Remove existing file if it exists to ensure overwrite
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed existing file: {filename}")

    #Prepare the hourly data
    hourly_data = []
    for hour in range(24):
        subset = data[data['hour'] == hour]
        points = subset[['Start_Lat', 'Start_Lng']].dropna().values.tolist()
        hourly_data.append(points)
    
    #Create the map
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
    HeatMapWithTime(
        hourly_data, 
        index=[f"{h}:00" for h in range(24)],
        radius=10, 
        auto_play=True, 
        max_opacity=0.8
    ).add_to(m)
    
    #Save
    m.save(filename)
    print(f"Map successfully saved to: {filename} (overwrote existing file)")
    return m

# =============================================================================
# Generate a severity map 
# =============================================================================
def generate_accident_map(data, filename=None):

    """Generate a map to visualize the locations of accidents and their severity

    Args:
        City Traffic Accidents

    Saves map to display in app:
        pandas DataFrame
    """
    if filename is None:
        filename = str(PROJECT_ROOT / "data" / "maps" / "accident_map.html")
    
    #Automatically create the directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    #Remove existing file if it exists to ensure overwrite
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed existing file: {filename}")

    #Prepare the data data
    df_sample = data[['Start_Lat', 'Start_Lng']].dropna()
    
    #Create the map United State Only
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
    FastMarkerCluster(data=df_sample.values.tolist()).add_to(m)
    
    #Save
    m.save(filename)
    print(f"Map successfully saved to: {filename} (overwrote existing file)")
    return m

# =============================================================================
# Calculate wind chill based on rules
# =============================================================================
def calculate_wind_chill(temp, speed, chill):
    """Calculate wind chill based on rules:
    - If wind_chill is not null, keep current value
    - If wind_chill is null and temp > 50°F, use temperature value
    - If wind_chill is null and temp <= 50°F, calculate using formula if wind_speed > 3 mph
    
    Formula: WindChill = 35.74 + 0.6215T - 35.75(V^0.16) + 0.4275T(V^0.16)
    T = Air Temperature (°F)
    V = Wind Speed (mph)
    
    Args:
        temp: Temperature(F) - scalar or Series
        speed: Wind_Speed(mph) - scalar or Series
        chill: Wind_Chill(F) - scalar or Series
    
    Returns:
        Calculated wind chill value(s)
    """
    # For each row, apply the logic
    if pd.isna(chill):
        # If wind_chill is null
        if temp > 50:
            # If temp above 50°F, use temperature as wind chill
            return temp
        elif speed > 3:
            # If temp <= 50°F and wind speed > 3 mph, calculate formula
            return 35.74 + (0.6215 * temp) - (35.75 * (speed**0.16)) + (0.4275 * temp * (speed**0.16))
        else:
            # If temp <= 50°F and wind speed <= 3 mph, use temperature
            return temp
    else:
        # Keep existing wind chill value if not null
        return chill

# =============================================================================
# Fill missing Airport_Code and Zipcode 
# =============================================================================
def airport_code_to_zip(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Airport_Code and Zipcode values using bidirectional lookup.
    
    - For rows where Airport_Code is null, uses Zipcode to lookup the nearest airport code
    - For rows where Zipcode is null, uses Airport_Code to lookup the nearest zip code
    
    Uses the zip_to_airport_lookup.csv file for mapping.

    Args:
        df: pandas DataFrame with 'Airport_Code' and 'Zipcode' columns

    Returns:
        DataFrame with filled Airport_Code and Zipcode values
    """
    try:
        lookup_file_path = PROJECT_ROOT / "data" / "lookup_tables" / "airport_lookup.csv"
        if not lookup_file_path.exists():
            print(f"Lookup file not found: {lookup_file_path}. Skipping airport/zip code fill.")
            return df
        
        lookup = pd.read_csv(lookup_file_path, dtype=str)
        
        # Create both forward and reverse mappings
        zip_to_airport = dict(zip(lookup["zip_code"], lookup["nearest_airport_iata"]))
        
        # Fill missing Airport_Code using Zipcode lookup
        ac_mask = df['Airport_Code'].isna()
        filled_ac = ac_mask.sum()
        df.loc[ac_mask, 'Airport_Code'] = df.loc[ac_mask, 'Zipcode'].map(zip_to_airport)
        
    except Exception as e:
        print(f"Error filling Airport_Code/Zipcode: {e}. Continuing without this step.")
    
    return df

# =============================================================================
# Fill in Street/City/Zip/Timezone
# =============================================================================
def fill_geographic_data(df: pd.DataFrame) -> pd.DataFrame:
    search = SearchEngine()
    
    # Identify rows where Street is missing (along with City/Zip/Timezone)
    mask = (df['Street'].isna()) | (df['Street'].astype(str).str.lower().isin(['none', 'nan', 'missing'])) | \
           (df['City'].isna()) | (df['Timezone'].isna())

    def repair_geo_details(row):
        result = search.by_coordinates(row['Start_Lat'], row['Start_Lng'], radius=50, returns=1)
        if result:
            res = result[0]
            # Use City Center (Major City) as a placeholder for Street if Street is missing
            return pd.Series({
                'Zipcode': res.zipcode,
                'City': res.major_city.lower(),
                'County': res.county.lower(),
                'Timezone': f"us/{res.timezone.lower()}" if res.timezone else row['Timezone'],
            })
        return pd.Series({'Zipcode': '00000', 'City': 'unknown', 'County': 'unknown', 'Timezone': 'missing'})

    # Apply the logic
    df.loc[mask, ['Zipcode', 'City', 'County', 'Timezone']] = df[mask].apply(repair_geo_details, axis=1)
    
    return df

# =============================================================================
# Create an inverse mapping for fast lookup
# =============================================================================
def add_census_regions(df):
    regions = {
        'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
        'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MS', 'NE', 'ND', 'OH', 'SD', 'WI'],
        'South': ['AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV'],
        'West': ['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
    }
    # Create an inverse mapping for fast lookup
    state_to_region = {state: region for region, states in regions.items() for state in states}
    
    df['Region'] = df['State'].str.upper().map(state_to_region).fillna('Other')
    return df

# =============================================================================
# Finds 10 high-density regions
# =============================================================================
def create_cluster_regions(df, n_clusters=10):
    # Use coordinates to find 10 high-density regions
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Geo_Cluster'] = kmeans.fit_predict(df[['Start_Lat', 'Start_Lng']])
    return df

# =============================================================================
# Calculate the 'Hotspot' for each region
# =============================================================================
def add_intra_region_distances(df, cluster_col='Geo_Cluster'):
    """
    Calculates the distance from each accident to the center 
    of its assigned cluster/region.
    """
    #Calculate the 'Hotspot' (Centroid) for each region
    centroids = df.groupby(cluster_col)[['Start_Lat', 'Start_Lng']].mean().reset_index()
    centroids.columns = [cluster_col, 'Centroid_Lat', 'Centroid_Lng']
    
    #Merge these centers back into the main dataframe
    df = df.merge(centroids, on=cluster_col, how='left')
    
    # Calculate distance (Haversine or simple Euclidean approximation)
    # A degree is roughly 69 miles
    lat_diff = (df['Start_Lat'] - df['Centroid_Lat']) ** 2
    lng_diff = (df['Start_Lng'] - df['Centroid_Lng']) ** 2
    
    df['dist_from_reg_hotspot'] = np.sqrt(lat_diff + lng_diff) * 69
    
    # Drop the temporary centroid columns to keep the DF clean
    df.drop(columns=['Centroid_Lat', 'Centroid_Lng'], inplace=True)
    
    return df

# =============================================================================
# FAST environmental data: Uses local Astral + regional median filling
# =============================================================================
def fast_environmental_data(df):
    """
    FAST environmental data: Uses local Astral + regional median filling.
    Fills ALL weather with cluster-based medians (instant, deterministic).
    """
    
    # ===== SUN DATA (Local Astral calculations) =====
    mask = df['Sunrise_Sunset'].isna()
    if mask.any():
        print(f"  Calculating sun data for {mask.sum()} rows...")
        
        missing_data = df[mask].copy()
        missing_data['date'] = missing_data['Start_Time'].dt.date
        unique_days = missing_data[['date', 'Start_Lat', 'Start_Lng']].drop_duplicates()

        results_map = {}
        for _, row in unique_days.iterrows():
            try:
                obs = Observer(latitude=row['Start_Lat'], longitude=row['Start_Lng'])
                s = sun(obs, date=row['date'])
                results_map[(row['date'], row['Start_Lat'], row['Start_Lng'])] = s
            except:
                continue

        def apply_sun_logic(row):
            key = (row['Start_Time'].date(), row['Start_Lat'], row['Start_Lng'])
            if key not in results_map:
                return pd.Series({
                    'Sunrise_Sunset': np.nan,
                    'Civil_Twilight': np.nan,
                    'Nautical_Twilight': np.nan,
                    'Astronomical_Twilight': np.nan
                })
            
            s = results_map[key]
            acc_t = row['Start_Time'].tz_localize('UTC') if row['Start_Time'].tzinfo is None else row['Start_Time']
            
            return pd.Series({
                'Sunrise_Sunset': 'Day' if s['sunrise'] < acc_t < s['sunset'] else 'Night',
                'Civil_Twilight': 'Day' if s['dawn'] < acc_t < s['dusk'] else 'Night',
                'Nautical_Twilight': 'Day' if s['dawn'] < acc_t < s['dusk'] else 'Night',
                'Astronomical_Twilight': 'Day' if s['sunrise'] < acc_t < s['sunset'] else 'Night'
            })

        sun_updates = df[mask].apply(apply_sun_logic, axis=1)
        df.loc[mask, sun_updates.columns] = sun_updates
    
    # ===== WEATHER DATA (Regional median filling) =====
    numeric_weather_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 
                            'Wind_Speed(mph)', 'Precipitation(in)']
    categorical_weather_cols = ['Wind_Direction', 'Weather_Condition']
    
    print(f"  Filling weather with regional medians...")
    
    # Numeric columns: use median
    for col in numeric_weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df.groupby('Geo_Cluster')[col].transform('median'))
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns: use mode (most common value)
    for col in categorical_weather_cols:
        if col in df.columns:
            # Mode-fill by cluster
            def fill_mode(group):
                if len(group) > 0:
                    mode_val = group.mode()[0] if len(group.mode()) > 0 else 'Unknown'
                    return group.fillna(mode_val)
                return group
            
            df[col] = df.groupby('Geo_Cluster')[col].transform(fill_mode)
            # Global fallback
            if df[col].isna().any():
                global_mode = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col] = df[col].fillna(global_mode)
    
    return df

# =============================================================================
# Identify top 20 words in the Descriptor column and features for each
# =============================================================================
def descriptor_word_count(df):
    """Add features for the top 20 words in the Description column."""
    if 'Description' in df.columns:
        # Combine all descriptions, lowercase them, and find all words
        all_text = ' '.join(df['Description'].dropna()).lower()
        words = re.findall(r'\w+', all_text)
        
        # Filter out stop words
        stop_words = {'on', 'at', 'the', 'and', 'of', 'to', 'in', 'from', 'near', 'i', 'rd', 'st', 'ave', 'blvd', 'hwy', 'highway', 'street', 'road', 'due', 'us', 'ca', 'la', 'ny', 'tx', 'fl', 'il', 'wa', 'pa', 'oh', 'mi', 'ga', 'nc', 'nj', 'va', 'ma', 'az', 'co', 'nv', 'with', 'dr', 's', 'n', 'e', 'w', '95'}
        keywords = Counter([w for w in words if w not in stop_words])
        
        # Get top 20 words
        top_20_words = [word for word, count in keywords.most_common(20)]
        
        # Create columns for each top word - count occurrences in each description
        for word in top_20_words:
            df[f'word_{word}'] = df['Description'].fillna('').apply(lambda x: x.lower().count(word))
        
        # Drop the Description column
        df = df.drop(columns=['Description'], errors='ignore')
    
    return df

# =============================================================================
# HINT 3: Weather Feature Processing
# =============================================================================
def process_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weather is a major factor in accident severity.

    Missing values in weather columns are NOT random — they often mean:
    - Weather station was offline
    - Data wasn't available at the time of the accident
    - The weather API didn't return data for that location

    Strategy: Create a "weather_data_available" flag, then impute or drop.

    Key weather features:
    - Temperature(F): Freezing conditions are dangerous
    - Visibility(mi): Low visibility = more severe accidents
    - Precipitation(in): Rain/snow increases severity
    - Weather_Condition: Categorical (Clear, Rain, Snow, Fog, etc.)
    """
    weather_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

    # Create Data Availability Flag
    # Returns 1 if ANY data for these columns, 0 if all are NaN
    df['weather_data_available'] = df[weather_cols].notna().any(axis=1).astype(int)

    # Temperature Logic (Freezing check)
    if 'Temperature(F)' in df.columns:
        # Binary flag for freezing (dangerous for road traction)
        df['is_freezing'] = (df['Temperature(F)'] <= 32).astype(int)

    # Visibility Logic (Low visibility check)
    if 'Visibility(mi)' in df.columns:
        # Binary flag for hazardous visibility (under 2 miles)
        df['low_visibility_severity'] = (df['Visibility(mi)'] < 2).astype(int)

    # Precipitation Logic (Rain/Snow check)
    if 'Precipitation(in)' in df.columns:
        # Flag if there is any measurable precipitation
        df['has_precipitation'] = (df['Precipitation(in)'] > 0).astype(int)

    # Categorize the String Conditions
    if 'Weather_Condition' in df.columns:
        df['weather_cluster'] = df['Weather_Condition'].apply(categorize_weather)

    # One-Hot Encode 'weather_cluster' and drop the string version immediately
    # Use prefix to keep columns organized (e.g., weather_cluster_rain)
    df = df.join(pd.get_dummies(df.pop('weather_cluster'), prefix='weather_cluster', dtype=int))

    df = df.drop(columns=['Weather_Condition'], errors='ignore')    #Drop the Weather_Condition column after processing

    return df

# =============================================================================
# Weather Feature Processing - Categorization Logic
# =============================================================================
def categorize_weather(condition) -> str:
    """Group detailed weather conditions into broader categories."""
    if pd.isna(condition):
        return 'unknown'

    condition = str(condition).lower()

    # Priority order: Storms and Snow/Ice are higher risk than Rain
    if any(w in condition for w in ['clear', 'fair']):
        return 'clear'
    elif any(w in condition for w in ['cloud', 'overcast']):
        return 'cloudy'
    elif any(w in condition for w in ['rain', 'drizzle', 'shower']):
        return 'rain'
    elif any(w in condition for w in ['snow', 'sleet', 'ice', 'wintry']):
        return 'snow_ice'
    elif any(w in condition for w in ['fog', 'mist', 'haze', 'smoke']):
        return 'low_visibility'
    elif any(w in condition for w in ['thunder', 'storm']):
        return 'storm'
    else:
        return 'other'

# =============================================================================
# HINT 4: Road Feature Processing
# =============================================================================
def process_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The dataset has 13 boolean road feature columns.

    These are already binary (True/False) and very useful for ML models.
    Consider creating aggregate features:
    - total_road_features: count of road features at the accident location
    - has_traffic_control: any of traffic signal, stop, give way, etc.
    """
    road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                     'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                     'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

    existing = [f for f in road_features if f in df.columns]

    # Total road features present
    df['n_road_features'] = df[existing].sum(axis=1)

    # Traffic control present
    control_features = ['Traffic_Signal', 'Stop', 'Give_Way', 'Traffic_Calming']
    existing_control = [f for f in control_features if f in df.columns]
    df['has_traffic_control'] = df[existing_control].any(axis=1).astype(int)

    return df

# =============================================================================
# HINT 5: Handling Severity Class Imbalance
# =============================================================================
def analyze_severity_distribution(df: pd.DataFrame):
    """
    Severity distribution is heavily imbalanced:
    - Severity 1: ~1-2% (very rare)
    - Severity 2: ~80% (dominant — this is your biggest challenge)
    - Severity 3: ~12-15%
    - Severity 4: ~5-8%

    This is a MAJOR challenge. If you just predict class 2 for everything,
    you'll get ~80% accuracy but your model is COMPLETELY USELESS.
    Weighted F1 is the real evaluation metric, not accuracy.

    Strategies:
    1. Class weights: Give higher weight to minority classes
       - sklearn: class_weight='balanced'
       - TensorFlow/Keras: class_weight parameter in model.fit()
    2. SMOTE or oversampling for minority classes
    3. Undersampling the majority class (Severity 2)
    4. Consider binary: "severe" (3-4) vs "not severe" (1-2)
    5. Focal loss — designed for class imbalance

    For evaluation: Use weighted F1, not just accuracy.
    Weighted F1 accounts for class imbalance by weighting each class by its support.
    """
    print("Severity Distribution:")
    print(df['Severity'].value_counts().sort_index())
    print(f"\nClass ratios:")
    print(df['Severity'].value_counts(normalize=True).sort_index().round(3))

# =============================================================================
# HINT 10: Geographic Feature Engineering
# =============================================================================
def create_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Location matters for accident severity prediction.

    Feature ideas:
    1. State-level patterns (some states have more severe accidents)
    2. Urban vs. rural (can infer from city population or zip code)
    3. Latitude as a proxy for climate (northern = more ice/snow)
    4. Distance from nearest airport (proxy for traffic volume)
    5. Cluster analysis on lat/lng to find accident hotspots

    Warning: Don't use raw lat/lng as features — they're too specific
    and lead to overfitting. Instead, bin them or use for clustering.
    """
    # State-level average severity (target encoding — be careful of leakage!)
    # Only compute on training data, then apply to test

    # Latitude bins (rough climate proxy)
    if 'Start_Lat' in df.columns:
        df['lat_bin'] = pd.cut(df['Start_Lat'], bins=10, labels=False)

    return df

# =============================================================================
# Find top 5 zipcode in each region and group the rest into "other" category
# =============================================================================
def create_zipcode_features(df: pd.DataFrame) -> pd.DataFrame:
    # Defines your existing region columns from add_census_regions
    region_cols = ['region_Midwest', 'region_Northeast', 'region_South', 'region_West', 'region_Other']
    
    # This finds which region column is 1 for every row
    temp_region = df[region_cols].idxmax(axis=1)

    #For each region, find the top 5 zipcodes
    is_top_5 = df.groupby(temp_region)['Zipcode'].transform(
        lambda x: x.isin(x.value_counts().nlargest(5).index)
    )

    #Create the new labels: "Region_Zip" if in top 5, else "Region_other"
    df['Zip_Grouped'] = temp_region + "_" + df['Zipcode'].astype(str)
    df.loc[~is_top_5, 'Zip_Grouped'] = temp_region + "_other"

    #One-Hot Encode the original
    df = pd.get_dummies(df, columns=['Zip_Grouped'], prefix='Zip')
    
    return df

# ==================================================================================================
# Find top 20 cities and counties and group the rest into "Other" category, then one-hot encode
# ==================================================================================================
def encode_top_geo_features(df, columns=['City', 'County']):
    for col in columns:
        # 1. Find the top 20 values for the current column
        top_20 = df[col].value_counts().nlargest(20).index
        
        # 2. Rename anything not in the top 20 to 'Other'
        df[col] = df[col].where(df[col].isin(top_20), 'Other')
    
    # 3. One-Hot Encode both columns at once
    # prefix=['City', 'Cty'] keeps the new column names clean
    df = pd.get_dummies(df, columns=columns, prefix=['City', 'Cty'])
    
    return df

# ==================================================================================================
# A WAY TO COMBINE WEATHER CONDITIONS INTO A SINGLE RISK SCORE
# ==================================================================================================
def dangerous_conditions_score(df): 
    df = df.copy()
    
    # Apply the scoring function to every row
    print("Engineering DangerousScore...")
    df['DangerousScore'] = df.apply(calculate_dangerous_score, axis=1)
    
    # NOW drop the columns once the scoring is done
    cols_to_drop = [
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 
        'Weather_Condition', 'Sunrise_Sunset', 'Astronomical_Twilight'
    ]
    
    # Only drop columns that actually exist in the dataframe
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    
    return df

def calculate_dangerous_score(row): 
    score = 0
    # Get weather text and handle potential missing values
    weather = str(row.get('Weather_Condition', '')).strip().lower()

    # Visibility
    visibility = row.get('Visibility(mi)')
    if pd.notna(visibility):
        if visibility < 1: score += 3
        elif visibility < 3: score += 2
        elif visibility < 5: score += 1

    # Precipitation
    precip = row.get('Precipitation(in)')
    if pd.notna(precip):
        if precip > 0.3: score += 2
        elif precip > 0: score += 1
    
    # Temperature / Wind Chill
    temp = row.get('Temperature(F)')
    wind_chill = row.get('Wind_Chill(F)')
    effective_temp = wind_chill if pd.notna(wind_chill) else temp

    if pd.notna(effective_temp):
        if effective_temp < 32: score += 2   # freezing
        elif effective_temp > 100: score += 1 # extreme heat

    # Wind speed
    wind = row.get('Wind_Speed(mph)')
    if pd.notna(wind):
        if wind > 40: score += 2
        elif wind > 25: score += 1

    # Darkness
    if row.get('Sunrise_Sunset') == 'Night': score += 1
    if row.get('Astronomical_Twilight') == 'Night': score += 1

    # Weather text categories
    if any(term in weather for term in ['tornado', 'thunderstorm', 'hail', 'squalls']):
        score += 3
    elif any(term in weather for term in ['freezing', 'sleet', 'ice', 'wintry']):
        score += 3
    elif any(term in weather for term in ['fog', 'mist', 'haze', 'smoke']):
        score += 2
    elif any(term in weather for term in ['rain', 'snow', 'drizzle']):
        score += 1

    return score

def engineer_road_features(df):
    """
    Groups individual boolean road features into a single 'n_road_features' count
    and a binary 'has_traffic_control' flag.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # Define all possible road features
    road_features = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
        'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
        'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
    ]
    
    #Filter for features that actually exist in the current dataframe
    existing = [f for f in road_features if f in df.columns]
    print(f"Aggregating {len(existing)} road features...")

    #Total road features present (Sum of booleans)
    df['n_road_features'] = df[existing].sum(axis=1)

    #Traffic control presence (Specific Subset)
    control_features = ['Traffic_Signal', 'Stop', 'Give_Way', 'Traffic_Calming']
    existing_control = [f for f in control_features if f in df.columns]
    
    # Create binary flag (1 if ANY control features are true, else 0)
    df['has_traffic_control'] = df[existing_control].any(axis=1).astype(int)

    # Remove original individual road features to reduce dimensionality
    # This helps models like Random Forest focus on the aggregated signal
    df = df.drop(columns=existing)
    
    print(f"Feature engineering complete. New columns: ['n_road_features', 'has_traffic_control']")
    return df