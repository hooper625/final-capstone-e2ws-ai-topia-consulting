import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

#Create interactive map
import folium
from folium.plugins import FastMarkerCluster

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# Add project root to sys.path
root_path = Path.cwd().parent 
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

#Import
from pipelines.data_pipeline import load_raw_data, clean_data, accident_engineer_features, complaints_engineer_features, save_processed_data, drop_low_variance_columns
from pipelines.data_pipeline import generate_hourly_heatmap, generate_accident_map # functions to create maps

#Load the City Traffic Accident Database
df_City_Traffic = load_raw_data("city_traffic_accidents.csv")
df_Complaints= load_raw_data("urbanpulse_311_complaints.csv")

########################################################### EDA City Traffic Accident ########################################################################## 
#check missing count
missing_count = df_City_Traffic.isnull().sum()
print(missing_count)

df_City_Traffic.head()

df_City_Traffic.columns.tolist()
df_City_Traffic.describe(include='object')

# Define your target variable
TARGET = 'Severity'  

# Basic statistics of target
print(f"Target Variable: {TARGET}")
print(f"\nBasic Statistics:")
print(df_City_Traffic[TARGET].describe())

#check descriptive stats for numerical features
df_City_Traffic['Visibility(mi)'].describe()
df_City_Traffic['Precipitation(in)'].describe()
df_City_Traffic['Weather_Condition'].describe()
df_City_Traffic['Temperature(F)'].describe()

# Visualize Severity by different features
fig, axes = plt.subplots(2,2,figsize = (12,10))

# Plot 1: Average Severity Based on Visibility(mi)
severity_avg_by_visibility = df_City_Traffic.groupby('Visibility(mi)')[TARGET].mean()
colors = {0: "#E1897F", 1: "#79813E"}
# Plot 1: Average Severity vs. Visibility on axes[0,0]
axes[0,0].scatter(
    severity_avg_by_visibility.index, 
    severity_avg_by_visibility.values, 
    color='#E74C3C',
    s=15, 
    alpha=0.8
)

axes[0,0].set_xlabel('Visibility (mi)')
axes[0,0].set_ylabel('Average Severity')
axes[0,0].set_title('Average Severity vs. Visibility')

# Plot 2: Average Severity Based on Precipitation
severity_avg_by_precip = df_City_Traffic.groupby('Precipitation(in)')[TARGET].mean()
axes[0,1].scatter(
    severity_avg_by_precip.index, 
    severity_avg_by_precip.values, 
    color='#3498DB', 
    s=15, 
    alpha=0.8
)
axes[0,1].set_xlabel('Precipitation (in)')
axes[0,1].set_ylabel('Average Severity')
axes[0,1].set_title('Average Severity vs. Precipitation')

# Remove the categorical labels since we are now using the continuous scale
axes[0,1].set_xticks(axes[0,1].get_xticks())

#Plot 3: Severity Based on Weather_Condition
top_weather = df_City_Traffic['Weather_Condition'].value_counts().head(20)
axes[1,0].bar(
    top_weather.index, 
    top_weather.values, 
    color='#F1C40F', 
    edgecolor='black',
    alpha=0.8
)
axes[1,0].set_xlabel('Weather Condition')
axes[1,0].set_ylabel('Number of Accidents')
axes[1,0].set_title('Top 20 Weather Conditions')
axes[1,0].tick_params(axis='x', rotation=45, labelsize=9)


#Plot 4: Severity trends for Temperature(F)
severity_avg_by_temp = df_City_Traffic.groupby('Temperature(F)')[TARGET].mean()
axes[1,1].scatter(
    severity_avg_by_temp.index, 
    severity_avg_by_temp.values, 
    color='#E67E22', # Orange color to represent temperature
    s=15, 
    alpha=0.6
)
axes[1,1].set_xlabel('Temperature (F)')
axes[1,1].set_ylabel('Average Severity')
axes[1,1].set_title('Average Severity vs. Temperature')
axes[1,1].axvline(x=32, color='blue', linestyle='--', alpha=0.3, label='Freezing (32°F)')

plt.tight_layout()
plt.show()

#Clean and engineer features for the City Traffic Accident Database
df_City_Traffic = clean_data(df_City_Traffic)                       #Clean the data (handle missing values, convert data types, etc.)
df_City_Traffic = accident_engineer_features(df_City_Traffic)       #Engineer features specific to traffic accidents (e.g., severity, weather conditions, etc.)

#Generate the heatmap and accident map for City Traffic Accident
generate_hourly_heatmap(df_City_Traffic)                            #Generate a heatmap to visualize the density of accidents over time and location
generate_accident_map(df_City_Traffic)                              #Generate a map to visualize the locations of

# Verify it worked
print(f"Total records loaded: {len(df_City_Traffic)}")
#print(f"Total records loaded: {len(df_Complaints)}")
print(df_City_Traffic.head())
df_City_Traffic.columns.tolist()
df_City_Traffic['Geo_Cluster'].describe()

# Visualize Severity by different features
fig, axes = plt.subplots(2,2,figsize = (12,10))

# Plot 1: Average Severity Based on Region
region_cols = ['region_Midwest', 'region_Northeast', 'region_South', 'region_West', 'region_Other']
region_stats = []
for col in region_cols:
    region_data = df_City_Traffic[df_City_Traffic[col] == 1]
    region_stats.append({
        'Region': col.replace('region_', ''),
        'Accident_Count': len(region_data),
        'Avg_Severity': region_data[TARGET].mean()
    })
stats_df = pd.DataFrame(region_stats)
axes[0,0].scatter(
    stats_df['Accident_Count'], 
    stats_df['Avg_Severity'], 
    color='#E74C3C', 
    s=200,     
    edgecolor='black',
    alpha=0.9
)
for i, row in stats_df.iterrows():
    axes[0,0].text(
        row['Accident_Count'], 
        row['Avg_Severity'] + 0.005,
        row['Region'], 
        fontsize=10, 
        ha='center',
        fontweight='bold'
    )
axes[0,0].set_xlabel('Total Accident Count')
axes[0,0].set_ylabel('Average Severity')
axes[0,0].set_title('Region Analysis: Volume vs. Severity')
axes[0,0].grid(True, linestyle='--', alpha=0.5)


# Plot 2: Average Severity Based on Geo_Cluster
cluster_severity = df_City_Traffic.groupby('Geo_Cluster')[TARGET].mean()
axes[0,1].scatter(
    cluster_severity.index, 
    cluster_severity.values, 
    color='#9B59B6', 
    s=100,          
    edgecolor='black',
    zorder=3
)
axes[0,1].set_xlabel('Geo Cluster ID')
axes[0,1].set_ylabel('Average Severity')
axes[0,1].set_title('Severity Risk by Geographic Cluster')
axes[0,1].set_xticks(range(10))
overall_mean = df_City_Traffic[TARGET].mean()
axes[0,1].axhline(overall_mean, color='red', linestyle='--', alpha=0.5, label='National Avg')
axes[0,1].legend()
axes[0,1].set_xticks(axes[0,1].get_xticks())

# Plot 3: Severity Based on Weather_Condition
weather_cols = [
    'weather_cluster_clear', 'weather_cluster_cloudy', 
    'weather_cluster_low_visibility', 'weather_cluster_other', 
    'weather_cluster_rain', 'weather_cluster_snow_ice', 'weather_cluster_storm'
]
weather_counts = df_City_Traffic[weather_cols].sum().sort_values(ascending=False)
clean_labels = [col.replace('weather_cluster_', '').replace('_', ' ').title() for col in weather_counts.index]
axes[1,0].bar(
    clean_labels, 
    weather_counts.values, 
    color='#F1C40F', 
    edgecolor='black',
    alpha=0.8
)
axes[1,0].set_xlabel('Weather Group')
axes[1,0].set_ylabel('Number of Accidents')
axes[1,0].set_title('Accidents by Weather Category')

# Plot 4: Severity trends for Distance from Regional Hotspot
df_City_Traffic['dist_rounded'] = df_City_Traffic['dist_from_reg_hotspot'].round(1)
severity_avg_by_dist = df_City_Traffic.groupby('dist_rounded')[TARGET].mean()

# 2. Plot on the desired axis
axes[1,1].scatter(
    severity_avg_by_dist.index, 
    severity_avg_by_dist.values, 
    color='#16A085', # Sea green color
    s=15, 
    alpha=0.6
)

# 3. Formatting
axes[1,1].set_xlabel('Distance from Regional Hotspot')
axes[1,1].set_ylabel('Average Severity')
axes[1,1].set_title('Severity Trend by Hotspot Proximity')

plt.tight_layout()
plt.show()

# Distribution of target variable
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df_City_Traffic[TARGET].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel(TARGET)
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Distribution of {TARGET}')

# Box plot
axes[1].boxplot(df_City_Traffic[TARGET].dropna())
axes[1].set_ylabel(TARGET)
axes[1].set_title(f'Box Plot of {TARGET}')

plt.tight_layout()
plt.show()

df_City_Traffic.head()

# 1. Use df.select_dtypes(include=[np.number]) to get numerical columns
numerical_df = df_City_Traffic.select_dtypes(include=[np.number])

# 2. Get the column names as a list with .columns.tolist()
numerical_cols= numerical_df.columns.tolist()

# 4. Print the count and list of numerical features
print(f'Count: {len(numerical_cols)}')
print(f'Features: {numerical_cols}')
385/15:
if len(numerical_cols) > 0:
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes

    for i, col in enumerate(numerical_cols):
        axes[i].hist(df_City_Traffic[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel('')

    # Hide empty subplots
    for j in range(len(numerical_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
else:
    print("No numerical features found (besides target).")
    
df_City_Traffic = drop_low_variance_columns(df_City_Traffic)

# 1. Use df.select_dtypes(include=[np.number]) to get numerical columns
numerical_df = df_City_Traffic.select_dtypes(include=[np.number])

# 2. Get the column names as a list with .columns.tolist()
numerical_cols= numerical_df.columns.tolist()

# 4. Print the count and list of numerical features
print(f'Count: {len(numerical_cols)}')
print(f'Features: {numerical_cols}')

if len(numerical_cols) > 0:
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes

    for i, col in enumerate(numerical_cols):
        axes[i].hist(df_City_Traffic[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel('')

    # Hide empty subplots
    for j in range(len(numerical_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
else:
    print("No numerical features found (besides target).")
df_City_Traffic.head()

missing_count = df_City_Traffic.isnull().sum()

print(missing_count)

df_City_Traffic = df_City_Traffic.dropna(axis=1) 
missing_count = df_City_Traffic.isnull().sum()

print(missing_count)
df_City_Traffic.info()
df_City_Traffic.info()
save_processed_data(df_City_Traffic, "city_traffic_processed.csv")


########################################################### EDA 311 Complaints Database ########################################################################## 

#Clean and engineer features for the 311 Complaints Database
df_Complaints= clean_data(df_Complaints)                            #Clean the data (handle missing values, convert data types, etc.)
df_Complaints = complaints_engineer_features(df_Complaints)         #Engineer features specific to 311 complaints (e.g., complaint type, resolution time, etc.)