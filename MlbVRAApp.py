import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, atan2, degrees
import pandas as pd

# Load the data (replace with the actual path to your CSV)
statcast_data = pd.read_csv('2024mlbvradata.csv')

# Streamlit app layout
st.title('Pitch Data Analysis')
st.sidebar.header('Select Player')

# Create a list of unique player names for the dropdown
player_names = statcast_data['player_name'].unique()
player_selection = st.sidebar.selectbox('Select Player', player_names)

# Filter the data based on selected player
filtered_data = statcast_data[statcast_data['player_name'] == player_selection]

# Define pitch colors
pitch_colors = {
    ## Fastballs ##
    'FF': {'colour': '#FF007D', 'name': '4-Seam Fastball'},
    'FA': {'colour': '#FF007D', 'name': 'Fastball'},
    'SI': {'colour': '#98165D', 'name': 'Sinker'},
    'FC': {'colour': '#BE5FA0', 'name': 'Cutter'},

    ## Offspeed ##
    'CH': {'colour': '#F79E70', 'name': 'Changeup'},
    'FS': {'colour': '#FE6100', 'name': 'Splitter'},
    'SC': {'colour': '#F08223', 'name': 'Screwball'},
    'FO': {'colour': '#FFB000', 'name': 'Forkball'},

    ## Sliders ##
    'SL': {'colour': '#67E18D', 'name': 'Slider'},
    'ST': {'colour': '#1BB999', 'name': 'Sweeper'},
    'SV': {'colour': '#376748', 'name': 'Slurve'},

    ## Curveballs ##
    'KC': {'colour': '#311D8B', 'name': 'Knuckle Curve'},
    'CU': {'colour': '#3025CE', 'name': 'Curveball'},
    'CS': {'colour': '#274BFC', 'name': 'Slow Curve'},
    'EP': {'colour': '#648FFF', 'name': 'Eephus'},

    ## Others ##
    'KN': {'colour': '#867A08', 'name': 'Knuckleball'},
    'PO': {'colour': '#472C30', 'name': 'Pitch Out'},
    'UN': {'colour': '#9C8975', 'name': 'Unknown'},
}

# Drop NA for specific columns
filtered_data.dropna(subset=['vy0', 'release_extension'], inplace=True)

# Calculate VAA
def calculate_vaa(row):
    yf = 17 / 12  # Home plate distance in feet, converted to inches
    ay = row['ay']  # Acceleration in y-dimension for the current pitch
    vy0 = row['vy0']  # Velocity in y-dimension at y=50 feet for the current pitch
    vz0 = row['vz0']  # Velocity in z-dimension at y=50 feet for the current pitch
    az = row['az']  # Acceleration in z-dimension for the current pitch

    vy_f = -sqrt(vy0 ** 2 - (2 * ay * (50 - yf)))
    t = (vy_f - vy0) / ay
    vz_f = vz0 + (az * t)
    vaa_rad = atan2(vz_f, vy_f)
    vaa_deg = (180 + degrees(vaa_rad)) * -1

    return vaa_deg

filtered_data['VAA'] = filtered_data.apply(calculate_vaa, axis=1)

# Calculate VRA and HRA with release height and plate height adjustments
def calculate_VRA(vy0, ay, release_extension, vz0, az, release_pos_z, plate_z):
    vy_s = -np.sqrt(vy0 ** 2 - 2 * ay * (60.5 - release_extension - 50))
    t_s = (vy_s - vy0) / ay
    vz_s = vz0 - az * t_s

    vertical_movement = vz_s + (release_pos_z - plate_z)
    VRA = -np.arctan(vertical_movement / vy_s) * (180 / np.pi)
    return VRA

def calculate_HRA(vy0, ay, release_extension, vx0, ax, release_pos_x):
    vy_s = -np.sqrt(vy0 ** 2 - 2 * ay * (60.5 - release_extension - 50))
    t_s = (vy_s - vy0) / ay
    vx_s = vx0 - ax * t_s

    horizontal_movement = vx_s + release_pos_x
    HRA = -np.arctan(horizontal_movement / vy_s) * (180 / np.pi)
    return HRA

filtered_data['VRA'] = filtered_data.apply(lambda x: calculate_VRA(
    x['vy0'], x['ay'], x['release_extension'], x['vz0'], x['az'], x['release_pos_z'], x['plate_z']), axis=1)

filtered_data['HRA'] = filtered_data.apply(lambda x: calculate_HRA(
    x['vy0'], x['ay'], x['release_extension'], x['vx0'], x['ax'], x['release_pos_x']), axis=1)

# Drop any rows where VRA or HRA is NaN
filtered_data = filtered_data.dropna(subset=['VRA', 'HRA'])

# Map pitch colors based on pitch type, using only the 'colour' field
filtered_data['pitch_color'] = filtered_data['pitch_type'].map(lambda x: pitch_colors.get(x, {'colour': '#9C8975'})['colour'])

# Step 1: Plot VRA vs HRA for the selected player
plt.figure(figsize=(10, 6))

# Scatter plot with pitch colors
scatter = plt.scatter(filtered_data['HRA'], filtered_data['VRA'],
                      c=filtered_data['pitch_color'], alpha=0.6, edgecolors='w', s=80)

# Add legend
# Create a list of unique pitch types for the legend
unique_pitch_types = filtered_data['pitch_type'].unique()
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pitch_colors[pt]['colour'], markersize=10) for pt in unique_pitch_types]
plt.legend(handles, [pitch_colors[pt]['name'] for pt in unique_pitch_types], title="Pitch Types")

# Title and labels
plt.title(f'Vertical Release Angle (VRA) vs Horizontal Release Angle (HRA)\n{player_selection}')
plt.xlabel('Horizontal Release Angle (HRA) [degrees]')
plt.ylabel('Vertical Release Angle (VRA) [degrees]')
plt.grid(True)

# Set axis limits
plt.xlim(-5, 11)  # Set x-axis limits
plt.ylim(-11, 8.25)  # Set y-axis limits

# Display the plot with Streamlit
st.pyplot(plt)
