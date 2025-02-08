import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(page_title="Environmental Data Dashboard", layout="wide")


co2_emissions = pd.read_csv('/Users/nikhilkasam/Desktop/DATA IO/CO2 Emissions_Canada.csv')
global_pollution = pd.read_csv('/Users/nikhilkasam/Desktop/DATA IO/global air pollution dataset.csv')
cement_emissions = pd.read_csv('/Users/nikhilkasam/Desktop/DATA IO/1. Cement_emissions.csv', encoding='ISO-8859-1')


def plot_co2_emissions(selected_vehicle_class):
    filtered_data = co2_emissions[co2_emissions["Vehicle Class"] == selected_vehicle_class]
    scatter = go.Scatter(
        x=filtered_data["Fuel Consumption Comb (L/100 km)"],
        y=filtered_data["CO2 Emissions(g/km)"],
        mode="markers",
        marker=dict(color='blue', size=8),
        name="Data Points"
    )
    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title=f"CO2 Emissions vs. Fuel Consumption ({selected_vehicle_class})",
        xaxis_title="Fuel Consumption (L/100 km)",
        yaxis_title="CO2 Emissions (g/km)",
        height=800
    )
    return fig


def plot_air_pollution(selected_country, selected_aqi_category):
    filtered_data = global_pollution[(global_pollution['Country'] == selected_country) & (global_pollution['AQI Category'] == selected_aqi_category)]
    fig = px.bar(
        filtered_data,
        x="City",
        y="PM2.5 AQI Value",
        color="AQI Category",
        title=f"PM2.5 Air Quality Index for {selected_country} - {selected_aqi_category}",
        labels={
            "PM2.5 AQI Value": "PM2.5 AQI Value",
            "City": "City"
        }
    )
    fig.update_layout(width=1100, height=800)
    return fig


def plot_ozone_pollution():
    fig = px.box(
        global_pollution,
        x="AQI Category",
        y="Ozone AQI Value",
        color="AQI Category",
        title="Ozone AQI Value Distribution by Category",
        labels={
            "Ozone AQI Value": "Ozone AQI Value",
            "AQI Category": "AQI Category"
        }
    )
    fig.update_layout(width=1100, height=800)
    return fig


def plot_cement_emissions():
    fig = px.line(
        cement_emissions,
        x="Year",
        y="Global",
        title="Global Cement Emissions Over Time",
        labels={
            "Year": "Year",
            "Global": "CO2 Emissions from Cement (Million Tonnes)"
        },
        line_shape="linear"
    )
    fig.update_layout(height=800, width=1100)
    return fig


def plot_vehicle_characteristics_impact():
    numerical_cols = ["Engine Size(L)", "Cylinders", "Fuel Consumption City (L/100 km)",
                      "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption Comb (L/100 km)", "CO2 Emissions(g/km)"]
    
    co2_corr = co2_emissions[numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(co2_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Scatter Plot: Engine Size vs CO2 Emissions")
    fig_scatter = px.scatter(co2_emissions, x="Engine Size(L)", y="CO2 Emissions(g/km)",
                             color="Cylinders", title="Engine Size vs CO2 Emissions",
                             labels={"Engine Size(L)": "Engine Size (L)", "CO2 Emissions(g/km)": "CO2 Emissions (g/km)"})
    st.plotly_chart(fig_scatter, use_container_width=True)


tabs = st.tabs(["Graphs", "Vehicle Impact Analysis", "Analysis", ])

with tabs[0]:
    st.title("Graph Dashboard")
    vehicle_classes = co2_emissions['Vehicle Class'].unique()
    selected_vehicle_class = st.selectbox("Select Vehicle Class", vehicle_classes, key="vehicle_class_dropdown")
    st.header("CO2 Emissions vs. Fuel Consumption")
    st.plotly_chart(plot_co2_emissions(selected_vehicle_class), use_container_width=True)
    countries = global_pollution['Country'].unique()
    selected_country = st.selectbox("Select a Country", countries, key="pollution_country_dropdown")
    aqi_categories = global_pollution['AQI Category'].unique()
    selected_aqi_category = st.selectbox("Select AQI Category", aqi_categories, key="aqi_category_dropdown")
    st.header("PM2.5 Air Quality Index by Country and AQI Category")
    if selected_country in global_pollution['Country'].values:
        st.plotly_chart(plot_air_pollution(selected_country, selected_aqi_category), use_container_width=True)
    else:
        st.write("No data available for the selected country.")
    st.header("Ozone AQI Value Distribution by Category")
    st.plotly_chart(plot_ozone_pollution(), use_container_width=True)

    st.plotly_chart(plot_cement_emissions(), use_container_width=True)
   
with tabs[2]:
    st.title("Analysis")
    
    
    st.subheader("Which Vehicle Types Contribute the Most CO2 Emissions?")
    st.write("The vehicle classes with the highest average CO2 emissions are:")
    st.write("1. **Passenger Vans** (397.21 g/km)")
    st.write("2. **Cargo Vans** (361.50 g/km)")
    st.write("3. **Standard SUVs** (304.84 g/km)")
    
    
    st.subheader("How Do CO2 Emissions Differ Across Fuel Types?")
    st.write("Fuel types with the highest average CO2 emissions per km:")
    st.write("- **E (Ethanol Blends):** 275.09 g/km")
    st.write("- **Z (Electric with Range Extender):** 266.04 g/km")
    st.write("- **D (Diesel):** 237.55 g/km")
    st.write("- **X (Gasoline & Electric Hybrid):** 235.12 g/km")
    st.write("- **N (Natural Gas):** 213.00 g/km")
    

with tabs[1]:
    st.title("Vehicle Characteristics Impact on CO2 Emissions")
    st.write("Analyzing the correlation between vehicle characteristics and CO2 emissions.")
    plot_vehicle_characteristics_impact()


