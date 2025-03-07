import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import calendar
from scipy import stats

# Set page configuration
st.set_page_config(
    page_title="Bike Rental Analysis Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        width: 100%;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🚲 Bike Rental Data Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive insights into bike rental patterns. Explore how weather, 
seasonality, and time factors influence rental behavior for both casual and registered users.
""")

# Function to load and prepare data
@st.cache_data
def load_data():
    # Load data
    hour_df = pd.read_csv('https://raw.githubusercontent.com/wandagustrifa/bike-sharing/refs/heads/main/Bike-sharing-dataset/hour.csv')
    day_df = pd.read_csv('https://raw.githubusercontent.com/wandagustrifa/bike-sharing/refs/heads/main/Bike-sharing-dataset/day.csv')
    
    # Convert date column
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    
    # Create additional columns
    for df in [hour_df, day_df]:
        df['year'] = df['dteday'].dt.year
        df['month'] = df['dteday'].dt.month
        df['day'] = df['dteday'].dt.day
        df['month_name'] = df['dteday'].dt.month_name()
        df['day_name'] = df['dteday'].dt.day_name()
    
    # Map categorical columns
    season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    yr_mapping = {0: '2011', 1: '2012'}
    weathersit_mapping = {
        1: 'Clear', 
        2: 'Mist', 
        3: 'Light Precipitation',
        4: 'Heavy Precipitation'
    }
    weekday_mapping = {
        0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
        4: 'Thursday', 5: 'Friday', 6: 'Saturday'
    }
    workingday_mapping = {0: 'No', 1: 'Yes'}
    holiday_mapping = {0: 'No', 1: 'Yes'}
    
    # Apply mappings
    for df in [hour_df, day_df]:
        df['season_name'] = df['season'].map(season_mapping)
        df['year_name'] = df['yr'].map(yr_mapping)
        df['weathersit_name'] = df['weathersit'].map(weathersit_mapping)
        df['weekday_name'] = df['weekday'].map(weekday_mapping)
        df['workingday_name'] = df['workingday'].map(workingday_mapping)
        df['holiday_name'] = df['holiday'].map(holiday_mapping)
        
        # Convert normalized values to actual values
        df['temp_celsius'] = df['temp'] * 41
        df['atemp_celsius'] = df['atemp'] * 50
        df['hum_percent'] = df['hum'] * 100
        df['windspeed_kph'] = df['windspeed'] * 67
    
    return hour_df, day_df

try:
    # Load data
    hour_df, day_df = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section",
        ["Overview", "Weather Impact", "User Comparison", "Time Patterns", "Anomaly Detection"]
    )
    
    # Add data info in the sidebar
    with st.sidebar.expander("About the Data"):
        st.markdown("""
        This dataset contains bike rental counts from a bike-sharing system, along with weather and seasonal information.
        
        * **Time period**: 2011-2012
        * **Frequency**: Hourly and daily records
        * **Features**: Weather conditions, temperature, humidity, etc.
        * **Target**: Number of bike rentals (casual, registered, total)
        """)
    
    if page == "Overview":
        st.header("Overview of Bike Rental Data")
        
        # Summary metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rentals", f"{day_df['cnt'].sum():,}")
        with col2:
            st.metric("Casual Users", f"{day_df['casual'].sum():,}")
        with col3:
            st.metric("Registered Users", f"{day_df['registered'].sum():,}")
        with col4:
            st.metric("Average Daily Rentals", f"{int(day_df['cnt'].mean())}")
        
        # Yearly comparison
        st.subheader("Yearly Comparison")
        yearly_rentals = day_df.groupby('year_name')[['casual', 'registered', 'cnt']].sum().reset_index()
        
        fig_yearly = px.bar(
            yearly_rentals, 
            x='year_name', 
            y=['casual', 'registered'],
            title="Bike Rentals by Year and User Type",
            labels={'value': 'Number of Rentals', 'year_name': 'Year', 'variable': 'User Type'},
            barmode='group',
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        st.plotly_chart(fig_yearly, use_container_width=True)
        
        # Seasonal trends
        st.subheader("Seasonal Trends")
        seasonal_rentals = day_df.groupby('season_name')[['casual', 'registered', 'cnt']].mean().reset_index()
        seasonal_rentals = seasonal_rentals.sort_values(by='season_name', 
                                                       key=lambda x: x.map({'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}))
        
        fig_seasonal = px.bar(
            seasonal_rentals,
            x='season_name',
            y='cnt',
            color='season_name',
            title="Average Daily Rentals by Season",
            labels={'cnt': 'Average Rentals', 'season_name': 'Season'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Monthly trends with line chart
        st.subheader("Monthly Trends")
        monthly_rentals = day_df.groupby(['month_name', 'month', 'year_name'])[['cnt']].mean().reset_index()
        monthly_rentals = monthly_rentals.sort_values(['year_name', 'month'])
        monthly_rentals['month_year'] = monthly_rentals['month_name'] + ' ' + monthly_rentals['year_name']
        
        fig_monthly = px.line(
            monthly_rentals,
            x='month_name',
            y='cnt',
            color='year_name',
            markers=True,
            title="Average Daily Rentals by Month and Year",
            labels={'cnt': 'Average Rentals', 'month_name': 'Month', 'year_name': 'Year'},
            category_orders={"month_name": ["January", "February", "March", "April", "May", "June", 
                                           "July", "August", "September", "October", "November", "December"]}
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
    elif page == "Weather Impact":
        st.header("Impact of Weather on Bike Rentals")
        
        # Weather condition impact
        st.subheader("Weather Condition Impact")
        weather_rentals = day_df.groupby('weathersit_name')[['cnt']].mean().reset_index()
        
        fig_weather = px.bar(
            weather_rentals,
            x='weathersit_name',
            y='cnt',
            color='weathersit_name',
            title="Average Rentals by Weather Condition",
            labels={'cnt': 'Average Rentals', 'weathersit_name': 'Weather Condition'},
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig_weather, use_container_width=True)
        
        # Temperature impact with scatter plot
        st.subheader("Temperature Impact")
        
        temp_fig = px.scatter(
            hour_df,
            x='temp_celsius',
            y='cnt',
            color='season_name',
            opacity=0.6,
            title="Relationship Between Temperature and Rentals",
            labels={'temp_celsius': 'Temperature (°C)', 'cnt': 'Number of Rentals', 'season_name': 'Season'},
            trendline="ols"
        )
        st.plotly_chart(temp_fig, use_container_width=True)
        
        # Humidity and wind speed impact - side by side
        st.subheader("Humidity and Wind Speed Impact")
        col1, col2 = st.columns(2)
        
        with col1:
            hum_fig = px.scatter(
                hour_df,
                x='hum_percent',
                y='cnt',
                opacity=0.6,
                color_discrete_sequence=['blue'],
                title="Humidity vs. Rentals",
                labels={'hum_percent': 'Humidity (%)', 'cnt': 'Number of Rentals'},
                trendline="ols"
            )
            st.plotly_chart(hum_fig, use_container_width=True)
            
        with col2:
            wind_fig = px.scatter(
                hour_df,
                x='windspeed_kph',
                y='cnt',
                opacity=0.6,
                color_discrete_sequence=['green'],
                title="Wind Speed vs. Rentals",
                labels={'windspeed_kph': 'Wind Speed (km/h)', 'cnt': 'Number of Rentals'},
                trendline="ols"
            )
            st.plotly_chart(wind_fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Between Weather Factors and Rentals")
        corr_data = hour_df[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']].corr()
        
        fig_corr = px.imshow(
            corr_data,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    elif page == "User Comparison":
        st.header("Comparison Between Casual and Registered Users")
        
        # User proportion
        st.subheader("User Type Proportion")
        user_count = day_df[['casual', 'registered']].sum()
        user_data = pd.DataFrame({
            'User Type': ['Casual', 'Registered'],
            'Count': [user_count['casual'], user_count['registered']]
        })
        
        fig_pie = px.pie(
            user_data,
            values='Count',
            names='User Type',
            title="Proportion of Casual vs. Registered Users",
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Weekly patterns
        st.subheader("Weekly Patterns")
        weekday_user = hour_df.groupby('weekday_name')[['casual', 'registered']].mean().reset_index()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_user['weekday_name'] = pd.Categorical(weekday_user['weekday_name'], categories=weekday_order, ordered=True)
        weekday_user = weekday_user.sort_values('weekday_name')
        
        weekday_melt = pd.melt(
            weekday_user, 
            id_vars=['weekday_name'], 
            value_vars=['casual', 'registered'],
            var_name='User Type', 
            value_name='Average Rentals'
        )
        
        fig_weekly = px.line(
            weekday_melt,
            x='weekday_name',
            y='Average Rentals',
            color='User Type',
            markers=True,
            title="Average Rentals by Day of Week and User Type",
            labels={'weekday_name': 'Day of Week'},
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Hourly patterns
        st.subheader("Hourly Patterns")
        hourly_user = hour_df.groupby('hr')[['casual', 'registered']].mean().reset_index()
        
        hourly_melt = pd.melt(
            hourly_user, 
            id_vars=['hr'], 
            value_vars=['casual', 'registered'],
            var_name='User Type', 
            value_name='Average Rentals'
        )
        
        fig_hourly = px.line(
            hourly_melt,
            x='hr',
            y='Average Rentals',
            color='User Type',
            markers=True,
            title="Average Rentals by Hour and User Type",
            labels={'hr': 'Hour of Day'},
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Seasonal patterns
        st.subheader("Seasonal Patterns")
        seasonal_user = hour_df.groupby('season_name')[['casual', 'registered']].mean().reset_index()
        season_order = ['Spring', 'Summer', 'Fall', 'Winter']
        seasonal_user['season_name'] = pd.Categorical(seasonal_user['season_name'], categories=season_order, ordered=True)
        seasonal_user = seasonal_user.sort_values('season_name')
        
        fig_season_stack = go.Figure()
        fig_season_stack.add_trace(go.Bar(
            x=seasonal_user['season_name'],
            y=seasonal_user['casual'],
            name='Casual',
            marker_color='#ff9999'
        ))
        fig_season_stack.add_trace(go.Bar(
            x=seasonal_user['season_name'],
            y=seasonal_user['registered'],
            name='Registered',
            marker_color='#66b3ff'
        ))
        
        fig_season_stack.update_layout(
            title="Average Rentals by Season and User Type",
            xaxis_title="Season",
            yaxis_title="Average Rentals",
            barmode='group'
        )
        st.plotly_chart(fig_season_stack, use_container_width=True)
        
    elif page == "Time Patterns":
        st.header("Time-Based Rental Patterns")
        
        # Time range selector
        date_range = st.date_input(
            "Select Date Range",
            value=[day_df['dteday'].min().date(), day_df['dteday'].max().date()],
            min_value=day_df['dteday'].min().date(),
            max_value=day_df['dteday'].max().date()
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_day_df = day_df[(day_df['dteday'].dt.date >= start_date) & 
                                     (day_df['dteday'].dt.date <= end_date)]
            
            # Daily trend
            st.subheader("Daily Trend")
            
            fig_daily = px.line(
                filtered_day_df,
                x='dteday',
                y=['casual', 'registered', 'cnt'],
                title="Daily Bike Rentals Over Time",
                labels={'dteday': 'Date', 'value': 'Number of Rentals', 'variable': 'User Type'},
                color_discrete_map={
                    'casual': '#ff9999',
                    'registered': '#66b3ff',
                    'cnt': '#99ff99'
                }
            )
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Monthly aggregation
            st.subheader("Monthly Aggregation")
            
            monthly_agg = filtered_day_df.groupby(['year_name', 'month_name', 'month'])[['casual', 'registered', 'cnt']].sum().reset_index()
            monthly_agg['month_year'] = monthly_agg['month_name'] + ' ' + monthly_agg['year_name']
            monthly_agg = monthly_agg.sort_values(['year_name', 'month'])
            
            # Create datetime index for better plotting
            dates = [f"{y}-{m:02d}-01" for y, m in zip(monthly_agg['year_name'], monthly_agg['month'])]
            monthly_agg['date'] = pd.to_datetime(dates)
            monthly_agg = monthly_agg.sort_values('date')
            
            fig_monthly = px.line(
                monthly_agg,
                x='date',
                y=['casual', 'registered', 'cnt'],
                title="Monthly Bike Rentals Over Time",
                labels={'date': 'Month', 'value': 'Number of Rentals', 'variable': 'User Type'},
                color_discrete_map={
                    'casual': '#ff9999',
                    'registered': '#66b3ff',
                    'cnt': '#99ff99'
                }
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Hourly patterns by day type
            st.subheader("Hourly Patterns by Day Type")
            
            day_type = st.selectbox(
                "Select Day Type:",
                options=["All Days", "Weekday", "Weekend", "Holiday"]
            )
            
            if day_type == "All Days":
                hourly_pattern = hour_df.groupby('hr')[['casual', 'registered', 'cnt']].mean().reset_index()
            elif day_type == "Weekday":
                hourly_pattern = hour_df[hour_df['weekday'].isin([1, 2, 3, 4, 5])].groupby('hr')[['casual', 'registered', 'cnt']].mean().reset_index()
            elif day_type == "Weekend":
                hourly_pattern = hour_df[hour_df['weekday'].isin([0, 6])].groupby('hr')[['casual', 'registered', 'cnt']].mean().reset_index()
            else:  # Holiday
                hourly_pattern = hour_df[hour_df['holiday'] == 1].groupby('hr')[['casual', 'registered', 'cnt']].mean().reset_index()
            
            fig_hourly_pattern = px.line(
                hourly_pattern,
                x='hr',
                y=['casual', 'registered', 'cnt'],
                title=f"Average Hourly Rentals ({day_type})",
                labels={'hr': 'Hour of Day', 'value': 'Average Rentals', 'variable': 'User Type'},
                color_discrete_map={
                    'casual': '#ff9999',
                    'registered': '#66b3ff',
                    'cnt': '#99ff99'
                }
            )
            st.plotly_chart(fig_hourly_pattern, use_container_width=True)
            
            # YoY growth analysis
            if len(monthly_agg['year_name'].unique()) > 1:
                st.subheader("Year-over-Year Growth")
                
                yoy = monthly_agg.pivot_table(index='month', columns='year_name', values='cnt').reset_index()
                if not yoy.empty and '2011' in yoy.columns and '2012' in yoy.columns:
                    yoy['growth_rate'] = (yoy['2012'] - yoy['2011']) / yoy['2011'] * 100
                    
                    fig_yoy = px.bar(
                        yoy,
                        x='month',
                        y='growth_rate',
                        title="Year-over-Year Growth Rate by Month (2011 to 2012)",
                        labels={'growth_rate': 'Growth Rate (%)', 'month': 'Month'},
                        color='growth_rate',
                        color_continuous_scale='RdBu',
                        range_color=[-10, 50]
                    )
                    fig_yoy.update_xaxes(tickvals=list(range(1, 13)),
                                         ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                    fig_yoy.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_yoy, use_container_width=True)
        
    elif page == "Anomaly Detection":
        st.header("Anomaly Detection")
        
        # Z-score threshold selector
        z_threshold = st.slider(
            "Z-Score Threshold for Anomaly Detection", 
            min_value=1.0, 
            max_value=4.0, 
            value=2.5, 
            step=0.1
        )
        
        # Prepare anomaly data
        day_anomaly = day_df.copy()
        day_anomaly['z_score'] = stats.zscore(day_anomaly['cnt'])
        day_anomaly['is_anomaly'] = day_anomaly['z_score'].apply(lambda x: True if abs(x) > z_threshold else False)
        anomalies = day_anomaly[day_anomaly['is_anomaly'] == True].sort_values('z_score', ascending=False)
        
        # Display count of anomalies
        st.info(f"Number of anomalous days detected: {len(anomalies)} out of {len(day_anomaly)} total days ({len(anomalies)/len(day_anomaly)*100:.1f}%)")
        
        # Visualize time series with anomalies
        st.subheader("Time Series with Anomalies")
        
        fig_anomaly = px.scatter(
            day_anomaly,
            x='dteday',
            y='cnt',
            color='is_anomaly',
            color_discrete_map={True: 'red', False: 'blue'},
            title="Daily Bike Rentals with Anomaly Detection",
            labels={'dteday': 'Date', 'cnt': 'Number of Rentals', 'is_anomaly': 'Is Anomaly'},
            hover_data=['weathersit_name', 'temp_celsius', 'hum_percent', 'z_score']
        )
        
        fig_anomaly.update_traces(marker=dict(size=8))
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Display anomalies in a table
        if not anomalies.empty:
            st.subheader("Anomalous Days")
            
            # Top anomalies (high rentals)
            st.markdown("#### Extremely High Rental Days")
            high_anomalies = anomalies[anomalies['z_score'] > 0].sort_values('z_score', ascending=False)
            if not high_anomalies.empty:
                st.dataframe(
                    high_anomalies[['dteday', 'season_name', 'weathersit_name', 'temp_celsius', 
                                   'hum_percent', 'cnt', 'z_score']].reset_index(drop=True),
                    hide_index=True
                )
            else:
                st.write("No high rental anomalies detected.")
            
            # Bottom anomalies (low rentals)
            st.markdown("#### Extremely Low Rental Days")
            low_anomalies = anomalies[anomalies['z_score'] < 0].sort_values('z_score')
            if not low_anomalies.empty:
                st.dataframe(
                    low_anomalies[['dteday', 'season_name', 'weathersit_name', 'temp_celsius', 
                                  'hum_percent', 'cnt', 'z_score']].reset_index(drop=True),
                    hide_index=True
                )
            else:
                st.write("No low rental anomalies detected.")
        
        # Compare characteristics between normal and anomalous days
        st.subheader("Comparison: Normal vs. Anomalous Days")
        
        anomaly_comparison = day_anomaly.groupby('is_anomaly').agg({
            'temp_celsius': 'mean',
            'hum_percent': 'mean',
            'windspeed_kph': 'mean',
            'cnt': 'mean'
        }).reset_index()
        
        anomaly_comparison['is_anomaly'] = anomaly_comparison['is_anomaly'].map({True: 'Anomaly', False: 'Normal'})
        
        for col in ['temp_celsius', 'hum_percent', 'windspeed_kph', 'cnt']:
            fig_comp = px.bar(
                anomaly_comparison,
                x='is_anomaly',
                y=col,
                color='is_anomaly',
                color_discrete_map={'Anomaly': 'red', 'Normal': 'blue'},
                title=f"Average {col} (Normal vs. Anomaly)",
                labels={'is_anomaly': 'Category', col: col}
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please make sure you have the correct data files (hour.csv and day.csv) in the same directory as this app.")
    
# Add footer
st.markdown("---")
st.markdown("Bike Rental Analysis Dashboard | Created with Streamlit")