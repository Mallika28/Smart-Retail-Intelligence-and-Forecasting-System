import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Define a consistent color scheme (matching visualizations.py)
COLORS = {
    'primary': '#4CAF50',
    'secondary': '#2E3B55',
    'accent': '#007acc',
    'background': '#f7f9fb',
    'neutral': '#95a5a6',
    'palette': px.colors.qualitative.G10
}

def generate_time_series_data(df, start_date=None, days=365):
    """
    Generate time series data by distributing existing sales data over time
    
    Parameters:
    df (pd.DataFrame): Original sales data
    start_date (str or datetime, optional): Starting date for the time series
    days (int): Number of days to distribute the data over
    
    Returns:
    pd.DataFrame: Time series data frame
    """
    if start_date is None:
        # Default to 1 year ago
        start_date = datetime.now() - timedelta(days=days)
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Create date range
    date_range = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Create time series template
    ts_data = pd.DataFrame({
        'Date': date_range,
        'Day_of_Week': date_range.day_name(),
        'Month': date_range.month_name(),
        'Year': date_range.year,
        'Week': [d.isocalendar()[1] for d in date_range],  # Updated for better compatibility
        'Day': date_range.day,
        'Sales': np.zeros(len(date_range))
    })
    
    # Total actual sales from the dataset
    total_sales = df['Item_Outlet_Sales'].sum()
    
    # Create a realistic sales pattern with day-of-week and seasonal effects
    # Day of week effect: Weekend (Fri-Sun) has higher sales
    day_of_week_effect = {
        'Monday': 0.8,
        'Tuesday': 0.9,
        'Wednesday': 1.0,
        'Thursday': 1.1,
        'Friday': 1.3,
        'Saturday': 1.5,
        'Sunday': 1.2
    }
    
    # Month effect: Holiday season and summer have higher sales
    month_effect = {
        'January': 0.9,    # Post-holiday slump
        'February': 0.85,  # Winter low
        'March': 0.95,     # Spring beginning
        'April': 1.0,      # Normal
        'May': 1.05,       # Pre-summer
        'June': 1.1,       # Summer beginning
        'July': 1.15,      # Summer peak
        'August': 1.1,     # Late summer
        'September': 1.0,  # Back to school
        'October': 1.05,   # Fall
        'November': 1.2,   # Pre-holiday
        'December': 1.5    # Holiday season
    }
    
    # Apply day of week and month effects
    for idx, row in ts_data.iterrows():
        day_effect = day_of_week_effect.get(row['Day_of_Week'], 1.0)
        month_effect_val = month_effect.get(row['Month'], 1.0)
        
        # Combine effects
        combined_effect = day_effect * month_effect_val
        
        # Add random noise (±10%)
        noise = np.random.normal(1, 0.1)
        
        # Apply trend over time (slight growth)
        trend = 1 + (idx / len(ts_data)) * 0.2  # 20% growth over the period
        
        # Store the effect
        ts_data.loc[idx, 'Effect'] = combined_effect * noise * trend
    
    # Normalize effects to ensure total matches original data
    total_effect = ts_data['Effect'].sum()
    ts_data['Sales'] = ts_data['Effect'] * (total_sales / total_effect)
    
    # Add weekly and monthly aggregated data
    ts_data = ts_data.drop(columns=['Effect'])
    
    return ts_data

def create_time_series_plot(ts_data):
    """
    Create an interactive time series plot
    
    Parameters:
    ts_data (pd.DataFrame): Time series data
    
    Returns:
    plotly.graph_objects.Figure: Interactive time series plot
    """
    # Weekly aggregation for smoother visualization
    weekly_data = ts_data.groupby(['Year', 'Week'])['Sales'].sum().reset_index()
    weekly_data['Date'] = weekly_data.apply(
        lambda x: pd.to_datetime(f"{int(x['Year'])}-W{int(x['Week'])}-1", format='%G-W%V-%u'), 
        axis=1
    )
    
    # Create the plot
    fig = go.Figure()
    
    # Add weekly sales line
    fig.add_trace(
        go.Scatter(
            x=weekly_data['Date'],
            y=weekly_data['Sales'],
            mode='lines+markers',
            name='Weekly Sales',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=6),
            hovertemplate='Week of %{x}<br>Sales: ₹%{y:.2f}<extra></extra>'
        )
    )
    
    # Add monthly trend line
    monthly_data = ts_data.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
    # Convert month names to datetime for proper ordering
    month_order = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    monthly_data['MonthNum'] = monthly_data['Month'].map(month_order)
    monthly_data['Date'] = monthly_data.apply(
        lambda x: pd.to_datetime(f"{int(x['Year'])}-{int(x['MonthNum'])}-15"), 
        axis=1
    )
    monthly_data = monthly_data.sort_values('Date')
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['Date'],
            y=monthly_data['Sales'],
            mode='lines',
            name='Monthly Trend',
            line=dict(color=COLORS['accent'], width=3, dash='dot'),
            hovertemplate='%{x}<br>Monthly Sales: ₹%{y:.2f}<extra></extra>'
        )
    )
    
    # Customize layout
    fig.update_layout(
        title='Sales Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Sales Value (₹)',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    # Make responsive
    fig.update_layout(
        autosize=True,
        height=400,
    )
    
    return fig

def forecast_sales(ts_data, periods=12):
    """
    Forecast sales for future periods
    
    Parameters:
    ts_data (pd.DataFrame): Time series data
    periods (int): Number of weeks to forecast
    
    Returns:
    tuple: (forecast_fig, forecast_data)
        - forecast_fig: plotly.graph_objects.Figure with forecast visualization
        - forecast_data: pd.DataFrame with forecast results
    """
    # Aggregate data to weekly level for forecasting
    weekly_data = ts_data.groupby(['Year', 'Week'])['Sales'].sum().reset_index()
    weekly_data['Date'] = weekly_data.apply(
        lambda x: pd.to_datetime(f"{int(x['Year'])}-W{int(x['Week'])}-1", format='%G-W%V-%u'), 
        axis=1
    )
    
    # Create features for forecasting (week number, etc.)
    weekly_data['WeekOfYear'] = weekly_data['Week']
    weekly_data['DayOfYear'] = weekly_data['Date'].dt.dayofyear
    weekly_data['MonthOfYear'] = weekly_data['Date'].dt.month
    
    # Prepare data for model
    X = np.column_stack([
        weekly_data['WeekOfYear'],
        np.sin(2 * np.pi * weekly_data['WeekOfYear'] / 52),  # Yearly seasonality
        np.cos(2 * np.pi * weekly_data['WeekOfYear'] / 52),
        np.sin(2 * np.pi * weekly_data['MonthOfYear'] / 12),  # Monthly seasonality
        np.cos(2 * np.pi * weekly_data['MonthOfYear'] / 12),
    ])
    y = weekly_data['Sales'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Create future dates
    last_date = weekly_data['Date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=7),
        periods=periods,
        freq='W-MON'
    )
    
    # Prepare future features
    future_weeks = [d.isocalendar()[1] for d in future_dates]
    future_months = [d.month for d in future_dates]
    
    future_X = np.column_stack([
        future_weeks,
        np.sin(2 * np.pi * np.array(future_weeks) / 52),
        np.cos(2 * np.pi * np.array(future_weeks) / 52),
        np.sin(2 * np.pi * np.array(future_months) / 12),
        np.cos(2 * np.pi * np.array(future_months) / 12),
    ])
    
    # Scale future features
    future_X_scaled = scaler.transform(future_X)
    
    # Make predictions
    future_y = model.predict(future_X_scaled)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_Sales': future_y,
        'Lower_Bound': future_y * 0.85,  # 15% lower bound
        'Upper_Bound': future_y * 1.15,  # 15% upper bound
    })
    
    # Create visualization
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=weekly_data['Date'],
            y=weekly_data['Sales'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=5),
            hovertemplate='%{x}<br>Sales: ₹%{y:.2f}<extra></extra>'
        )
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecasted_Sales'],
            mode='lines+markers',
            name='Forecasted Sales',
            line=dict(color=COLORS['accent'], width=2, dash='dot'),
            marker=dict(size=7, symbol='diamond'),
            hovertemplate='%{x}<br>Forecast: ₹%{y:.2f}<extra></extra>'
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
            y=forecast_df['Upper_Bound'].tolist() + forecast_df['Lower_Bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 122, 204, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='Forecast Range'
        )
    )
    
    # Customize layout
    fig.update_layout(
        title='Sales Forecast for Coming Weeks',
        xaxis_title='Date',
        yaxis_title='Weekly Sales (₹)',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    # Make responsive
    fig.update_layout(
        autosize=True,
        height=400,
    )
    
    return fig, forecast_df