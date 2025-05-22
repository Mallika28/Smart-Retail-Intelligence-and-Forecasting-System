import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Define a consistent color scheme
COLORS = {
    'primary': '#4CAF50',
    'secondary': '#2E3B55',
    'accent': '#007acc',
    'background': '#a6bfd9',
    'neutral': '#95a5a6',
    'palette': px.colors.qualitative.G10
}

def create_mrp_plot(df):
    """
    Create an interactive histogram for Item MRP distribution
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataframe
    
    Returns:
    plotly.graph_objects.Figure: Interactive histogram
    """
    # Create histogram with density curve
    fig = px.histogram(
        df, 
        x='Item_MRP',
        color_discrete_sequence=[COLORS['accent']],
        opacity=0.7,
        histnorm='probability density',
        marginal='box', 
        title='Distribution of Item MRP'
    )
    
    # Add KDE curve
    kde = df['Item_MRP'].plot.kde()
    x = kde.get_children()[0]._x
    y = kde.get_children()[0]._y
    
    fig.add_trace(
        go.Scatter(
            x=x, 
            y=y, 
            mode='lines', 
            line=dict(width=2, color=COLORS['primary']),
            name='Density'
        )
    )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title='Item MRP ($)',
        yaxis_title='Density',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Make mobile responsive
    fig.update_layout(
        autosize=True,
        height=400,
    )
    
    return fig

def create_outlet_size_plot(df):
    """
    Create an interactive boxplot for sales by outlet size
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataframe
    
    Returns:
    plotly.graph_objects.Figure: Interactive boxplot
    """
    # Calculate statistics for annotations
    size_stats = df.groupby('Outlet_Size')['Item_Outlet_Sales'].agg(['mean', 'median']).reset_index()
    
    # Create boxplot
    fig = px.box(
        df, 
        x='Outlet_Size', 
        y='Item_Outlet_Sales',
        color='Outlet_Size',
        color_discrete_map={
            'Small': COLORS['palette'][0],
            'Medium': COLORS['palette'][1],
            'High': COLORS['palette'][2]
        },
        title='Sales Distribution by Outlet Size',
        points='outliers'
    )
    
    # Add mean markers
    for idx, row in size_stats.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['Outlet_Size']],
                y=[row['mean']],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=12,
                    color='yellow',
                    line=dict(color='black', width=1)
                ),
                name='Mean',
                showlegend=False,
                hoverinfo='y',
                hovertemplate='Mean: $%{y:.2f}'
            )
        )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title='Outlet Size',
        yaxis_title='Sales Value ($)',
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Make mobile responsive
    fig.update_layout(
        autosize=True,
        height=400,
    )
    
    return fig

def create_outlet_type_plot(df):
    """
    Create an interactive violin plot for sales by outlet type
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataframe
    
    Returns:
    plotly.graph_objects.Figure: Interactive violin plot
    """
    # Create violin plot
    fig = px.violin(
        df, 
        x='Outlet_Type', 
        y='Item_Outlet_Sales',
        color='Outlet_Type',
        box=True,
        points='outliers',
        title='Sales Distribution by Outlet Type'
    )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title='Outlet Type',
        yaxis_title='Sales Value ($)',
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Adjust for mobile responsiveness
    fig.update_layout(
        autosize=True,
        height=450,
    )
    
    # Rotate x-axis labels for better readability on mobile
    fig.update_xaxes(
        tickangle=30,
        tickfont=dict(size=10)
    )
    
    return fig

def create_correlation_heatmap(df):
    """
    Create an interactive correlation heatmap
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataframe
    
    Returns:
    plotly.graph_objects.Figure: Interactive heatmap
    """
    # Calculate correlation matrix for numerical features
    corr_matrix = df.select_dtypes(include='number').corr().round(2)
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title='Correlation Matrix of Numerical Features'
    )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
        coloraxis_colorbar=dict(
            title="Correlation",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            xanchor="left", x=1.05,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Make mobile responsive
    fig.update_layout(
        autosize=True
    )
    
    return fig
