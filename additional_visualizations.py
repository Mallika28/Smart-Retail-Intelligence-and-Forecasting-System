import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from visualizations import COLORS

def create_feature_importance_plot(importance_dict):
    """
    Create a horizontal bar chart for feature importance
    
    Parameters:
    importance_dict (dict): Dictionary of feature names and importance scores
    
    Returns:
    plotly.graph_objects.Figure: Interactive feature importance plot
    """
    if not importance_dict:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Feature Importance (No Data Available)",
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            height=400
        )
        return fig
    
    # Sort and convert to DataFrame
    sorted_importance = {k: v for k, v in sorted(importance_dict.items(), 
                                              key=lambda item: item[1], 
                                              reverse=True)}
    
    # Limit to top 10 features
    top_features = dict(list(sorted_importance.items())[:10])
    df = pd.DataFrame({
        'Feature': list(top_features.keys()),
        'Importance': list(top_features.values())
    })
    
    # Create horizontal bar chart
    fig = px.bar(
        df,
        y='Feature',
        x='Importance',
        orientation='h',
        color='Importance',
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'Importance': 'Relative Importance', 'Feature': ''},
        title='Feature Importance Analysis'
    )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_colorbar=dict(
            title="Importance",
            thicknessmode="pixels", 
            thickness=15,
            lenmode="pixels", 
            len=300,
            xanchor="left", 
            x=1.01
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Make responsive
    fig.update_layout(
        autosize=True,
        height=400,
    )
    
    return fig

def create_product_category_plot(df):
    """
    Create an interactive bar chart for sales by product category
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataframe
    
    Returns:
    plotly.graph_objects.Figure: Interactive bar chart
    """
    # Aggregate data by product category
    category_sales = df.groupby('Item_Type')['Item_Outlet_Sales'].agg(['mean', 'sum']).reset_index()
    category_sales.columns = ['Category', 'Average Sales', 'Total Sales']
    category_sales = category_sales.sort_values('Total Sales', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        category_sales,
        y='Category',
        x='Total Sales',
        color='Average Sales',
        orientation='h',
        color_continuous_scale=px.colors.sequential.Teal,
        labels={'Total Sales': 'Total Sales (₹)', 'Category': '', 'Average Sales': 'Average Sales (₹)'},
        title='Sales Performance by Product Category'
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Total Sales: ₹%{x:.2f}<br>Average Sales: ₹%{marker.color:.2f}<extra></extra>'
    )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_colorbar=dict(
            title="Avg Sales",
            thicknessmode="pixels", 
            thickness=15,
            lenmode="pixels", 
            len=300,
            xanchor="left", 
            x=1.01
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Make responsive
    fig.update_layout(
        autosize=True,
        height=450,
    )
    
    return fig

def create_outlet_comparison_plot(df):
    """
    Create an interactive radar chart for outlet comparison
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataframe
    
    Returns:
    plotly.graph_objects.Figure: Interactive radar chart
    """
    # Calculate metrics by outlet
    outlet_metrics = df.groupby('Outlet_Identifier').agg({
        'Item_Outlet_Sales': ['mean', 'std', 'count', 'sum'],
        'Item_MRP': 'mean',
        'Item_Visibility': 'mean'
    }).reset_index()
    
    # Flatten multi-level column index
    outlet_metrics.columns = ['_'.join(col).strip('_') for col in outlet_metrics.columns.values]
    
    # Calculate additional derived metrics
    outlet_metrics['Sales_Efficiency'] = outlet_metrics['Item_Outlet_Sales_sum'] / outlet_metrics['Item_Outlet_Sales_count']
    outlet_metrics['Price_Performance'] = outlet_metrics['Item_Outlet_Sales_mean'] / outlet_metrics['Item_MRP_mean']
    outlet_metrics['Visibility_Impact'] = outlet_metrics['Item_Outlet_Sales_mean'] / (outlet_metrics['Item_Visibility_mean'] + 0.001)
    
    # Normalize metrics for radar chart
    metrics_to_normalize = ['Item_Outlet_Sales_mean', 'Item_Outlet_Sales_sum', 
                          'Sales_Efficiency', 'Price_Performance', 'Visibility_Impact']
    
    for metric in metrics_to_normalize:
        min_val = outlet_metrics[metric].min()
        max_val = outlet_metrics[metric].max()
        outlet_metrics[f'{metric}_norm'] = (outlet_metrics[metric] - min_val) / (max_val - min_val)
    
    # Create radar chart
    fig = go.Figure()
    
    # Radar chart categories
    categories = ['Avg Sales', 'Total Sales', 'Sales Efficiency', 
                'Price Performance', 'Visibility Impact']
    
    # Add trace for each outlet
    for i, outlet in enumerate(outlet_metrics['Outlet_Identifier']):
        values = [
            outlet_metrics.loc[i, 'Item_Outlet_Sales_mean_norm'],
            outlet_metrics.loc[i, 'Item_Outlet_Sales_sum_norm'],
            outlet_metrics.loc[i, 'Sales_Efficiency_norm'],
            outlet_metrics.loc[i, 'Price_Performance_norm'],
            outlet_metrics.loc[i, 'Visibility_Impact_norm']
        ]
        
        # Close the polygon by repeating the first value
        values.append(values[0])
        categories_closed = categories + [categories[0]]
        
        color_idx = i % len(COLORS['palette'])
        # Convert hex to rgba for transparency
        hex_color = COLORS['palette'][color_idx].lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill='toself',
            name=outlet,
            line_color=COLORS['palette'][color_idx],
            fillcolor=f'rgba({r},{g},{b},0.3)'  # Use rgba for transparency
        ))
    
    # Customize layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Outlet Performance Comparison",
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(size=12),
        margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=500,
        autosize=True,
    )
    
    return fig

def create_mrp_sales_scatter(df):
    """
    Create an interactive scatter plot of MRP vs Sales
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataframe
    
    Returns:
    plotly.graph_objects.Figure: Interactive scatter plot
    """
    # Create sample for better visualization (if dataset is large)
    if len(df) > 5000:
        plot_df = df.sample(n=5000, random_state=42)
    else:
        plot_df = df.copy()
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='Item_MRP',
        y='Item_Outlet_Sales',
        color='Outlet_Type',
        size='Item_Visibility',
        size_max=15,
        opacity=0.7,
        hover_name='Item_Identifier',
        color_discrete_sequence=COLORS['palette'],
        labels={
            'Item_MRP': 'Item Price (₹)',
            'Item_Outlet_Sales': 'Sales (₹)',
            'Outlet_Type': 'Outlet Type',
            'Item_Visibility': 'Visibility'
        },
        title='Relationship between Product Price and Sales'
    )
    
    # Customize hover information
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>Price: ₹%{x:.2f}<br>Sales: ₹%{y:.2f}<br>Visibility: %{marker.size:.4f}<extra></extra>'
    )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
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
    
    # Make responsive
    fig.update_layout(
        autosize=True,
        height=450,
    )
    
    return fig