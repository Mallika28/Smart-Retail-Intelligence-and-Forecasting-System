import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
from datetime import datetime
from fpdf import FPDF
import io
import streamlit as st

# Set consistent style for the reports
plt.style.use('seaborn-v0_8-whitegrid')

def create_pdf_report(df, predictions=None, feature_importance=None):
    """
    Create a PDF report with key insights from the retail data
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataframe
    predictions (dict, optional): Dictionary with prediction results
    feature_importance (dict, optional): Feature importance dictionary
    
    Returns:
    BytesIO: PDF file as bytes
    """
    # Create PDF object
    pdf = FPDF()
    pdf.add_page()
    
    # Set styles
    pdf.set_font('Arial', 'B', 16)
    
    # Title
    pdf.cell(190, 10, 'Retail Forecasting Dashboard - Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(190, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    pdf.ln(10)
    
    # Summary Statistics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Summary Statistics', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    # Create summary statistics
    summary = df.describe().round(2)
    
    # Convert to text for PDF
    summary_text = "Key Statistics for Sales: \n"
    summary_text += f"- Average Sales: ₹{df['Item_Outlet_Sales'].mean():.2f}\n"
    summary_text += f"- Minimum Sales: ₹{df['Item_Outlet_Sales'].min():.2f}\n"
    summary_text += f"- Maximum Sales: ₹{df['Item_Outlet_Sales'].max():.2f}\n"
    summary_text += f"- Standard Deviation: ₹{df['Item_Outlet_Sales'].std():.2f}\n\n"
    
    summary_text += "Dataset Information: \n"
    summary_text += f"- Total Records: {len(df)}\n"
    summary_text += f"- Unique Products: {df['Item_Identifier'].nunique()}\n"
    summary_text += f"- Unique Outlets: {df['Outlet_Identifier'].nunique()}\n\n"
    
    # Add summary text to PDF
    pdf.multi_cell(190, 5, summary_text)
    pdf.ln(5)
    
    # Add outlet performance section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Outlet Performance', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    # Calculate outlet performance
    outlet_performance = df.groupby('Outlet_Identifier')['Item_Outlet_Sales'].agg(['mean', 'sum']).sort_values('sum', ascending=False).reset_index()
    outlet_performance = outlet_performance.round(2)
    
    # Create outlet performance text
    outlet_text = "Top Performing Outlets by Total Sales:\n"
    for i, row in outlet_performance.head(5).iterrows():
        outlet_text += f"{i+1}. {row['Outlet_Identifier']}: ₹{row['sum']:.2f} (Avg: ₹{row['mean']:.2f})\n"
    
    pdf.multi_cell(190, 5, outlet_text)
    pdf.ln(5)
    
    # Add product performance section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Product Performance', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    # Calculate product type performance
    product_performance = df.groupby('Item_Type')['Item_Outlet_Sales'].agg(['mean', 'sum']).sort_values('sum', ascending=False).reset_index()
    product_performance = product_performance.round(2)
    
    # Create product performance text
    product_text = "Top Performing Product Categories by Total Sales:\n"
    for i, row in product_performance.head(5).iterrows():
        product_text += f"{i+1}. {row['Item_Type']}: ₹{row['sum']:.2f} (Avg: ₹{row['mean']:.2f})\n"
    
    pdf.multi_cell(190, 5, product_text)
    pdf.ln(5)
    
    # Add prediction results if provided
    if predictions is not None:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(190, 10, 'Sales Prediction Results', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        prediction_text = "Prediction Details:\n"
        prediction_text += f"- Estimated Sales: ₹{predictions.get('prediction', 0):.2f}\n"
        prediction_text += f"- Expected Range: ₹{predictions.get('lower_bound', 0):.2f} to ₹{predictions.get('upper_bound', 0):.2f}\n\n"
        
        prediction_text += "Input Parameters:\n"
        for key, value in predictions.get('inputs', {}).items():
            prediction_text += f"- {key}: {value}\n"
        
        pdf.multi_cell(190, 5, prediction_text)
        pdf.ln(5)
    
    # Add feature importance if provided
    if feature_importance is not None and len(feature_importance) > 0:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(190, 10, 'Feature Importance Analysis', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        importance_text = "Key Factors Influencing Sales (in order of importance):\n"
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
            importance_text += f"{i+1}. {feature}: {importance:.4f}\n"
        
        pdf.multi_cell(190, 5, importance_text)
        pdf.ln(5)
    
    # Create a BytesIO object to store the PDF
    pdf_output = io.BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)
    
    return pdf_output

def get_download_link(pdf_bytes, filename="retail_forecast_report.pdf"):
    """
    Generate a download link for the PDF report
    
    Parameters:
    pdf_bytes (BytesIO): PDF file as bytes
    filename (str): Name of the output file
    
    Returns:
    str: HTML link for downloading the PDF
    """
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href

def export_to_excel(df, predictions=None):
    """
    Export data to Excel format
    
    Parameters:
    df (pd.DataFrame): DataFrame to export
    predictions (dict, optional): Dictionary with prediction results
    
    Returns:
    BytesIO: Excel file as bytes
    """
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Export main data
    df.to_excel(writer, sheet_name='Data', index=False)
    
    # Add summary sheet
    summary = pd.DataFrame({
        'Metric': ['Average Sales', 'Minimum Sales', 'Maximum Sales', 'Standard Deviation',
                  'Total Records', 'Unique Products', 'Unique Outlets'],
        'Value': [
            f"₹{df['Item_Outlet_Sales'].mean():.2f}",
            f"₹{df['Item_Outlet_Sales'].min():.2f}",
            f"₹{df['Item_Outlet_Sales'].max():.2f}",
            f"₹{df['Item_Outlet_Sales'].std():.2f}",
            len(df),
            df['Item_Identifier'].nunique(),
            df['Outlet_Identifier'].nunique()
        ]
    })
    summary.to_excel(writer, sheet_name='Summary', index=False)
    
    # Add outlet performance sheet
    outlet_performance = df.groupby('Outlet_Identifier')['Item_Outlet_Sales'].agg(['mean', 'sum']).sort_values('sum', ascending=False).reset_index()
    outlet_performance.columns = ['Outlet_Identifier', 'Average Sales', 'Total Sales']
    outlet_performance.to_excel(writer, sheet_name='Outlet Performance', index=False)
    
    # Add product type performance sheet
    product_performance = df.groupby('Item_Type')['Item_Outlet_Sales'].agg(['mean', 'sum']).sort_values('sum', ascending=False).reset_index()
    product_performance.columns = ['Product Type', 'Average Sales', 'Total Sales']
    product_performance.to_excel(writer, sheet_name='Product Performance', index=False)
    
    # Add predictions if provided
    if predictions is not None:
        pred_df = pd.DataFrame({
            'Parameter': list(predictions.get('inputs', {}).keys()) + ['Predicted Sales', 'Lower Bound', 'Upper Bound'],
            'Value': list(predictions.get('inputs', {}).values()) + [
                f"₹{predictions.get('prediction', 0):.2f}",
                f"₹{predictions.get('lower_bound', 0):.2f}",
                f"₹{predictions.get('upper_bound', 0):.2f}"
            ]
        })
        pred_df.to_excel(writer, sheet_name='Prediction Results', index=False)
    
    writer.close()
    output.seek(0)
    
    return output

def get_excel_download_link(excel_bytes, filename="retail_forecast_data.xlsx"):
    """
    Generate a download link for the Excel file
    
    Parameters:
    excel_bytes (BytesIO): Excel file as bytes
    filename (str): Name of the output file
    
    Returns:
    str: HTML link for downloading the Excel file
    """
    b64 = base64.b64encode(excel_bytes.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel Data</a>'
    return href