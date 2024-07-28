"""
Created on Sun Jul 28 16:22:47 2024

@author: mhack
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image

# Load the Excel file from the specified location
file_path = 'C:/Users/mhack/OneDrive/Documents/Projects/Bond_Portfolio_Project.xlsx'
spreadsheet = pd.ExcelFile(file_path)

# Load the data from the two sheets into dataframes
mock_bond_portfolio_df = pd.read_excel(file_path, sheet_name='Mock Bond Portfolio')
benchmark_portfolio_df = pd.read_excel(file_path, sheet_name='Benchmark Portfolio')

# Define the function to calculate the fair value of a bond using discounted cash flows
def calculate_fair_value(face_value, coupon_rate, yield_to_maturity, time_to_maturity, frequency):
    periods = int(time_to_maturity * frequency)
    coupon_payment = face_value * coupon_rate / frequency
    discount_rate = yield_to_maturity / frequency / 100
    
    cash_flows = np.full(periods, coupon_payment)
    cash_flows[-1] += face_value
    
    discount_factors = (1 + discount_rate) ** np.arange(1, periods + 1)
    present_value = np.sum(cash_flows / discount_factors)
    
    return present_value

# Function to calculate effective duration and convexity
def calculate_duration_convexity(face_value, coupon_rate, yield_to_maturity, time_to_maturity, frequency, delta_y):
    initial_price = calculate_fair_value(face_value, coupon_rate, yield_to_maturity, time_to_maturity, frequency)
    
    yield_to_maturity_up = yield_to_maturity + delta_y
    price_up = calculate_fair_value(face_value, coupon_rate, yield_to_maturity_up, time_to_maturity, frequency)
    
    yield_to_maturity_down = yield_to_maturity - delta_y
    price_down = calculate_fair_value(face_value, coupon_rate, yield_to_maturity_down, time_to_maturity, frequency)
    
    duration = (price_down - price_up) / (2 * initial_price * delta_y)
    convexity = (price_down + price_up - 2 * initial_price) / (initial_price * delta_y ** 2)
    
    return duration, convexity

# Function to calculate DV01
def calculate_dv01(face_value, coupon_rate, yield_to_maturity, time_to_maturity, frequency):
    initial_price = calculate_fair_value(face_value, coupon_rate, yield_to_maturity, time_to_maturity, frequency)
    
    yield_to_maturity_up = yield_to_maturity + 0.0001  # 1 basis point increase
    price_up = calculate_fair_value(face_value, coupon_rate, yield_to_maturity_up, time_to_maturity, frequency)
    
    yield_to_maturity_down = yield_to_maturity - 0.0001  # 1 basis point decrease
    price_down = calculate_fair_value(face_value, coupon_rate, yield_to_maturity_down, time_to_maturity, frequency)
    
    dv01 = (price_down - price_up) / 2
    
    return dv01

# Map frequency strings to numerical values
frequency_map = {'Annual': 1, 'Semi-Annual': 2, 'Quarterly': 4}

# Function to process the portfolio data
def process_portfolio(df):
    df['Frequency Num'] = df['Frequency'].map(frequency_map)
    df['Fair Value/Bond'] = df.apply(
        lambda row: calculate_fair_value(
            row['Face Value/Bond'], 
            row['Coupon Rate'] / 100, 
            row['Yield to Maturity'], 
            row['Time to Maturity'], 
            row['Frequency Num']
        ), 
        axis=1
    )
    df['Fair Value Total'] = df['Fair Value/Bond'] * df['Quantity']
    df['Market Value Total'] = df['Market Value/Bond'] * df['Quantity']
    
    delta_y = 25 / 10000  # 25 basis points change in yield
    df[['Effective Duration', 'Convexity']] = df.apply(
        lambda row: calculate_duration_convexity(
            row['Face Value/Bond'], 
            row['Coupon Rate'] / 100, 
            row['Yield to Maturity'], 
            row['Time to Maturity'], 
            row['Frequency Num'],
            delta_y
        ), 
        axis=1, result_type='expand'
    )
    
    df['DV01'] = df.apply(
        lambda row: calculate_dv01(
            row['Face Value/Bond'], 
            row['Coupon Rate'] / 100, 
            row['Yield to Maturity'], 
            row['Time to Maturity'], 
            row['Frequency Num']
        ), 
        axis=1
    )
    
    total_market_value = df['Market Value Total'].sum()
    total_fair_value = df['Fair Value Total'].sum()
    
    return df, total_market_value, total_fair_value

# Process both portfolios
mock_bond_portfolio_df, total_market_value_mock, total_fair_value_mock = process_portfolio(mock_bond_portfolio_df)
benchmark_portfolio_df, total_market_value_benchmark, total_fair_value_benchmark = process_portfolio(benchmark_portfolio_df)

# Calculate portfolio-level metrics
def calculate_portfolio_metrics(df):
    total_market_value = df['Market Value Total'].sum()
    total_fair_value = df['Fair Value Total'].sum()
    weighted_duration = (df['Effective Duration'] * df['Market Value Total']).sum() / total_market_value
    weighted_convexity = (df['Convexity'] * df['Market Value Total']).sum() / total_market_value
    weighted_dv01 = df['DV01'].sum()
    
    return total_market_value, total_fair_value, weighted_duration, weighted_convexity, weighted_dv01

# Correct the unpacking here to match the return values
total_market_value_mock, total_fair_value_mock, mock_duration, mock_convexity, mock_dv01 = calculate_portfolio_metrics(mock_bond_portfolio_df)
total_market_value_benchmark, total_fair_value_benchmark, benchmark_duration, benchmark_convexity, benchmark_dv01 = calculate_portfolio_metrics(benchmark_portfolio_df)

# Create the new DataFrame for portfolio metrics
metrics_data = {
    'Metric': ['Market Value', 'Fair Value', 'Effective Duration', 'Convexity', 'DV01'],
    'Benchmark': [
        total_market_value_benchmark,
        total_fair_value_benchmark,
        benchmark_duration,
        benchmark_convexity,
        benchmark_dv01
    ],
    'Bond Portfolio': [
        total_market_value_mock,
        total_fair_value_mock,
        mock_duration,
        mock_convexity,
        mock_dv01
    ]
}

metrics_df = pd.DataFrame(metrics_data)

# Define asset classes
asset_classes = ["High Yield", "Private Placements", "CML", "Structured Credit", "MBS"]

# Create a new Excel writer object with 'replace' option for if_sheet_exists
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    metrics_df.to_excel(writer, sheet_name='Portfolio Metrics', index=False)

    # Overwrite the existing "Mock Bond Portfolio" and "Benchmark Portfolio" sheets
    mock_bond_portfolio_df.to_excel(writer, sheet_name='Mock Bond Portfolio', index=False)
    benchmark_portfolio_df.to_excel(writer, sheet_name='Benchmark Portfolio', index=False)

    for asset_class in asset_classes:
        # Filter the dataframes by asset class
        mock_asset_class_df = mock_bond_portfolio_df[mock_bond_portfolio_df['Category'] == asset_class]
        benchmark_asset_class_df = benchmark_portfolio_df[benchmark_portfolio_df['Category'] == asset_class]
        
        # Calculate metrics for the mock portfolio asset class
        total_market_value_mock_asset, total_fair_value_mock_asset, mock_duration_asset, mock_convexity_asset, mock_dv01_asset = calculate_portfolio_metrics(mock_asset_class_df)
        
        # Calculate metrics for the benchmark portfolio asset class
        total_market_value_benchmark_asset, total_fair_value_benchmark_asset, benchmark_duration_asset, benchmark_convexity_asset, benchmark_dv01_asset = calculate_portfolio_metrics(benchmark_asset_class_df)
        
        # Create the new DataFrame for asset class metrics
        metrics_data_asset_class = {
            'Metric': ['Market Value', 'Fair Value', 'Effective Duration', 'Convexity', 'DV01'],
            'Benchmark': [
                total_market_value_benchmark_asset,
                total_fair_value_benchmark_asset,
                benchmark_duration_asset,
                benchmark_convexity_asset,
                benchmark_dv01_asset
            ],
            'Bond Portfolio': [
                total_market_value_mock_asset,
                total_fair_value_mock_asset,
                mock_duration_asset,
                mock_convexity_asset,
                mock_dv01_asset
            ]
        }
        
        asset_class_metrics_df = pd.DataFrame(metrics_data_asset_class)
        
        # Export the asset class metrics to a new sheet in the Excel file
        sheet_name = f"{asset_class} Metrics"
        asset_class_metrics_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Display the DataFrame
        print(asset_class_metrics_df)

    # Generate and save charts for portfolio vs. benchmark comparison
    metrics = ['Market Value', 'Fair Value', 'Effective Duration', 'Convexity', 'DV01']

    # Create a new figure for each metric and save the charts as images
    figures = []
    for metric in metrics:
        fig, ax = plt.subplots()
        ax.bar(['Benchmark', 'Bond Portfolio'], metrics_df.loc[metrics_df['Metric'] == metric, ['Benchmark', 'Bond Portfolio']].values[0], color=['blue', 'orange'])
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        chart_filename = f"{metric}_comparison.png"
        plt.savefig(chart_filename)
        figures.append(chart_filename)
        plt.close(fig)

    # Create a new sheet in the workbook for charts
    workbook = writer.book
    worksheet = workbook.create_sheet(title='Charts')

    for idx, metric in enumerate(metrics):
        chart_filename = figures[idx]
        img = Image(chart_filename)
        worksheet.add_image(img, f'A{idx * 15 + 1}')

    workbook.save(file_path)

# Display the new DataFrame
print(metrics_df)
