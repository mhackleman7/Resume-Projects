# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:19:52 2024

@author: mhack
"""

import yfinance as yf
import pandas as pd
import os
from openpyxl import Workbook
import numpy as np

# Define the list of ticker symbols
ticker_symbols = ["INTC", "GOOG", "AAPL", "MSFT", "KO", "TSLA", "NVDA", "AMZN", "AVGO", "V", "WMT", "UNH", "HD", "OXY", "JNJ", "COST", "CVX", "AMD", "MCD", "CSCO", "ABT", "CAT", "TXN", "PFE", "PM", "UPS", "T", "VZ", "DE", "BA", "LMT", "SBUX"]

# Initialize an empty DataFrame to store the DuPont Analysis breakdown
dupont_breakdown_df = pd.DataFrame(columns=["Net_Income_to_EBT", "EBT_to_EBIT", "EBIT_to_Revenues", "Revenues_to_Assets", "Assets_to_Equity"], index=ticker_symbols)

# Iterate over each ticker symbol
for ticker_symbol in ticker_symbols:
    try:
        # Fetch data for the current ticker
        ticker = yf.Ticker(ticker_symbol)
        
        # Fetch the income statement and balance sheet
        income_statement = ticker.financials
        balance_sheet = ticker.balance_sheet
        
        # Define the years
        years = [2023, 2022, 2021, 2020]
        
        # Initialize lists to store the DuPont ratios for each year
        net_income_to_ebt_list = []
        ebt_to_ebit_list = []
        ebit_to_revenues_list = []
        revenues_to_assets_list = []
        assets_to_equity_list = []

        # Perform DuPont Analysis for each year
        for year in years:
            # Get the relevant financial data for the year
            net_income = income_statement.loc['Net Income', str(year)]
            interest_expense = income_statement.loc['Interest Expense', str(year)]
            ebit = income_statement.loc['EBIT', str(year)]
            total_revenue = income_statement.loc['Total Revenue', str(year)]
            total_assets = balance_sheet.loc['Total Assets', str(year)]
            stockholders_equity = balance_sheet.loc['Stockholders Equity', str(year)]

            # Calculate DuPont ratios
            net_income_to_ebt = net_income / (ebit - interest_expense)
            ebt_to_ebit = (ebit - interest_expense) / ebit
            ebit_to_revenues = ebit / total_revenue
            revenues_to_assets = total_revenue / total_assets
            assets_to_equity = total_assets / stockholders_equity

            # Append the calculated ratios to the respective lists
            net_income_to_ebt_list.append(net_income_to_ebt)
            ebt_to_ebit_list.append(ebt_to_ebit)
            ebit_to_revenues_list.append(ebit_to_revenues)
            revenues_to_assets_list.append(revenues_to_assets)
            assets_to_equity_list.append(assets_to_equity)

        # Assign the lists of ratios to the corresponding rows in the DataFrame
        dupont_breakdown_df.loc[ticker_symbol] = [net_income_to_ebt_list, ebt_to_ebit_list, ebit_to_revenues_list, revenues_to_assets_list, assets_to_equity_list]

    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")

# Transpose the DataFrame so that the years become columns
dupont_breakdown_df = dupont_breakdown_df.transpose()

# Display the transposed DuPont Analysis breakdown DataFrame
print(dupont_breakdown_df)

# Re-initialize the DataFrame with the correct columns
metrics = ["Net_Income_to_EBT", "EBT_to_EBIT", "EBIT_to_Revenues", "Revenues_to_Assets", "Assets_to_Equity"]
years = [2023, 2022, 2021, 2020]
ticker_symbols = ["INTC", "GOOG", "AAPL", "MSFT", "KO", "TSLA", "NVDA", "AMZN", "AVGO", "V", "WMT", "UNH", "HD", "OXY", "JNJ", "COST", "CVX", "AMD", "MCD", "CSCO", "ABT", "CAT", "TXN", "PFE", "PM", "UPS", "T", "VZ", "DE", "BA", "LMT", "SBUX"]

# Create a list of all column names needed
column_names = [f"{ticker} {year}" for ticker in ticker_symbols for year in years]

# Initialize structured_df with the correct column names
structured_df = pd.DataFrame(index=metrics, columns=column_names)

# Ensure each metric's value is a scalar by explicitly converting it if necessary
for ticker_symbol in ticker_symbols:
    for year in years:
        for metric in metrics:
            year_index = years.index(year)
            try:
                metric_series = dupont_breakdown_df.loc[metric, ticker_symbol][year_index]
                
                # Check if the series is empty or if it's a single value
                if isinstance(metric_series, pd.Series) and not metric_series.empty:
                    metric_value = metric_series.iloc[0]
                else:
                    metric_value = np.nan  # Assign NaN if the series is empty or not a series
                
                column_name = f"{ticker_symbol} {year}"
                structured_df.loc[metric, column_name] = metric_value
                
            except IndexError:
                # Handle the case where the series is empty or out of bounds
                print(f"Data not found for {ticker_symbol}, {metric}, {year}.")
                structured_df.loc[metric, column_name] = np.nan  # Assign NaN if data is not available

structured_df.head()  # Display the first few rows to verify

structured_df

# Calculate ROE and add it to structured_df
roe_row = []

# Loop through each column (ticker and year combination)
for column_name in structured_df.columns:
    product = 1  # Start with a product of 1
    for metric in metrics:
        value = structured_df.loc[metric, column_name]
        product *= value if not pd.isnull(value) else 0  # Multiply by value if not NaN, else multiply by 1
    
    roe_row.append(product)

# Add the ROE row to structured_df
structured_df.loc['ROE'] = roe_row

structured_df.head()  # This will now include the ROE at the bottom

# Excel file path - this names the file "Hackleman Resume Project.xlsx"
excel_path = r'C:\Users\mhack\OneDrive\Documents\Investment Analysis\Hackleman Resume Project.xlsx'

# Check if the directory exists, if not, create it
directory = os.path.dirname(excel_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Create a new Excel file with an initial sheet
wb = Workbook()
ws = wb.active
ws.title = "ROE Calc"  # This names the first sheet within the workbook

# Save the workbook with the specified file name
wb.save(filename=excel_path)

# Write the DataFrame to the Excel file
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    structured_df.to_excel(writer, sheet_name='ROE Calc')

print("DuPont Analysis data for Portfolio has been saved to Excel.")