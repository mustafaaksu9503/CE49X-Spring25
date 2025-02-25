import pandas as pd
import numpy as num
from tabulate import tabulate

def load_data(file_path):
    """
    Load dataset from file.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def clean_data(df):
    """
    Fill missing values with column mean and remove outliers.
    """
    if df is None:
        return None
    
    df.fillna(df.mean(), inplace=True)
    
    for column in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
        if column in df.columns:
            mean, std = df[column].mean(), df[column].std()
            df = df[(df[column] >= mean - 3 * std) & (df[column] <= mean + 3 * std)]
    
    return df

def compute_statistics(df):
    """
    Compute and print descriptive statistics.
    """
    if df is None:
        print("No data available.")
        return
    
    stats = []
    for column in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
        stats.append([column, f"{df[column].min():.2f}", f"{df[column].max():.2f}", f"{df[column].mean():.2f}", f"{df[column].median():.2f}", f"{df[column].std():.2f}"])
    
    print(tabulate(stats, headers=["Category", "Min", "Max", "Mean", "Median", "Std Dev"], tablefmt='grid'))

def main():
    """
    Run soil analysis workflow.
    """
    file_path = "../../datasets/soil_test.csv"
    df = clean_data(load_data(file_path))
    compute_statistics(df)

if __name__ == "__main__":
    main()