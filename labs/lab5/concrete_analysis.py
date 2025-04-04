import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(file_path):
    """Load the concrete strength dataset."""
    try:
        data = pd.read_excel(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def create_visualizations(data):
    """Create exploratory data analysis visualizations."""

    features = data.columns[:-1]  # All columns except last one
    target = data.columns[-1]     # Last column (compressive strength)
    
    # Calculate number of rows needed for subplots
    n_features = len(features)
    n_rows = (n_features + 1) // 2  # 2 plots per row, rounded up
    
    # Create figure with subplots
    plt.figure(figsize=(15, 4*n_rows))
    
    # Create a scatter plot for each feature vs compressive strength
    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, 2, i)
        plt.scatter(data[feature], data[target], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('Compressive Strength')
        plt.title(f'{feature} vs Compressive Strength')
    
    plt.tight_layout()  # Adjust spacing between subplots


def preprocess_data(data):
    """Preprocess the data for modeling."""

    data = data.fillna(data.mean())

    # Get the target column name (last column)
    target_column = data.columns[-1]
    
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train and evaluate the linear regression model."""
    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model, y_pred

def plot_results(y_test, y_pred):
    """Create visualization plots for model results."""
    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.75)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')

    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.75)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

def main():
    # Load the data
    concrete_excel = "../../datasets/concrete_strength/Concrete_Data.xls"
    data = load_data(concrete_excel)
    
    if data is None:
        return
    
    # Display basic information
    print("\nFirst few rows of the dataset:")
    print(data.head())
    print("\nSummary statistics:")
    print(data.describe())
    
    # Create visualizations
    create_visualizations(data)
    
    # Preprocess the data
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(data)
    
    # Train and evaluate the model
    model, y_pred = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Create result visualizations
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main() 