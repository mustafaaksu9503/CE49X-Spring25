import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    return data

def add_month(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['month'] = data['timestamp'].dt.month
    return data

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    
def add_season(data):
    data['season'] = data['month'].apply(get_season)
    return data

def monthseason(data):
    data = add_month(data)
    data = add_season(data)
    return data

def clean_data(data):
    if data is None:
        print("Data is missing")
        return None
    data = data.dropna()
    return data

def wind_speed(data):
    data['wind_speed'] = np.sqrt(data['u10m']**2 + data['v10m']**2)
    return data

def monthly_avg(data):
    monthly = data.groupby('month')['wind_speed'].mean().reset_index()
    return monthly

def seasonal_avg(data):
    seasonal = data.groupby('season')['wind_speed'].mean().reset_index()
    return seasonal

def add_hour(data):
    data['hour'] = data['timestamp'].dt.hour
    return data

def diurnal_avg(data):
    data = add_hour(data)
    diurnal = data.groupby('hour')['wind_speed'].mean().reset_index()
    return diurnal

def find_extremes(data, city_name):
    idx = data['wind_speed'].idxmax()
    extreme = data.loc[idx]
    print(f"\nHighest wind speed in {city_name}: {extreme['wind_speed']:.2f} m/s at {extreme['timestamp']}")
    return extreme

def plot_monthly_avg(berlin, munich):
    
    berlin_monthly = monthly_avg(berlin)
    munich_monthly = monthly_avg(munich)
    
    berlin_monthly["City"] = "Berlin"
    munich_monthly["City"] = "Munich"
    
    combined = pd.concat([berlin_monthly, munich_monthly])
    
    plt.figure(figsize=(10,6))
    sns.lineplot(data=combined, x="month", y="wind_speed", hue="City", marker="o")
    plt.xlabel("Month")
    plt.ylabel("Average Wind Speed (m/s)")
    plt.title("Monthly Average Wind Speed (m/s)")
    plt.grid(True)
    plt.show()

def plot_seasonal_avg(berlin, munich):
    
    berlin_seasonal = seasonal_avg(berlin)
    munich_seasonal = seasonal_avg(munich)
    
    berlin_seasonal["City"] = "Berlin"
    munich_seasonal["City"] = "Munich"
    
    combined = pd.concat([berlin_seasonal, munich_seasonal])
    
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    
    plt.figure(figsize=(10,6))
    sns.barplot(data=combined, x="season", y="wind_speed", hue="City", order=season_order)
    plt.xlabel("Season")
    plt.ylabel("Average Wind Speed (m/s)")
    plt.title("Seasonal Average Wind Speed (m/s)")
    plt.grid(True)
    plt.show()

def plot_diurnal_comparison(berlin, munich):
    
    berlin_diurnal = diurnal_avg(berlin)
    munich_diurnal = diurnal_avg(munich)
    
    berlin_diurnal["City"] = "Berlin"
    munich_diurnal["City"] = "Munich"
    
    combined = pd.concat([berlin_diurnal, munich_diurnal])
    
    plt.figure(figsize=(10,6))
    sns.lineplot(data=combined, x="hour", y="wind_speed", hue="City", marker="o")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Average Wind Speed (m/s)")
    plt.title("Diurnal Comparison of Wind Speed (m/s)")
    plt.grid(True)
    plt.show()

def main():
    berlin_file = "../../datasets/berlin_era5_wind_20241231_20241231.csv"
    munich_file = "../../datasets/munich_era5_wind_20241231_20241231.csv"
    
    berlin_data = load_data(berlin_file)
    munich_data = load_data(munich_file)
    
    berlin_wind = wind_speed(clean_data(monthseason(berlin_data)))
    munich_wind = wind_speed(clean_data(monthseason(munich_data)))
    
    print("Berlin Wind Speed Averages")
    print("\nMonthly Averages:")
    print(monthly_avg(berlin_wind).to_string(index=False))
    print("\nSeasonal Averages:")
    print(seasonal_avg(berlin_wind).to_string(index=False))
    print("\nMunich Wind Speed Averages")
    print("\nMonthly Averages:")
    print(monthly_avg(munich_wind).to_string(index=False))
    print("\nSeasonal Averages:")
    print(seasonal_avg(munich_wind).to_string(index=False))
    
    find_extremes(berlin_wind, "Berlin")
    find_extremes(munich_wind, "Munich")
    
    plot_monthly_avg(berlin_wind, munich_wind)
    plot_seasonal_avg(berlin_wind, munich_wind)
    plot_diurnal_comparison(berlin_wind, munich_wind)

if __name__ == "__main__":
    main()
