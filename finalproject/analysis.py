import sys
sys.path.append('..')
import os

import pandas as pd
import matplotlib.pyplot as plt
from src.data_input import DataInput
from src.calculations import LCACalculator
from src.visualization import LCAVisualizer
from src.utils import save_results


data_input = DataInput()

product_data = data_input.read_data('data/raw/sample_data.csv')
print("Product Data Shape:", product_data.shape)
print("First 5 rows of the product data")
print(product_data.head())

impact_factors = data_input.read_impact_factors('data/raw/impact_factors.json')
print("Available Materials:", list(impact_factors.keys()))
print("\nImpact Factors for Wood:")
print(pd.DataFrame(impact_factors['wood']))

calculator = LCACalculator(impact_factors_path='data/raw/impact_factors.json')

impacts = calculator.calculate_impacts(product_data)
print("\n Calculated Impacts Shape:", impacts.shape)
print(impacts.head())
total_impacts = calculator.calculate_total_impacts(impacts)
print("\n Total Impacts by Product:")
print(total_impacts)
normalized_impacts = calculator.normalize_impacts(total_impacts)
print("\n First 5 rows of normalized impacts")
print(normalized_impacts.head())
comparison = calculator.compare_alternatives(impacts, ['P001', 'P002'])
print("\n Product Comparison:")
print(comparison)

os.makedirs('outputs', exist_ok=True)
save_results(total_impacts, 'outputs/total_impacts.xlsx', format='xlsx')
print("\n Results saved to 'outputs/total_impacts.xlsx'")

visualizer = LCAVisualizer()

fig = visualizer.plot_impact_breakdown(impacts, 'carbon_impact', 'material_type')
plt.show()
fig = visualizer.plot_life_cycle_impacts(impacts, 'P001')
plt.show()
fig = visualizer.plot_product_comparison(impacts, ['P001', 'P002'])
plt.show()
fig = visualizer.plot_end_of_life_breakdown(impacts, 'P001')
plt.show()
fig = visualizer.plot_impact_correlation(impacts)
plt.show()