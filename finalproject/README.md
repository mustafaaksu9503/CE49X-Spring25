# Life Cycle Analysis Tool - Mustafa Aksu - CE49X Spring 2025 Final Project 

#  ✍️ What This Is
This project is a Life Cycle Analysis (LCA) tool. It helps calculate and visualize the environmental footprint (carbon, energy, water, etc.) of different products across their full life cycle — from manufacturing to waste. The tool uses Python for data input, processing, and plotting.

#  ⚙️ Main Features:
* Reads and validates raw product data (CSV)

* Loads material impact factors from a JSON file

* Calculates environmental impacts (carbon, energy, water) using life cycle stage and material type

* Visualizes the results via bar plots, radar charts, and more

#  🧠 How It Works (In Simple Terms)

1. Input Data: You give it a .csv file with product info and a .json file with material impact factors.
2. Processing: The tool calculates how much carbon, energy, and water each material and stage contributes.
3. Output: It visualizes the environmental impact in different ways so it's easier to compare and analyze.

#  📂 File Structure

```bash
finalproject/
├── src/
├─────── data_input.py              # Reading + validating input data
├─────── calculations.py            # Main impact calculations
├─────── visualization.py           # Graph generation
├─────── utils.py                   # Helper functions
├── tests/
├─────── test_data_input.py         # Unit tests for data input
├─────── test_calculations.py       # Unit tests for impact calculations
├─────── test_visualization.py      # Unit tests for plots
├── data/
├─────── sample_data.csv            # Material data
├─────── impact_factors.json        # Impact Factors data
├── outputs/                        
├─────── outputs.csv/.xlsx/.json    # Outputs from save_results              
├── requirements.txt                # File with necessary libraries
├── README.md                       # (This file)
├── lca_analysis.ipynb              # Basic guide notebook to see what each function does
└── analysis.py                     # Does every task in one script
```

#  🚀 How To Run It
1. Clone the repo:
```bash
git clone https://github.com/mustafaaksu9503/CE49X-Spring25
cd finalproject
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Python script 'analysis.py' from your terminal.
```bash
python analysis.py
```

# 📊 Data Structure

## Product Data (CSV)
The tool expects product data in CSV format with the following columns:
- `product_id`: Unique identifier for the product
- `product_name`: Name of the product
- `life_cycle_stage`: Stage in the life cycle (Manufacturing, Transportation, End-of-Life)
- `material_type`: Type of material used
- `quantity_kg`: Quantity in kilograms
- `energy_consumption_kwh`: Energy consumption in kilowatt-hours
- `transport_distance_km`: Transportation distance in kilometers
- `transport_mode`: Mode of transportation
- `waste_generated_kg`: Waste generated in kilograms
- `recycling_rate`: Rate of recycling (0-1)
- `landfill_rate`: Rate of landfill disposal (0-1)
- `incineration_rate`: Rate of incineration (0-1)
- `carbon_footprint_kg_co2e`: Carbon footprint in kg CO2e
- `water_usage_liters`: Water usage in liters

## Impact Factors (JSON)
Impact factors are stored in JSON format with the following structure:
```json
{
    "material_name": {
        "life_cycle_stage": {
            "carbon_impact": value,
            "energy_impact": value,
            "water_impact": value
        }
    }
}
```


# 🧪 Running Tests

To verify everything is working:

```bash
cd tests
pytest test_data_input.py
pytest test_calculations.py
pytest test_visualization.py
```

# ✅ Test Results
1. Data Input Test - 4/4 - %100
2. Calculation Test - 4/4 - %100
3. Visualization Test - 5/5 - %100

# 💭 What I Learned

* How to organize a Python project with testing and modular design
* How to use Python for real-world sustainability analysis
* How to structure real-world CSV and JSON data
* Basics of carbon, water, and energy impact modeling
