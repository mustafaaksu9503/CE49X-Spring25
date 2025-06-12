"""
Data input module for LCA tool.
Handles reading and validating input data from various sources.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Union

class DataInput:
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.json']
        self.required_columns = [
            'product_id', 'product_name', 'life_cycle_stage', 'material_type',
            'quantity_kg', 'energy_consumption_kwh', 'transport_distance_km',
            'transport_mode', 'waste_generated_kg', 'recycling_rate',
            'landfill_rate', 'incineration_rate', 'carbon_footprint_kg_co2e',
            'water_usage_liters'
        ]
    
    def read_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read data from various file formats.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            DataFrame containing the input data
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.xlsx':
            return pd.read_excel(file_path)
        elif file_path.suffix == '.json':
            return pd.read_json(file_path)
            
    def validate_data(self, df: pd.DataFrame) -> bool:
        required_columns = [
        "product_id", "product_name", "life_cycle_stage", "material_type",
        "quantity_kg", "energy_consumption_kwh", "transport_distance_km",
        "transport_mode", "waste_generated_kg", "recycling_rate",
        "landfill_rate", "incineration_rate", "carbon_footprint_kg_co2e",
        "water_usage_liters"
        ]

        numeric_columns = [
        "quantity_kg", "energy_consumption_kwh", "transport_distance_km",
        "waste_generated_kg", "recycling_rate", "landfill_rate", "incineration_rate",
        "carbon_footprint_kg_co2e", "water_usage_liters"
        ]

        # Check for missing columns
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"[ERROR] Missing columns: {missing}")
            return False

        # Check for NaNs
        if df[required_columns].isnull().any().any():
            print("[ERROR] Data contains missing (NaN) values.")
            return False

        # Check numeric validity
        for col in numeric_columns:
            if not pd.to_numeric(df[col], errors="coerce").notna().all():
                print(f"[ERROR] Non-numeric values found in column: {col}")
                return False

        # Check that EoL rates sum to 1.0 per row (with some tolerance)
        rate_sum = df["recycling_rate"] + df["landfill_rate"] + df["incineration_rate"]
        if not ((rate_sum - 1.0).abs() < 0.01).all():
            print("[ERROR] End-of-life rates must sum to 1.0 per row.")
            return False

        return True

        
    def read_impact_factors(self, file_path: Union[str, Path]) -> Dict:
        """
        Read impact factors from JSON file.
        
        Args:
            file_path: Path to the impact factors JSON file
            
        Returns:
            Dictionary containing impact factors
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Impact factors file not found: {file_path}")
            
        if file_path.suffix != '.json':
            raise ValueError("Impact factors must be provided in JSON format")
            
        with open(file_path, 'r') as f:
            impact_factors = json.load(f)
            
        return impact_factors 