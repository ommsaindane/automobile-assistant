# backend/compare.py

import os
import pandas as pd

# Path to the dataset
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "Car Dataset 1945-2020.csv")

# Load dataset once
try:
    car_df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    car_df = pd.DataFrame()
    print(f"Dataset not found at {DATA_PATH}")

def compare_cars(car1: str, car2: str, features: list = []):
    """
    Compare two cars based on specified features.
    If features list is empty, return all columns.
    """
    if car_df.empty:
        raise ValueError("Car dataset not loaded.")

    # Check if cars exist
    car1_row = car_df[car_df['Car'].str.lower() == car1.lower()]
    car2_row = car_df[car_df['Car'].str.lower() == car2.lower()]

    if car1_row.empty:
        raise ValueError(f"Car '{car1}' not found in dataset.")
    if car2_row.empty:
        raise ValueError(f"Car '{car2}' not found in dataset.")

    # Select features or all columns
    if features:
        missing_features = [f for f in features if f not in car_df.columns]
        if missing_features:
            raise ValueError(f"Features not found in dataset: {missing_features}")
        car1_row = car1_row[features]
        car2_row = car2_row[features]
    else:
        features = car_df.columns.tolist()  # all columns

    comparison = {}
    for f in features:
        comparison[f] = {
            car1: car1_row.iloc[0][f],
            car2: car2_row.iloc[0][f]
        }

    return comparison
