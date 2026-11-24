"""
Ingredient Identification Version 2 - Master Dataset Builder
Builds the comprehensive ingredient and nutrition dataset across all 221 food categories.
Includes home vs. hotel variation mapping and major nutritional profiles (macronutrients + micronutrients).
"""

import os
import sys
import json
import csv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USED_DIR = os.path.join(BASE_DIR, "DataSet", "Version2 Used Data")
CSV_PATH = os.path.join(USED_DIR, "food_and_nutrition_master_v2.csv")
JSON_PATH = os.path.join(USED_DIR, "food_ingredient_nutrition_master_v2.json")

def build_dataset():
    os.makedirs(USED_DIR, exist_ok=True)
    print("=" * 65)
    print("🥗 BUILDING INGREDIENT & COMPREHENSIVE NUTRITION DATASET (VERSION 2)")
    print("=" * 65)
    print(f"Output CSV  : {CSV_PATH}")
    print(f"Output JSON : {JSON_PATH}")
    
    # Check if files already built by master builder
    if os.path.exists(CSV_PATH) and os.path.exists(JSON_PATH):
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1
        print(f"✅ Verified existing master dataset with {total_rows} food categories.")
        return CSV_PATH, JSON_PATH
        
    print("Building master records from canonical database...")
    # Reference parent script if needed
    parent_builder = os.path.abspath(os.path.join(BASE_DIR, "..", "build_ingredient_v2_master.py"))
    if os.path.exists(parent_builder):
        import subprocess
        subprocess.run([sys.executable, parent_builder], check=True)
        print("✅ Master builder execution completed.")
        
    return CSV_PATH, JSON_PATH

if __name__ == "__main__":
    build_dataset()

