import pandas as pd
import numpy as np

def estimate_calories(volume_cm3, food_name, density_df, calorie_df):
    """
    Estimate calories from volume for a given food.

    Parameters:
        volume_cm3 (float): The volume in cubic centimeters (cm^3).
        food_name (str): The name of the food item.
        density_df (pd.DataFrame): DataFrame with columns ['food', 'density_g_per_cm3'].
        calorie_df (pd.DataFrame): DataFrame with columns ['food', 'weight_g', 'calories'].

    Returns:
        Estimated calories (float).
    """
    # Ensure food exists in density data
    if food_name not in density_df['food'].values:
        raise ValueError(f"Food '{food_name}' not found in density data.")

    # Get density
    density = density_df.loc[density_df['food'] == food_name, 'density_g_per_cm3'].values[0]

    # Estimate weight
    estimated_weight = volume_cm3 * density

    # Filter calorie_df for this food
    food_calorie_data = calorie_df[calorie_df['food'] == food_name]

    if food_calorie_data.empty:
        raise ValueError(f"Calorie data not available for food: {food_name}")

    # Find the closest weight entry
    closest_row = food_calorie_data.iloc[(food_calorie_data['weight_g'] - estimated_weight).abs().argsort().iloc[0]]

    return closest_row['calories']

# Example usage:
# density_df = pd.read_csv("densities.csv")
# calorie_df = pd.read_csv("calories_by_weight.csv")
# volume = 240  # cm^3, for example
# food = "apple_pie"
# calories = estimate_calories(volume, food, density_df, calorie_df)
# print(f"Estimated calories: {calories:.1f} kcal")
