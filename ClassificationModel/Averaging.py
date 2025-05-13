import pandas as pd

# Load nutrition data once
nutrition_df = pd.read_csv("nutrition.csv")

# Precompute average portion sizes (in grams)
average_portion_sizes = (
    nutrition_df
    .groupby('label')['weight']
    .mean()
    .to_dict()
)

def get_average_portion(label: str) -> float:
    """
    Return the average portion size (in grams) for the given food label.
    Defaults to 100g if the label isn't found.
    """
    return average_portion_sizes.get(label.lower(), 100.0)

def estimate_calories(label: str, est_weight: float = None) -> float:
    """
    Estimate calories for a food label.
    If est_weight is None, uses the average portion size.
    Finds the closest weight entry in nutrition_df and returns its calories.
    """
    # Standardize
    label = label.lower()
    df_label = nutrition_df[nutrition_df['label'] == label]
    if df_label.empty:
        raise ValueError(f"No nutrition data for label '{label}'")

    # Determine weight to use
    if est_weight is None:
        est_weight = get_average_portion(label)

    # Find the closest matching weight entry
    closest_idx = (df_label['weight'] - est_weight).abs().idxmin()
    calories = df_label.loc[closest_idx, 'calories']
    return calories
