# Example usage:
label = 'apple_pie'
# Use average portion
cal_avg = estimate_calories(label)
print(f"Estimated Calories (avg {get_average_portion(label):.0f}g): {cal_avg} kcal")

# Or specify custom weight
cal_custom = estimate_calories(label, est_weight=130)
print(f"Estimated Calories (130g): {cal_custom} kcal")
