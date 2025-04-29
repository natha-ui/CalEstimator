from ultralytics import CalorieEstimator

# Initialize model
model = CalorieEstimator(
    food_classes=["apple", "pizza", "salad"],
    calorie_map={"apple": 0.52, "pizza": 2.65, "salad": 0.35}  # calories per pixel
)

# Train on food dataset
model.train(data="food_calories.yaml", epochs=50)

# Predict calories
results = model.predict("meal.jpg")

# Show results
for result in results:
    for box in result.boxes:
        print(f"{result.names[int(box.cls)]}: {box.calories:.2f} calories")
