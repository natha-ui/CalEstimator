from ultralytics import YOLOv10
model = YOLOv10.from_pretrained('jameslahm/yolov10s')
source = 'food1.jpeg'
results = model.predict(source=source, save=True)
for result in results:
    print(result.boxes)
