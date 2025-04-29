from ultralytics.models.yolo.detect import DetectionValidator

class CalorieEstimationValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = CalorieMetrics()
        
    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        self.metrics.update(preds, batch["calorie_info"])
