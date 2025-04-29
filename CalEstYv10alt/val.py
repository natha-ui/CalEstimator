from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import DetMetrics, CalorieMetrics

class YOLOv10sCalorieValidator(BaseValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {
            'detection': DetMetrics(),
            'calorie': CalorieMetrics()
        }
    
    def init_metrics(self):
        return self.metrics
    
    def update_metrics(self, preds, batch):
        # Update detection metrics
        self.metrics['detection'].update(
            preds['detection'], 
            batch['bboxes']
        )
        
        # Update calorie metrics
        self.metrics['calorie'].update(
            preds['calorie'],
            batch['calorie_targets']
        )
