from ultralytics.engine.trainer import BaseTrainer
from .val import YOLOv10sCalorieValidator
from .loss import YOLOv10sCalorieLoss

class YOLOv10sCalorieTrainer(BaseTrainer):
    def get_model(self, cfg=None, weights=None):
        model = YOLOv10sCalorieModel(cfg or self.args.cfg, 
                                   nc=self.data['nc'],
                                   calorie_nc=self.data['calorie_nc'])
        if weights:
            model.load(weights)
        return model
    
    def get_validator(self):
        return YOLOv10sCalorieValidator(self.test_loader, 
                                      save_dir=self.save_dir,
                                      args=copy(self.args))
    
    def get_criterion(self):
        return YOLOv10sCalorieLoss(self.model, self.args)
    
    def preprocess_batch(self, batch):
        batch['calorie_targets'] = batch.pop('calorie')  # Move to device
        return super().preprocess_batch(batch)
3. Validation Component (val.py)
python
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
