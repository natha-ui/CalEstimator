from ultralytics.models.yolo.detect import DetectionTrainer
from .val import CalorieEstimationValidator

class CalorieEstimationTrainer(DetectionTrainer):
    def get_validator(self):
        return CalorieEstimationValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args),
            _callbacks=self.callbacks
        )
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CalorieEstimationModel(
            cfg, 
            nc=self.data["nc"],
            calorie_map=self.data["calorie_map"],
            verbose=verbose
        )
        if weights:
            model.load(weights)
        return model
