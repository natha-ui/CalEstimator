from ultralytics.engine.model import Model
from ultralytics.nn.tasks import CalorieEstimationModel
from .val import CalorieEstimationValidator
from .predict import CalorieEstimationPredictor
from .train import CalorieEstimationTrainer
from huggingface_hub import PyTorchModelHubMixin

class CalorieEstimator(Model, PyTorchModelHubMixin):
    def __init__(self, model="calorie-estimator.pt", task=None, verbose=False, 
                 food_classes=None, calorie_map=None):
        super().__init__(model=model, task=task, verbose=verbose)
        self.food_classes = food_classes or DEFAULT_FOOD_CLASSES
        self.calorie_map = calorie_map or DEFAULT_CALORIE_MAP
        
    @property
    def task_map(self):
        return {
            "calorie": {
                "model": CalorieEstimationModel,
                "trainer": CalorieEstimationTrainer,
                "validator": CalorieEstimationValidator,
                "predictor": CalorieEstimationPredictor,
            }
        }
