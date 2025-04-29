from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v10DetectionLoss
import torch.nn as nn

class YOLOv10sCalorieModel(DetectionModel):
    def __init__(self, cfg='yolov10s.yaml', ch=3, nc=80, calorie_nc=100):
        super().__init__(cfg, ch, nc)
        
        # Add calorie-specific heads
        self.calorie_reg = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, max(round(x * 0.25), 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(max(round(x * 0.25), 1, 1)
            ) for x in self.save
        )
        
        # Food classification enhancement
        self.food_cls = nn.ModuleList(
            nn.Conv2d(x, calorie_nc, 1) for x in self.save
        )
        
        # Initialize new layers
        self.initialize_biases()
        
    def forward(self, x):
        # Original YOLOv10s forward
        y = super().forward(x)
        
        # Add calorie outputs
        calorie_pred = [reg(feat) for reg, feat in zip(self.calorie_reg, y[1])]
        food_cls = [cls(feat) for cls, feat in zip(self.food_cls, y[1])]
        
        return {
            'detection': y[0],
            'calorie': calorie_pred,
            'food_cls': food_cls
        }
