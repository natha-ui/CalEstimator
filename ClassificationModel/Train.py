from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.cfg import get_cfg
import os

import torch

yaml_path = 'yolov10s-cls.yaml'
data_path = '../food-101'
num_classes = len([name for name in os.listdir(f"{data_path}/train") if os.path.isdir(os.path.join(f"{data_path}/train", name))])
imgsz = 256
epochs = 50
batch = 32
device = 0 if torch.cuda.is_available() else 'cpu'

print(num_classes)

cfg = get_cfg(yaml_path)
cfg.model.nc = num_classes
cfg.task = 'classify'
cfg.imgsz = imgsz
cfg.epochs = epochs
cfg.batch = batch
cfg.data = data_path
cfg.device = device

# model = ClassificationModel(cfg)

# checkpoint = torch.load('yolov10s.pt', map_location='cpu')
# model.model.load_state_dict(checkpoint['model'].float().state_dict(), strict=False)

trainer = ClassificationTrainer(model=yaml_path, overrides=cfg)
trainer.train()
