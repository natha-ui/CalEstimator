from ultralytics.models.yolo.detect import DetectionPredictor
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results

# Example calorie mapping (you can expand this)
CALORIE_MAP = {
    'apple': {'calories_per_item': 95},
    'banana': {'calories_per_item': 105},
    'orange': {'calories_per_item': 62},
    'pizza': {'calories_per_100g': 266},
    'burger': {'calories_per_100g': 295},
}

class YOLOv10DetectionPredictor(DetectionPredictor):
    def estimate_calories(self, label_name, box):
        # Simplistic calorie estimation
        # If per item calories exist
        if label_name in CALORIE_MAP:
            if 'calories_per_item' in CALORIE_MAP[label_name]:
                return CALORIE_MAP[label_name]['calories_per_item']
            elif 'calories_per_100g' in CALORIE_MAP[label_name]:
                # Estimate area as proxy for weight (very rough)
                box_area = (box[2] - box[0]) * (box[3] - box[1])  # x1, y1, x2, y2
                estimated_weight = box_area / 10000 * 150  # Scale factor
                calories = estimated_weight * CALORIE_MAP[label_name]['calories_per_100g'] / 100
                return round(calories, 1)
        return None

    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if preds.shape[-1] == 6:
            pass
        else:
            preds = preds.transpose(-1, -2)
            bboxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, preds.shape[-1] - 4)
            bboxes = ops.xywh2xyxy(bboxes)
            preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)

        preds = [p[mask[idx]] for idx, p in enumerate(preds)]

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred)

            # Add estimated calories
            calories = []
            for box in pred:
                label = int(box[5])
                label_name = self.model.names[label]
                calorie_est = self.estimate_calories(label_name, box[:4])
                calories.append((label_name, calorie_est))

            result.calories = calories  # Custom attribute
            results.append(result)

        return results
