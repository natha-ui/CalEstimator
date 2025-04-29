from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils.ops import scale_boxes

class YOLOv10sCaloriePredictor(BasePredictor):
    def postprocess(self, preds, img, orig_imgs):
        # Process detections
        dets = non_max_suppression(
            preds['detection'],
            self.args.conf,
            self.args.iou,
            multi_label=True
        )
        
        results = []
        for i, det in enumerate(dets):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if len(det) == 0:
                results.append(Results(orig_img, names=self.model.names))
                continue
                
            # Scale boxes
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], orig_img.shape)
            
            # Add calorie estimates
            for *xyxy, conf, cls in det:
                # Get corresponding calorie prediction
                region_idx = self.find_region(xyxy, preds['calorie'][i])
                calories = preds['calorie'][i][region_idx].item()
                
                # Store results
                results.append(CalorieResults(
                    orig_img,
                    path=self.batch[0][i],
                    names=self.model.names,
                    boxes=det,
                    calories=calories
                ))
        return results
