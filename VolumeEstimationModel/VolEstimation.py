import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob

# Load pre-trained Mask R-CNN model
def load_maskrcnn():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# Preprocessing functions
def preprocess_midas(image, target_size=384):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(target_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def postprocess_depth(depth):
    depth = depth.squeeze().cpu().numpy()
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

def get_segmentation_mask(model, image, original_size):
    # Preprocess image for Mask R-CNN
    img_tensor = F.to_tensor(image).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Get the first prediction
    pred = predictions[0]
    
    # Find the mask with highest score
    if len(pred['scores']) > 0:
        # Get highest scoring mask
        best_idx = torch.argmax(pred['scores']).item()
        mask = pred['masks'][best_idx, 0].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)
        # Resize to original dimensions
        mask = cv2.resize(mask, original_size)
        return mask
    else:
        # Return empty mask if no objects detected
        return np.zeros(original_size[::-1], dtype=np.uint8)

def estimate_volume(seg_mask, depth_map):
    # Calculate area and average depth
    area = np.sum(seg_mask)
    if area == 0:
        return 0.0  # No food detected
    
    # Get depth values only in the food region
    depth_values = depth_map[seg_mask == 1]
    avg_depth = np.mean(depth_values)
    
    # Calculate thickness (standard deviation of depth)
    thickness = np.std(depth_values) if len(depth_values) > 0 else 0.01
    
    # Volume estimation formula (heuristic)
    volume = area * avg_depth * thickness
    return volume

def main(image_path):
    # Load models
    maskrcnn = load_maskrcnn()
    midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
    
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size  # (width, height)
    
    # Get segmentation mask
    seg_mask = get_segmentation_mask(maskrcnn, original_image, original_size)
    
    # Get depth estimation
    midas_input = preprocess_midas(original_image)
    with torch.no_grad():
        depth = midas(midas_input)
    depth_map = postprocess_depth(depth)
    depth_map = cv2.resize(depth_map, original_size)
    
    # Estimate volume
    volume = estimate_volume(seg_mask, depth_map)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Overlay segmentation mask
    masked_image = np.array(original_image)
    masked_image[seg_mask == 1] = (masked_image[seg_mask == 1] * 0.7 + 
                                  np.array([255, 0, 0]) * 0.3).astype(np.uint8)
    axes[1].imshow(masked_image)
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    axes[2].imshow(depth_map, cmap='jet')
    axes[2].set_title('Depth Map')
    axes[2].axis('off')
    
    plt.suptitle(f'Estimated Volume: {volume:.2f} (arbitrary units)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return volume

def batch_process(food_class, num_images=5):
    """Process multiple images from a food class"""
    base_path = f"/content/drive/MyDrive/CalEstDS/food-101-resized/{food_class}/test"
    images = glob.glob(os.path.join(base_path, "*.jpg"))[:num_images]
    
    volumes = []
    for img_path in images:
        print(f"Processing {os.path.basename(img_path)}...")
        try:
            vol = main(img_path)
            volumes.append(vol)
            print(f"Estimated Volume: {vol:.2f} arbitrary units\n")
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            volumes.append(0.0)
    
    return volumes

if __name__ == "__main__":
    # Mount Google Drive (if in Colab)
    if 'google.colab' in str(get_ipython()):
        from google.colab import drive
        drive.mount('/content/drive')
    
    # Example usage for single image
    image_path = "/content/drive/MyDrive/CalEstDS/food-101-resized/apple_pie/test/1005649.jpg"
    estimated_volume = main(image_path)
    print(f"Estimated Volume: {estimated_volume:.2f} arbitrary units")
    
    # Example batch processing
    pizza_volumes = batch_process("pizza", num_images=20)
