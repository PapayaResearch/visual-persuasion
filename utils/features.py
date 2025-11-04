import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def compute_ssim(image_a: Image, image_b: Image) -> float:
    image_a_np = np.array(image_a.resize((256, 256)).convert("L"))
    image_b_np = np.array(image_b.resize((256, 256)).convert("L"))
    ssim_value = ssim(image_a_np, image_b_np)
    print(f"SSIM: {ssim_value}")
    return ssim_value


def count_objects(image: Image) -> int:
    model = YOLO("yolov8n.pt")
    results = list(model.predict(image))
    n_results = len(results[0].boxes)
    print(f"Number of detection results: {n_results}")
    return n_results


def estimate_mean_depth_entropy(image: Image) -> np.ndarray:
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    model.eval()
    input_batch = transforms.small_transform(np.array(image))
    with torch.no_grad():
        prediction = model(input_batch)
        depth = prediction.squeeze().cpu().numpy()
    depth_entropy = -depth * np.log(depth + 1e-8)
    mean_entropy = np.mean(depth_entropy)
    print(f"Mean Depth Entropy: {mean_entropy}")
    return mean_entropy
