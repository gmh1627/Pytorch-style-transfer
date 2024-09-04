# utils.py

import numpy as np
from PIL import Image

def load_image(image_path, size=None, scale=None):
    img = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    elif scale:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.LANCZOS)
    return img

def save_image(tensor, path):
    img = tensor.clone().detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)