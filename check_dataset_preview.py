import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def preview_images(folder, label):
    path = os.path.join("dataset", folder)
    images = os.listdir(path)
    print(f"{label} - {len(images)} images")
    sample = random.sample(images, min(5, len(images)))
    for img in sample:
        img_path = os.path.join(path, img)
        Image.open(img_path).show()

preview_images("normal", "Normal")
preview_images("faulty", "Faulty")
