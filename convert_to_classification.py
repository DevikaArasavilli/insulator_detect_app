import os
import shutil

base_folder = 'insulator-faults-detection-1'
output_folder = 'dataset'

os.makedirs(f"{output_folder}/normal", exist_ok=True)
os.makedirs(f"{output_folder}/faulty", exist_ok=True)

def convert_images(split):
    labels_path = os.path.join(base_folder, split, 'labels')
    images_path = os.path.join(base_folder, split, 'images')

    for label_file in os.listdir(labels_path):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(labels_path, label_file)
        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_path, image_file)

        if not os.path.exists(image_path):
            continue

        with open(label_path, 'r') as f:
            labels = f.readlines()

        destination = "normal"
        for line in labels:
            if not line.strip():
                continue
            class_id = int(line.split()[0])
            if class_id != 0:
                destination = "faulty"
                break

        shutil.copy(image_path, os.path.join(output_folder, destination, image_file))

convert_images('train')
convert_images('valid')

print("✅ Conversion complete! Check the 'dataset' folder.")
