# -*- coding: utf-8 -*-

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="xPHsHiUNZMnqk24t67JV")
project = rf.workspace("makineproje").project("no-entry-ysuzn")
version = project.version(1)
dataset = version.download("yolov8")

!pip install ultralytics
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

import os
import random
import shutil
import yaml

# Rutas base
dataset_path = '/content/no-entry-1'
train_img_path = os.path.join(dataset_path, 'train/images')
train_lbl_path = os.path.join(dataset_path, 'train/labels')
val_img_path = os.path.join(dataset_path, 'valid/images')
val_lbl_path = os.path.join(dataset_path, 'valid/labels')
yaml_path = os.path.join(dataset_path, 'data.yaml')

# Crear directorios de validación si no existen
os.makedirs(val_img_path, exist_ok=True)
os.makedirs(val_lbl_path, exist_ok=True)

# Obtener lista de imágenes en train
images = [f for f in os.listdir(train_img_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Seleccionar 100 imágenes al azar para validación
num_to_move = min(100, len(images))
val_samples = random.sample(images, num_to_move)

print(f"Moviendo {num_to_move} archivos de train a valid...")

for img_name in val_samples:
    # Mover imagen
    shutil.move(os.path.join(train_img_path, img_name), os.path.join(val_img_path, img_name))

    # Mover etiqueta correspondiente (mismo nombre y extensión .txt)
    lbl_name = os.path.splitext(img_name)[0] + '.txt'
    src_lbl = os.path.join(train_lbl_path, lbl_name)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, os.path.join(val_lbl_path, lbl_name))

# Actualizar el archivo data.yaml
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# Asegurar que las rutas apunten correctamente
data['train'] = os.path.join(dataset_path, 'train/images')
data['val'] = os.path.join(dataset_path, 'valid/images')

with open(yaml_path, 'w') as f:
    yaml.dump(data, f)

print("Dataset listo y data.yaml actualizado.")

!yolo task=detect mode=train model=yolov8n.pt data=/content/no-entry-1/data.yaml epochs=40 imgsz=320 plots=True

"""Validación"""

!yolo val model="/content/runs/detect/train/weights/best.pt" data=/content/no-entry-1/data.yaml

"""Predicción"""

!yolo predict model="/content/runs/detect/train/weights/best.pt" source='/content/aula2.png' imgsz=320