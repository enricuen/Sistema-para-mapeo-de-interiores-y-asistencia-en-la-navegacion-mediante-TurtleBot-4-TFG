# -*- coding: utf-8 -*-

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="xPHsHiUNZMnqk24t67JV")
project = rf.workspace("germantrafficsigns-zl3mn").project("no-entry-ug5iz")
version = project.version(2)
dataset = version.download("yolov8")

!pip install ultralytics
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

!yolo task=detect mode=train model=yolov8s.pt data=/content/No-Entry-2/data.yaml epochs=500 imgsz=320 plots=True

"""Validación"""

!yolo val model="/content/runs/detect/train/weights/best.pt" data=/content/No-Entry-2/data.yaml


"""Predicción"""

!yolo predict model="/content/runs/detect/train/weights/best.pt" source="/content/image4.jpg" imgsz=320