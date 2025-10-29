from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(
    data="/content/Animal_yolo-1/data.yaml",
    epochs=50,
    imgsz=320,
    batch=16,
    name="Animal_detect_tranfer"
)

from ultralytics import YOLO
model = YOLO("yolo11m.pt")
print(model.info()) 


import pandas as pd

yolo = pd.read_csv(yolo_path)
print("=== YOLO11m metrics ===")
print(yolo.columns.tolist())
print(yolo.tail(3))
