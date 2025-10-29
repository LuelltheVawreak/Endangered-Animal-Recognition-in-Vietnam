from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(
    data="/content/Animal_yolo-1/data.yaml",
    epochs=50,
    imgsz=320,
    batch=16,
    name="Animal_detect_tranfer"
)
