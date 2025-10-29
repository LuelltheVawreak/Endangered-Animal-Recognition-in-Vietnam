import pandas as pd
import matplotlib.pyplot as plt


yolo_path = "runs/detect/Animal_detect_tranfer/results.csv"
mobilenet_path = "runs/mobilenet_results.csv"


yolo = pd.read_csv(yolo_path)
mobile = pd.read_csv(mobilenet_path)

print("YOLO columns:", list(yolo.columns))
print("MobileNet columns:", list(mobile.columns))

plt.figure(figsize=(10,6))
plt.plot(yolo["epoch"], yolo["metrics/mAP50(B)"], label="YOLO11m (mAP@0.5)", linewidth=2)
plt.plot(mobile["epoch"], mobile["val_accuracy"], label="MobileNetV3 (Val Accuracy)", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Performance")
plt.title("So sánh hiệu suất: YOLO11m vs MobileNetV3")
plt.legend()
plt.grid(True)
plt.show()


#
plt.figure(figsize=(10,6))
plt.plot(yolo["epoch"], yolo["train/box_loss"], label="YOLO11m Box Loss")
plt.plot(yolo["epoch"], yolo["train/cls_loss"], label="YOLO11m Cls Loss")
plt.plot(mobile["epoch"], mobile["loss"], label="MobileNetV3 Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("So sánh Loss giữa YOLO11m và MobileNetV3")
plt.legend()
plt.grid(True)
plt.show()


#
plt.figure(figsize=(10,6))
plt.plot(yolo["epoch"], yolo["metrics/precision(B)"], label="YOLO11m Precision", linestyle="--")
plt.plot(yolo["epoch"], yolo["metrics/recall(B)"], label="YOLO11m Recall", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision và Recall của YOLO11m")
plt.legend()
plt.grid(True)
plt.show()
