import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# Загружаем модель YOLO
model = YOLO(r"E:\python\PythonProject\runs\obb\train10\weights\best.pt")

# Область экрана (в координатах монитора)
monitor = {
    "top": 100,      # сместить на нужную позицию по вертикали
    "left": 100,     # сместить на нужную позицию по горизонтали
    "width": 800,
    "height": 600
}

sct = mss()

while True:
    # Снимок области экрана
    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # Предсказание YOLO
    results = model.predict(source=frame, conf=0.4, verbose=False)

    for r in results:
        obbs = r.obb.xyxyxyxy.cpu().numpy()
        confs = r.obb.conf.cpu().numpy()
        classes = r.obb.cls.cpu().numpy()
        names = r.names

        for obb, conf, cls_id in zip(obbs, confs, classes):
            color = tuple(np.random.randint(100, 255, size=3).tolist())
            pts = obb.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            label = f"{names[int(cls_id)]} {conf:.2f}"
            text_pos = tuple(pts[0][0])
            cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO OBB - Screen Capture", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
        break

cv2.destroyAllWindows()
