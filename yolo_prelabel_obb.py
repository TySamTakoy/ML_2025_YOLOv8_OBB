from ultralytics import YOLO
import os
from tqdm import tqdm

# === Настройки ===
model_path = r'E:/python/PythonProject/runs/obb/train9/weights/best.pt'  # путь к весам модели
img_dir = r'E:/python/PythonProject/.venv/v1-20000-out-of-89434/images'                # путь к папке с изображениями
output_dir = r'E:/python/PythonProject/.venv/prelabels_yolo_obb'               # папка для сохранения аннотаций
device = 0  # 0 — GPU, 'cpu' — если без CUDA

# === Инициализация ===
model = YOLO(model_path)
os.makedirs(output_dir, exist_ok=True)

# === Предсказание ===
for img_name in tqdm(os.listdir(img_dir)):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(img_dir, img_name)
    results = model.predict(img_path, device=device)

    for result in results:
        obb_preds = result.obb  # .obb содержит: class x1 y1 x2 y2 x3 y3 x4 y4
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)

        with open(txt_path, 'w') as f:
            for row in obb_preds.cpu().numpy():
                class_id = int(row[0])
                coords = ' '.join(f'{v:.6f}' for v in row[1:])
                f.write(f'{class_id} {coords}\n')
