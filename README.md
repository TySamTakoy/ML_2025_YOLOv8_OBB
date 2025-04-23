# ML_2025_YOLOv8_OBB

XRay Baggage Detection With YOLOv8 OBB

## XRay Baggage Dataset Link:

https://www.kaggle.com/datasets/muhammadzakria2001/baggage-xray-threats?select=Augmented_Dataset

Структура датасета следующая:

```
dataset/
├── GDXray/  (рентгеновские снимки багажа в серо-черных тонах)
│   ├── images/
│   └── masks/
├── PIDRAY/  (известный датасет под обучение нейронных сетей для детектирования потенциально нежелательно объектов)
│   ├── images/
│   └── masks/
└── Sixray/  (шестицветный рентгеновский снимок)
    ├── images/
    └── masks/
```

## Конфигурационный файл dataset-v1.yaml

Конфигурационный YAML-файл, в котором указаны пути к папкам и список классов 

Классы:

    0. ELECTRONICS
    1. EXPLOSIVE
    2. HAZARDOUS_LIQUID
    3. INSTRUMENT
    4. METAL
    5. SHARP_OBJECT
    6. WEAPON
    7. WIRE
    8. MEDICINE

# Обучение CNN

## Запуск предобучения на размеченных 4000 изображениях

```
yolo detect train model=yolov8n-obb.pt data=dataset-v1.yaml epochs=50 imgsz=640 device=0
```

### Обучение на GPU
Файл ```cuda-check.py``` позволяет проверить наличие видеоадаптера, его id и текущую версию CUDA для работы в обучении


## Аннотация 11000 изображениях на основе предобученной модели YOLOv8 OBB

Рабочий файл ```yolo_prelabel_obb.py```

## Обучение модели на 15000 размеченных изображениях

```
yolo task=obb mode=train model=runs/obb/train9/weights/last.pt data=dataset-v1.yaml epochs=100 imgsz=640 device=0
```

# Запуск готового интерфейса для просмотра работы модели в режиме реального времени

```THROUGH-SCREEN-CAPTURE-PREDICT.PY```

Запускает окно, которое для области экрана 800x600 (отступы вправо и низ на 100) в режиме реального времени будет
Детектировать объекты на изображении, т.е. просто в эту область переводится окно с изображением и детектируется.

На ```exmpl1.png``` и ```exmpl2.png``` слева открыто окно «Фотографии» — стандартное приложение для просмотра фотографий в Windows 11, справа – реализованное решение для просмотра изображений и определенных моделью аннотаций в режиме реального времени:
