# ML_2025_YOLOv8_OBB

XRay Baggage Detection With YOLOv8 OBB

## XRay Baggage Dataset Link:

https://www.kaggle.com/datasets/muhammadzakria2001/baggage-xray-threats?select=Augmented_Dataset

Структура датасета следующая:
--	GDXray (рентгеновские снимки багажа в серо-черных тонах)
    --	images
    --	masks
--	PIDRAY (известный датасет под обучение нейронных сетей для детектирования потенциально нежелательно объектов)
    --	images
    --	masks
--	Sixray (шестицветный рентгеновский снимок)
    --	images
    --	masks


## Конфигурационный файл dataset-v1.yaml

Конфигурационный YAML-файл, в котором указаны пути к папкам и список классов 

Классы:
    0: ELECTRONICS
    1: EXPLOSIVE
    2: HAZARDOUS_LIQUID
    3: INSTRUMENT
    4: METAL
    5: SHARP_OBJECT
    6: WEAPON
    7: WIRE
    8: MEDICINE

# Обучение CNN

## Запуск предобучения на размеченных 4000 изображениях

Обучение на GPU

```
yolo detect train model=yolov8n-obb.pt data=dataset-v1.yaml epochs=50 imgsz=640 device=0
```

## Аннотация 11000 изображениях на основе предобученной модели YOLOv8 OBB











