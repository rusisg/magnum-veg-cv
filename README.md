# Распознавание овощей на полке магазина (Track A — Computer Vision)

Решение тестового задания на позицию **Junior Computer Vision / Machine Learning Engineer (Magnum)**.  
Проект включает криминалистический аудит датасета на утечки данных (`MD5`, `dHash`/`pHash`, видеосерии), построение чистого группового разбиения (`StratifiedGroupKFold`), обучение лёгкого классификатора (`MobileNetV3-Large`) с аугментациями под условия торгового зала, верификацию внимания через **Grad-CAM**, калибровку температуры и порога для класса **`unknown`** (OOD), экспорт в **ONNX** и полностью автономный CPU-скрипт инференса `predict.py`.

---

## Быстрый старт: запуск `predict.py` (CPU, без интернета)

Скрипт инференса использует **только** `onnxruntime` (`CPUExecutionProvider`), `numpy` и `Pillow` (плюс стандартную библиотеку Python) — установка занимает несколько секунд, не требует PyTorch/CUDA и не выполняет сетевых запросов.

### 1. Установка минимальных CPU-зависимостей
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Запуск предсказания на папке с изображениями
```bash
python predict.py --input путь/к/папке --output preds.csv
```
* Поддерживаются форматы `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` (регистронезависимо, включая вложенные подпапки).
* Применяется `PIL.ImageOps.exif_transpose` до конвертации в `RGB` (корректная ориентация мобильных фото с EXIF-тегами).
* Повреждённые файлы не роняют процесс: логируется `WARNING`, файл безопасно пропускается.

### 3. Формат выходного файла `preds.csv`
```csv
filename,label,confidence
IMG_0001.jpg,Carrot,0.9603
IMG_0002.jpg,unknown,0.3198
```
Допустимые значения `label` (строгое написание датасета): `Capsicum`, `Carrot`, `Cucumber`, `Potato`, `Raddish`, `Squash`, `unknown`.

---

## Воспроизведение EDA, обучения и экспериментов с нуля

### 1. Установка полного стека обучения и анализа
```bash
pip install -r requirements-train.txt
```

### 2. Подготовка данных, аудит утечек и групповое разбиение (Phase 1)
Скрипт проверяет наличие `preethi866/vegetable-dataset` в `data/` (при отсутствии скачивает через Kaggle CLI), проводит аудит `Dataset/` и `final_dataset/` и строит leak-free сплиты `70 / 15 / 15`:
```bash
python src/eda_and_split.py --data-dir data/ --out-dir splits/
```

### 3. Обучение, оценка, Grad-CAM, калибровка `unknown` и экспорт ONNX (Phases 2–3)
```bash
python src/train.py --splits-dir splits/ --epochs 20 --batch-size 64 --seed 42
python src/evaluate.py --checkpoint models/best_model.pt --test-csv splits/test.csv
python src/gradcam.py --checkpoint models/best_model.pt --test-csv splits/test.csv
python src/calibrate_ood.py --checkpoint models/best_model.pt --val-csv splits/val.csv
python src/export_onnx.py --checkpoint models/best_model.pt --out models/veg_classifier.onnx
```
Интерактивный отчёт со всеми графиками и таблицами доступен в `notebooks/01_eda_and_experiments.ipynb`.

---

## Ключевые результаты (на очищенном leak-free тестовом сплите)

| Метрика / Параметр | Значение |
| :--- | :--- |
| **Архитектура** | `MobileNetV3-Large` (`IMAGENET1K_V2` pretrained) |
| **Обнаруженная утечка в исходном `final_dataset/test`** | **96.25%** (`769 / 799`: `717` точных MD5 + `52` near-dup) |
| **Утечка в нашем групповом разбиении (`splits/*.csv`)** | **0.00%** (`0` общих MD5, кластеров и `dHash/pHash <= 6`) |
| **Test Accuracy (leak-free `splits/test.csv`, N=598)** | **0.9950** (`99.50%`, `595 / 598`) |
| **Test Macro-F1 (leak-free `splits/test.csv`)** | **0.9950** (`Store-Stress Macro-F1`: **0.9959**) |
| **Калибровка `unknown` (Temperature Scaling + MSP)** | `T = 0.7231`, `unknown_threshold = 0.8900` (`AUROC = 0.9990`) |
| **Время инференса 1 фото (PyTorch CPU)** | **39.63 ms** (`p50 = 39.30 ms`, `p95 = 41.08 ms`) |
| **Время инференса 1 фото (ONNX Runtime CPU)** | **2.60 ms** (`p50 = 2.54 ms`, `p95 = 3.03 ms` — ускорение **15.23x**) |
| **Размер модели (`models/veg_classifier.onnx`)** | **16.05 MB** (opset 17, `max |PT - ONNX| = 3.64e-06`) |

---

## Структура репозитория

```text
├── README.md
├── REPORT.md                         # Аналитический отчёт (<= 2 стр. PDF)
├── requirements.txt                  # Минимальные CPU-зависимости для predict.py
├── requirements-train.txt            # Полный стек для EDA, обучения и ноутбука
├── predict.py                        # Автономный CPU CLI-скрипт инференса
├── models/
│   ├── veg_classifier.onnx           # Экспортированная ONNX-модель (16.05 MB)
│   └── config.json                   # Классы, нормировка, temperature, unknown_threshold
├── splits/
│   ├── train.csv                     # 2,718 снимков (70.07%, 667 кластеров)
│   ├── val.csv                       #   563 снимка  (14.51%, 155 кластеров)
│   └── test.csv                      #   598 снимков (15.42%, 150 кластеров)
├── notebooks/
│   └── 01_eda_and_experiments.ipynb  # Выполненный воспроизводимый ноутбук
├── src/
│   ├── eda_and_split.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── calibrate_ood.py
│   └── export_onnx.py
└── assets/
    ├── class_distribution.png
    ├── duplicates_examples.png
    ├── confusion_matrix.png
    ├── gradcam_grid.png
    └── ood_threshold_curve.png
```

---

## Проверенные версии окружения

* **Python:** `3.11.17` (совместимо с `3.10+`)
* **Инференс (`requirements.txt`):** `onnxruntime==1.32.0`, `numpy==2.4.6`, `Pillow==13.0.0`
* **Обучение и EDA (`requirements-train.txt`):** `torch==2.15.0`, `torchvision==0.29.2`, `timm==1.0.31`, `albumentations==2.0.8`, `grad-cam==1.5.7`, `imagehash==4.3.2`, `scikit-learn==1.9.1`, `onnx==1.23.2`, `pandas==3.1.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`
