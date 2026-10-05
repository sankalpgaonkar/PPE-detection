# 🦺 Personal Protective Equipment (PPE) Detection System

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ppe-web-app.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end computer vision solution engineered for real-time Personal Protective Equipment (PPE) detection and workplace safety compliance monitoring. Powered by a fine-tuned **YOLOv8** deep learning model and deployed as an interactive **Streamlit** web application.

---

## 🌐 Live Web Application

Experience the live app in action directly in your browser:
👉 **[PPE Detector Web App on Streamlit Cloud](https://ppe-web-app.streamlit.app/)**

![PPE Detection Interface](https://github.com/user-attachments/assets/d849947c-4299-49e2-933b-9bd0a308a1d9)

---

## 🎯 Key Features

- **Multi-Modal Detection**:
  - 🖼️ **Image Detection**: Upload images (`.jpg`, `.jpeg`, `.png`) for instant detection with bounding boxes and confidence scores.
  - 🎥 **Video Processing**: Process pre-recorded videos (`.mp4`, `.avi`, `.mov`) frame-by-frame with downloadable annotated video outputs.
  - 📹 **Live Webcam & Feeds**: Real-time video stream detection for on-site monitoring.
- **8 Distinct Safety Classes**:
  - `Boots` 🥾
  - `Ear-protection` 🎧
  - `Glass` (Safety Goggles) 🥽
  - `Glove` 🧤
  - `Helmet` ⛑️
  - `Mask` 😷
  - `Person` 🚶
  - `Vest` 🦺
- **Configurable Inference**:
  - Adjustable confidence thresholds slider (0.10 to 1.00).
  - Distinct color-coded bounding boxes per PPE class for clear visual differentiation.
- **Compliance & Metrics Dashboard**:
  - Detection summary tables showing counts per detected equipment class.
  - Real-time timestamped event logs.
- **Export Capabilities**:
  - Processed video download capability directly from the browser.

---

## 🎨 Color-Coded Class Mapping

| Class Name | Label | Color Indicator | BGR Hex Code |
| :--- | :--- | :---: | :--- |
| **Boots** | Footwear Protection | 🟧 Orange | `#FFA500` |
| **Ear-protection** | Hearing Safety | 🟪 Purple | `#9B59B6` |
| **Glass** | Eye Protection / Goggles | 🟦 Cyan | `#00BCD4` |
| **Glove** | Hand Protection | 🟨 Yellow | `#F1C40F` |
| **Helmet** | Head Protection | 🟩 Green | `#2ECC71` |
| **Mask** | Respiratory Protection | 🟪 Pink | `#FF69B4` |
| **Person** | Individual / Worker | 🟥 Red | `#E74C3C` |
| **Vest** | High-Visibility Vest | 🟦 Blue | `#3498DB` |

---

## 📁 Repository Structure

```plaintext
PPE-detection/
├── app.py                      # Main Streamlit web application
├── model.py                    # Inference preprocessing and helper routines
├── best.pt                     # Fine-tuned YOLOv8 model weights
├── PPE_Notebook--Part-1.ipynb  # End-to-end model training & evaluation notebook
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── logo.png                    # Project branding logo
├── logo_small.png              # Navbar icon branding
├── output/                     # Output directory for processed videos
└── safety-1/                   # Dataset configuration & splits
    ├── data.yaml               # YOLO dataset configuration file
    ├── README.dataset.txt      # Roboflow dataset details
    ├── README.roboflow.txt     # Roboflow export metadata
    ├── train/                  # Training set (images & labels)
    ├── valid/                  # Validation set (images & labels)
    └── test/                   # Test set (images & labels)
```

---

## 📊 Dataset & Model Training

The model was trained on the Roboflow Safety Gear Dataset (`safety-pyazl-6lzii`), formatted in YOLOv8 standard annotation style.

- **Dataset Source**: [Roboflow Universe - Safety Dataset](https://universe.roboflow.com/badminton-szmsf/safety-pyazl-6lzii/dataset/1)
- **Model Architecture**: Ultralytics YOLOv8
- **Fine-Tuned Weights**: Included in [`best.pt`](best.pt)
- **Training Pipeline**: Detailed step-by-step training process, loss curves, and evaluation metrics are preserved in [`PPE_Notebook--Part-1.ipynb`](PPE_Notebook--Part-1.ipynb).

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

- Python 3.8 or higher
- `git` installed
- (Optional) CUDA-compatible GPU for accelerated inference

### 2. Clone the Repository

```bash
git clone https://github.com/sankalpgaonkar/PPE-detection.git
cd PPE-detection
```

### 3. Create a Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch the Streamlit App

```bash
streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your browser.

---

## 🛠️ Tech Stack

- **Deep Learning**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/)
- **Computer Vision**: [OpenCV](https://opencv.org/), [Pillow (PIL)](https://python-pillow.org/)
- **Frontend / Deployment**: [Streamlit](https://streamlit.io/)
- **Dataset Management**: [Roboflow](https://roboflow.com/)

---

## 👤 Author

Developed by **Sankalp Gaonkar**  
- **GitHub**: [@sankalpgaonkar](https://github.com/sankalpgaonkar)
- **Live Application**: [PPE Detection App](https://ppe-web-app.streamlit.app/)

---

## 📄 License

The dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
This repository and code are made available for educational and commercial workplace safety applications.
