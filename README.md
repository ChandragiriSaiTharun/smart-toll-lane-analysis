# 🚗 Smart Toll Lane Analysis System

## 📌 Overview

This project is a **real-time traffic analysis system** that detects vehicles, analyzes multiple toll lanes, and recommends the best lane based on estimated waiting time.

It uses **YOLOv8 for vehicle detection**, tracking for accurate counting, and a **Streamlit dashboard** for live visualization.

---

## 🚀 Features

* 🚗 Vehicle Detection using YOLOv8
* 🎯 Multi-lane ROI (Region of Interest) analysis
* 🔁 Vehicle Tracking (avoids duplicate counting)
* ⏱️ Waiting Time Estimation using vehicle weights
* 📊 Real-time dashboard using Streamlit
* 🧠 Smart lane recommendation system
* 📈 Performance metrics & comparison analysis

---

## 🛠️ Tech Stack

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* NumPy
* Streamlit

---

## 📂 Project Structure

```
smart-toll-lane-analysis/
│
├── src/                          # Core modules
│   ├── vehicle_detector.py       # YOLOv8 vehicle detection
│   ├── vehicle_tracker.py        # Tracking with unique IDs
│   ├── lane_analyzer.py          # Lane assignment & analysis
│   ├── realtime_processor.py     # Main processing pipeline
│   ├── metrics_calculator.py     # Performance metrics
│   ├── baseline_comparison.py    # Comparison analysis
│   ├── config.py                 # Configuration settings
│   └── lane_drawer.py            # ROI lane drawing tool
│
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Ignored files
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/smart-toll-lane-analysis.git
cd smart-toll-lane-analysis
```

### 2. Create virtual environment

```
python -m venv project_env
source project_env/bin/activate   # Linux/Mac
project_env\Scripts\activate      # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```
streamlit run app.py
```

---

## 🤖 Model Information

YOLO model weights (`.pt` files) are **not included** in this repository due to size limitations.
They will be **automatically downloaded** when running the project.

---

## 📊 How It Works

1. Detect vehicles using YOLOv8
2. Track vehicles using unique IDs
3. Assign vehicles to lanes using ROI polygons
4. Calculate lane-wise load using vehicle weights
5. Estimate waiting time for each lane
6. Recommend the best lane

---

## 💡 Future Improvements

* Real-time waiting time using entry-exit tracking
* Advanced tracking (DeepSORT / ByteTrack)
* Integration with real toll systems
* Improved accuracy with larger YOLO models
