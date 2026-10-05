# Taekwondo Poomsae Kick Detection 🥋

An AI-powered system designed to detect and classify Taekwondo kicks in real-time using computer vision and deep learning. This project transforms raw video feeds into skeletal keypoints and uses a temporal Neural Network to identify the specific dynamics of a kick.

## 🚀 Features
- **Real-time Detection**: Analyzes live camera feeds to identify kicks.
- **Skeletal Tracking**: Uses YOLOv8-Pose for high-accuracy human pose estimation.
- **Temporal Analysis**: Implements a sliding window approach to analyze movement over time rather than single frames.
- **Research-Grounded**: Methodology aligned with Human Activity Recognition (HAR) research (Haider et al., 2023), utilizing hip-centered normalization and joint angle dynamics.

## 📂 Project Structure
- `Taekwondo_Kick_Dataset_Creator/`: Tools for generating the training dataset.
    - `annotator.py`: Visual tool to mark kick start/end frames in videos.
    - `main.py`: Processes videos and extracts skeletal features into a CSV.
- `Taekwondo_Model_Creator/`: AI training pipeline.
    - `model.ipynb`: Jupyter notebook for feature engineering, MLP training, and evaluation.
- `Taekwondo_Kick_Pose_Predict/`: Real-time deployment.
    - `predict_realtime.py`: The main application for live kick detection.
- `requirements.txt`: Centralized list of all project dependencies.

## 🛠️ Technical Stack
- **Language**: Python 3.10+
- **Pose Estimation**: Ultralytics YOLOv8-Pose
- **Deep Learning**: TensorFlow / Keras (MLP Architecture)
- **Data Processing**: Pandas, NumPy, Scikit-Learn
- **Computer Vision**: OpenCV

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd TaekwondoProject
```

### 2. Create and activate a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

## 📈 Workflow

### Step 1: Create the Dataset
1. Place your videos in `Taekwondo_Kick_Dataset_Creator/videos/`.
2. Run `python Taekwondo_Kick_Dataset_Creator/annotator.py` to mark the kick intervals.
3. Run `python Taekwondo_Kick_Dataset_Creator/main.py` to generate `poses_dataset.csv`.

### Step 2: Train the Model
1. Open `Taekwondo_Model_Creator/model.ipynb` in Jupyter or VS Code.
2. Run all cells to train the MLP and save the `kick_detection_model.h5` and `scaler.pkl`.

### Step 3: Run Real-time Detection
```powershell
python Taekwondo_Kick_Pose_Predict/predict_realtime.py
```

## 🔬 Methodology
The system follows a sophisticated pipeline to ensure accuracy:
1. **Normalization**: All keypoints are translated relative to the hip center to ensure the model works regardless of the person's position in the frame.
2. **Feature Extraction**: Calculates joint angles (knees/hips) and foot distances to capture the physics of a kick.
3. **Temporal Windowing**: Instead of analyzing a single frame, the model looks at a window of 30 frames, allowing it to recognize the *motion* of the kick.
4. **Strict Visibility Filter**: Only predicts when key points (Hips and Ankles) are clearly visible to prevent false positives.
