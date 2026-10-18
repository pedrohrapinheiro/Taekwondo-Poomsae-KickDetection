# Taekwondo Kick Detection 🥋

An AI-powered system designed to detect whether a Taekwondo kick is happening in real-time using computer vision and deep learning. This project transforms raw video feeds into skeletal keypoints and uses a temporal Neural Network to classify movements as either "Kicking" or "No Kick".

## 🚀 Features
- **Real-time Binary Detection**: Identifies if a kick is currently being executed.
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

## 🖥️ Desktop App (Taekwondo Trainer)
All features in one window. Double-click **`run_app.bat`** or run:
```powershell
python Taekwondo_App/app.py
```
- **Kick Detection** — live webcam with skeleton overlay, KICKING/NO KICK status, kick counter and kick speed.
- **Poomsae Evaluation** — open a video or record one with the webcam (5 s countdown), choose the Taegeuk or let it auto-detect, then see the overall score, a per-move table with corrections, and the feedback video with the reference "ghost". Click a move to jump to it.

Recordings and feedback videos are saved in `Taekwondo_App/output/`.

## 🔬 Methodology
The system follows a sophisticated pipeline to ensure accuracy:
1. **Normalization**: All keypoints are translated relative to the hip center to ensure the model works regardless of the person's position in the frame.
2. **Feature Extraction**: Calculates joint angles (knees/hips) and foot distances to capture the physics of a kick.
3. **Temporal Windowing**: Instead of analyzing a single frame, the model looks at a window of 30 frames, allowing it to recognize the *motion* of the kick.
4. **Strict Visibility Filter**: Only predicts when key points (Hips and Ankles) are clearly visible to prevent false positives.

## 🥋 Poomsae Evaluator (Taegeuk 1–8)
Scores how correctly a Taegeuk poomsae is performed in a video by comparing it with a reference performance.

```powershell
python Taekwondo_Poomsae_Evaluator/evaluate_poomsae.py my_poomsae.mp4 --output-video feedback.mp4
```

- **Automatic form and camera detection**: identifies which Taegeuk (1–8) is performed and whether it was filmed from the front or the side.
- **Per-move score (0–100)** with corrections such as `right forearm is 45° off — raise it`, plus rhythm warnings (`too fast`/`too slow`).
- **Annotated video**: your skeleton (green, red where it deviates) with the reference pose overlaid as a white "ghost".
- Options: `--form N` and `--view frontal|lateral` to skip auto-detection, `--json report.json` to save the full report.

**Recording tips:** keep the camera still, film the whole body, and place the camera like the reference — facing the starting direction of the form (frontal) or from the side (lateral). Stance depth is only visible from the side.

**How it works:** poses are extracted with the same distance-invariant `PoseDetector` used for kick detection, normalized by hip center and torso length, and aligned to the reference with subsequence Dynamic Time Warping (so pauses before/after and a different tempo don't matter). The reference is split into moves at the pauses between techniques; each move is scored by its three worst-matching joints, and the final phase of each move is compared limb by limb for the corrections.

**Reference data:** `Taekwondo_Poomsae_Evaluator/references/` holds keypoints extracted from a dual-view (frontal/lateral) Taegeuk 1–8 dataset. To rebuild them, extract the dataset's `Data.zip` into `Taekwondo_Poomsae_Evaluator/reference_data/` and run `python Taekwondo_Poomsae_Evaluator/build_references.py`.

**Limitations:** there is a single reference performer per form, so body proportions and personal style differences also cost a few points; the scores are best used to find the moves that need work rather than as an official grade.
