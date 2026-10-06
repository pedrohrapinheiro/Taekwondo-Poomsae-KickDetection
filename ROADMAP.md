# Project Roadmap: Evolution to Advanced Kick Analysis & Poomsae Detection

## Current State (Baseline)
- **Capability**: Binary Classification (Kick vs. No-Kick).
- **Architecture**: YOLOv8-Pose $\to$ Hip-Centered Normalization $\to$ 30-Frame Sliding Window $\to$ MLP.
- **Key Features**: Relative coordinates, joint angles, and temporal aggregation based on Haider et al. (2023).

---

## 🗺️ Evolution Roadmap

### Phase 1: Multi-Class Action Recognition (The "What")
**Goal**: Identify the specific type of kick (e.g., Front Kick, Roundhouse, Side Kick).
- [ ] **Annotator Update**: Modify `annotator.py` to allow selecting a specific kick label per interval instead of just 0/1.
- [ ] **Dataset Expansion**: Collect and label diverse examples of different kick types.
- [ ] **Model Architecture Update**:
    - Change output layer from `Dense(1, sigmoid)` $\to$ `Dense(N, softmax)`.
    - Update loss function to `categorical_crossentropy`.
- [ ] **Evaluation**: Use a Confusion Matrix to identify which kicks the model confuses with each other.

### Phase 2: Biomechanical Metrics (The "How")
**Goal**: Quantify the performance of the kick (Speed and Estimated Power).
- [ ] **Velocity Calculation**: Implement linear velocity tracking for the ankle keypoint ($\Delta \text{position} / \Delta \text{time}$).
- [ ] **Power Estimation**: Develop a formula to estimate kinetic energy based on ankle velocity and estimated leg mass.
- [ ] **Real-time Overlay**: Update `predict_realtime.py` to display peak velocity and power on screen during the kick.

### Phase 3: Poomsae Analysis Mode (The "Quality")
**Goal**: Detect errors in Poomsae execution by comparing user movement to a "Gold Standard".
- [ ] **Master Sequence Recording**: Create a "perfect" dataset of a specific Poomsae to serve as the reference.
- [ ] **Temporal Alignment (DTW)**: Implement **Dynamic Time Warping (DTW)** to align the user's performance with the master sequence regardless of speed.
- [ ] **Error Detection Logic**:
    - **Pose Error**: Compare joint angles between user and master ($\text{Deviation} > \text{Threshold}$).
    - **Balance Error**: Monitor hip-center stability during static positions.
    - **Timing Error**: Detect delays or rushed transitions between movements.
- [ ] **Feedback System**: Create a visual or text-based feedback system (e.g., "Lift your knee higher").

---

## 🛠️ Technical Requirements for Future Phases
- **DTW Library**: `fastdtw` or `dtaidistance`.
- **More Data**: A wider variety of kick types and expert-level Poomsae recordings.
- **Advanced Visualization**: Possibly moving to a more complex UI to show side-by-side comparisons of the user vs. the master.
