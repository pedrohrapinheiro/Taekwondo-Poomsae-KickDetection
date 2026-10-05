import cv2 as cv
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import joblib
import os
import threading
import warnings

# Silenciar avisos do TensorFlow e sklearn para limpar o terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 0. VIDEO STREAM THREADING
# ==========================================
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv.VideoCapture(src)
        self.stream.set(cv.CAP_PROP_FRAME_WIDTH, 320)
        self.stream.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        self.T = threading.Thread(target=self.update, args=(), daemon=True)
        self.T.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# 1. GEOMETRY UTILS
# ==========================================
def calculate_angle(a, b, c):
    """Calcula o ângulo em graus entre três pontos (a, b, c) onde b é o vértice."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# ==========================================
# 2. LOAD MODELS AND ASSETS
# ==========================================
print("Loading models... please wait.")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n-pose.pt")
KICK_MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "kick_detection_model.h5")
SCALER_PATH = os.path.join(SCRIPT_DIR, "model", "scaler.pkl")

pose_model = YOLO(POSE_MODEL_PATH)
kick_model = tf.keras.models.load_model(KICK_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- TEMPORAL BUFFER FOR MLP ---
WINDOW_SIZE = 30
feature_buffer = deque(maxlen=WINDOW_SIZE)

# Voting Buffer
prediction_buffer = deque(maxlen=5) # Reduced from 15 to 5 for faster response
prev_angles = None

# Performance optimizations
FRAME_SKIP = 2 # Reduced from 5 to 2 for more frequent AI updates
frame_count = 0

# Initialize Threaded Stream
vs = VideoStream(src=0).start()

print("System Ready! Press 'q' to quit.")

while True:
    frame = vs.read()
    if frame is None:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        results = pose_model(frame, verbose=False)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            xy = results[0].keypoints.xy[0].cpu().numpy()
            conf = results[0].keypoints.conf[0].cpu().numpy()

            # Only focus on lower body points
            important_indices = [11, 12, 13, 14, 15, 16]
            conf = results[0].keypoints.conf[0].cpu().numpy()

            # STRICT VISIBILITY: Require Hips (11,12) AND at least one ankle (15 or 16)
            hips_visible = conf[11] > 0.5 and conf[12] > 0.5
            ankles_visible = conf[15] > 0.5 or conf[16] > 0.5
            legs_visible = hips_visible and ankles_visible

            current_kp = []
            for i in range(17):
                if i < len(xy):
                    current_kp.append([xy[i][0], xy[i][1], conf[i]])
                else:
                    current_kp.append([0, 0, 0])


            if not legs_visible:
                label = 0
            else:
                try:
                    # 1. Normalize based on Hip Center
                    hip_center = np.array([
                        (current_kp[11][0] + current_kp[12][0]) / 2,
                        (current_kp[11][1] + current_kp[12][1]) / 2
                    ])

                    # 2. Extract Relative Coordinates for LOWER BODY ONLY (6 points * 2 = 12 features)
                    relative_coords = []
                    for idx in important_indices:
                        px = current_kp[idx][0] - hip_center[0]
                        py = current_kp[idx][1] - hip_center[1]
                        relative_coords.extend([px, py])

                    # 3. Calculate Angles
                    angle_knee_l = calculate_angle([current_kp[11][0], current_kp[11][1]], [current_kp[13][0], current_kp[13][1]], [current_kp[15][0], current_kp[15][1]])
                    angle_knee_r = calculate_angle([current_kp[12][0], current_kp[12][1]], [current_kp[14][0], current_kp[14][1]], [current_kp[16][0], current_kp[16][1]])
                    angle_hip_l = calculate_angle([current_kp[5][0], current_kp[5][1]], [current_kp[11][0], current_kp[11][1]], [current_kp[13][0], current_kp[13][1]])
                    angle_hip_r = calculate_angle([current_kp[6][0], current_kp[6][1]], [current_kp[12][0], current_kp[12][1]], [current_kp[14][0], current_kp[14][1]])

                    current_angles = np.array([angle_knee_l, angle_knee_r, angle_hip_l, angle_hip_r])

                    if prev_angles is not None:
                        angle_deltas = current_angles - prev_angles
                    else:
                        angle_deltas = np.zeros(4)

                    prev_angles = current_angles

                    # 4. Distances
                    dist_foot_l = np.linalg.norm(np.array([current_kp[15][0], current_kp[15][1]]) - hip_center)
                    dist_foot_r = np.linalg.norm(np.array([current_kp[16][0], current_kp[16][1]]) - hip_center)

                    # FINAL FEATURE VECTOR PER FRAME (12 relative + 4 angles + 2 dist = 18 features)
                    current_frame_features = relative_coords + [
                        angle_knee_l, angle_knee_r, angle_hip_l, angle_hip_r,
                        dist_foot_l, dist_foot_r
                    ]

                    # --- TEMPORAL WINDOW PREDICTION ---
                    feature_buffer.append(current_frame_features)

                    if len(feature_buffer) == WINDOW_SIZE:
                        # Flatten buffer: (30, 18) -> (1, 540)
                        window_data = np.array(feature_buffer).flatten().reshape(1, -1)

                        try:
                            scaled_features = scaler.transform(window_data)
                            prediction_prob = kick_model(scaled_features, training=False).numpy()[0][0]
                        except Exception as e:
                            print(f"Scaling/Prediction Error: {e}")
                            prediction_prob = 0.0
                    else:
                        prediction_prob = 0.0

                    # Rule-based filter: Must have some angular velocity
                    total_angular_velocity = np.sum(np.abs(angle_deltas))
                    if total_angular_velocity < 3:
                        label = 0
                    else:
                        label = 1 if prediction_prob > 0.8 else 0

                except Exception as e:
                    print(f"Error in prediction loop: {e}")
                    label = 0

            prediction_buffer.append(label)
        else:
            prediction_buffer.append(0)

    # --- UI UPDATE ---
    if len(prediction_buffer) > 0:
        vote_score = sum(prediction_buffer) / len(prediction_buffer)
        if vote_score > 0.6: # Lowered from 0.85 to 0.6 for faster trigger
            text, color = "KICKING!", (0, 255, 0)
        else:
            text, color = "NO KICK", (0, 0, 255)
    else:
        text, color = "Analyzing...", (255, 255, 255)

    cv.putText(frame, f"Status: {text}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv.imshow('Taekwondo Kick Detection - Realtime', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv.destroyAllWindows()
