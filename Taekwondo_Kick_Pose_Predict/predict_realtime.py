import cv2 as cv
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import joblib
import os
import threading
import warnings
import time
import sys

# Adicionar a raiz do projeto ao path para importar o core
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.pose_engine import PoseEngine

# Silenciar avisos do TensorFlow e sklearn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 0. VIDEO STREAM THREADING
# ==========================================
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv.VideoCapture(src)
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
# 2. AI PROCESSOR THREAD
# ==========================================
class AIProcessor(threading.Thread):
    def __init__(self, vs, pose_model, kick_model, scaler):
        super().__init__(daemon=True)
        self.vs = vs
        self.pose_model = pose_model
        self.kick_model = kick_model
        self.scaler = scaler

        self.WINDOW_SIZE = 20
        self.feature_buffer = deque(maxlen=self.WINDOW_SIZE)
        self.prediction_buffer = deque(maxlen=3)

        # SPEED TRACKING
        self.prev_ankles = None
        self.prev_time = time.time()
        self.speed_buffer = deque(maxlen=5)

        self.lock = threading.Lock()
        self.current_status = "Analyzing..."
        self.current_color = (255, 255, 255)
        self.current_speed = 0.0
        self.stopped = False

    def run(self):
        print("AI Processor Thread started.")
        while not self.stopped:
            frame = self.vs.read()
            if frame is None: continue

            results = self.pose_model(frame, verbose=False)

            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                xy = results[0].keypoints.xy[0].cpu().numpy()
                conf = results[0].keypoints.conf[0].cpu().numpy()

                hips_visible = conf[11] > 0.3 and conf[12] > 0.3
                ankles_visible = conf[15] > 0.3 or conf[16] > 0.3
                legs_visible = hips_visible and ankles_visible

                if not legs_visible:
                    current_label = 0
                    avg_speed = 0.0
                    with self.lock:
                        self.current_speed = 0.0
                else:
                    avg_speed = 0.0
                    try:
                        pts = {i: (xy[i][0], xy[i][1]) for i in range(17)}

                        # --- SPEED CALCULATION ---
                        curr_time = time.time()
                        dt = curr_time - self.prev_time
                        self.prev_time = curr_time

                        # Use the core engine to get ref_dist (hip width)
                        f_list, ref_dist = PoseEngine.extract_polar_features(pts)

                        curr_ankles = np.array([pts[15], pts[16]])
                        if self.prev_ankles is not None:
                            dists = np.linalg.norm(curr_ankles - self.prev_ankles, axis=1)
                            norm_dists = dists / ref_dist if ref_dist != 0 else 0
                            instant_speed = np.max(norm_dists) / dt if dt > 0 else 0
                            self.speed_buffer.append(instant_speed)

                        self.prev_ankles = curr_ankles
                        avg_speed = np.mean(self.speed_buffer) if self.speed_buffer else 0

                        # --- POSE FEATURES ---
                        self.feature_buffer.append(f_list)

                        if len(self.feature_buffer) == self.WINDOW_SIZE:
                            window = np.array(self.feature_buffer).reshape(1, self.WINDOW_SIZE, -1)
                            window_res = window.reshape(-1, window.shape[-1])
                            scaled = self.scaler.transform(window_res).reshape(1, self.WINDOW_SIZE, -1)
                            prob = self.kick_model(scaled, training=False).numpy()[0][0]
                            current_label = 1 if prob > 0.5 else 0
                        else:
                            current_label = 0
                    except Exception as e:
                        print(f"AI Error: {e}")
                        current_label = 0
                        avg_speed = 0

                self.prediction_buffer.append(current_label)

                with self.lock:
                    self.current_speed = avg_speed

            with self.lock:
                if len(self.prediction_buffer) > 0:
                    vote_score = sum(self.prediction_buffer) / len(self.prediction_buffer)
                    if vote_score > 0.6:
                        self.current_status, self.current_color = "KICKING!", (0, 255, 0)
                    else:
                        self.current_status, self.current_color = "NO KICK", (0, 0, 255)
                else:
                    self.current_status, self.current_color = "Analyzing...", (255, 255, 255)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
print("Loading models... please wait.")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pose_model = YOLO(os.path.join(SCRIPT_DIR, "yolov8n-pose.pt"))
kick_model = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, "model", "kick_detection_model.h5"))
scaler = joblib.load(os.path.join(SCRIPT_DIR, "model", "scaler.pkl"))

vs = VideoStream(src=0).start()
ai_thread = AIProcessor(vs, pose_model, kick_model, scaler)
ai_thread.start()

print("System Ready! Press 'q' to quit.")

while True:
    frame = vs.read()
    if frame is None: break

    with ai_thread.lock:
        text = ai_thread.current_status
        color = ai_thread.current_color
        speed = ai_thread.current_speed

    cv.putText(frame, f"Status: {text}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv.putText(frame, f"Speed: {speed:.2f} units/s", (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv.imshow('Taekwondo Kick Detection - Polar LSTM', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

ai_thread.stopped = True
vs.stop()
cv.destroyAllWindows()
