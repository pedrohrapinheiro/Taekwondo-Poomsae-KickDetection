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
# 1. POLAR GEOMETRY UTILS
# ==========================================
def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_normalized_distance(p1, p2, ref_dist):
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist / ref_dist if ref_dist != 0 else 0.0

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

        self.WINDOW_SIZE = 20 # Janela menor para resposta mais rápida
        self.feature_buffer = deque(maxlen=self.WINDOW_SIZE)
        self.prediction_buffer = deque(maxlen=3) # Votação curta para estabilidade instantânea

        self.lock = threading.Lock()
        self.current_status = "Analyzing..."
        self.current_color = (255, 255, 255)
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

                # Visibilidade Flexível
                hips_visible = conf[11] > 0.3 and conf[12] > 0.3
                ankles_visible = conf[15] > 0.3 or conf[16] > 0.3
                legs_visible = hips_visible and ankles_visible

                if not legs_visible:
                    if time.time() % 5 < 0.1:
                        print("Debug: Visibilidade insuficiente (Quadris ou Tornozelos ausentes)")
                    current_label = 0
                else:
                    try:
                        pts = {i: (xy[i][0], xy[i][1]) for i in range(17)}
                        f_list = [
                            calculate_angle(pts[11], pts[13], pts[15]),
                            calculate_angle(pts[12], pts[14], pts[16]),
                            calculate_angle(pts[5], pts[11], pts[13]),
                            calculate_angle(pts[6], pts[12], pts[14]),
                        ]
                        ref_dist = np.linalg.norm(np.array(pts[11]) - np.array(pts[12]))
                        f_list.append(get_normalized_distance(pts[11], pts[15], ref_dist))
                        f_list.append(get_normalized_distance(pts[12], pts[16], ref_dist))

                        self.feature_buffer.append(np.array(f_list))

                        if len(self.feature_buffer) == self.WINDOW_SIZE:
                            window = np.array(self.feature_buffer).reshape(1, self.WINDOW_SIZE, -1)
                            window_res = window.reshape(-1, window.shape[-1])
                            scaled = self.scaler.transform(window_res).reshape(1, self.WINDOW_SIZE, -1)
                            prob = self.kick_model(scaled, training=False).numpy()[0][0]

                            print(f"DEBUG: Probabilidade de Chute: {prob:.4f}")
                            current_label = 1 if prob > 0.5 else 0
                        else:
                            current_label = 0
                    except Exception as e:
                        print(f"AI Error: {e}")
                        current_label = 0

                self.prediction_buffer.append(current_label)

            # Votação para definir o status final
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

    cv.putText(frame, f"Status: {text}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv.imshow('Taekwondo Kick Detection - Polar LSTM', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

ai_thread.stopped = True
vs.stop()
cv.destroyAllWindows()
