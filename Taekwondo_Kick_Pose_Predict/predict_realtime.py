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
from core.pose_engine import PoseEngine, PoseDetector

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
        self.frame_id = 0
        self.stopped = False

    def start(self):
        self.T = threading.Thread(target=self.update, args=(), daemon=True)
        self.T.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            self.frame_id += 1

    def read(self):
        return self.frame

    def read_with_id(self):
        return self.frame, self.frame_id

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
        self.detector = PoseDetector(pose_model)
        self.kick_model = kick_model
        self.scaler = scaler

        # Deve ser igual ao WINDOW_SIZE usado no treino (model.ipynb)
        self.WINDOW_SIZE = 30
        # FPS dos vídeos de treino: a janela é reamostrada para este ritmo, então ela sempre
        # cobre ~1 s de movimento, independente de quantos frames por segundo o loop consegue processar
        self.TRAIN_FPS = 30.0
        self.feature_buffer = deque(maxlen=self.WINDOW_SIZE * 4)  # (timestamp, features)

        # Chamar o modelo Keras direto (eager) custa ~150-200 ms por frame; compilado custa ~2 ms
        self.predict_fn = tf.function(
            lambda x: self.kick_model(x, training=False),
            input_signature=[tf.TensorSpec((1, self.WINDOW_SIZE, 6), tf.float32)])
        self.predict_fn(tf.zeros((1, self.WINDOW_SIZE, 6)))  # compila agora, não no primeiro chute

        # SPEED TRACKING
        self.prev_ankles = None
        self.prev_time = time.time()
        self.speed_buffer = deque(maxlen=5)

        self.lock = threading.Lock()
        self.current_status = "Analyzing..."
        self.current_color = (255, 255, 255)
        self.current_speed = 0.0
        self.current_label = None  # None = janela ainda incompleta
        self.last_frame_id = -1
        self.stopped = False

    def _resampled_window(self):
        """Janela de WINDOW_SIZE passos a TRAIN_FPS terminando no frame mais recente (interpolação linear)."""
        times = np.array([t for t, _ in self.feature_buffer])
        span = (self.WINDOW_SIZE - 1) / self.TRAIN_FPS
        if times[-1] - times[0] < span:
            return None
        feats = np.array([f for _, f in self.feature_buffer])
        target = times[-1] - span + np.arange(self.WINDOW_SIZE) / self.TRAIN_FPS
        return np.stack([np.interp(target, times, feats[:, j]) for j in range(feats.shape[1])], axis=1)

    def run(self):
        print("AI Processor Thread started.")
        while not self.stopped:
            frame, frame_id = self.vs.read_with_id()
            # Não processar o mesmo frame duas vezes (duplicaria passos na janela temporal)
            if frame is None or frame_id == self.last_frame_id:
                time.sleep(0.001)
                continue
            self.last_frame_id = frame_id

            detection = self.detector.detect(frame)

            if detection is not None:
                xy, conf = detection

                hips_visible = conf[11] > 0.3 and conf[12] > 0.3
                ankles_visible = conf[15] > 0.3 or conf[16] > 0.3
                legs_visible = hips_visible and ankles_visible

                if not legs_visible:
                    current_label = None
                    avg_speed = 0.0
                    # A janela precisa de frames consecutivos: descarta a sequência interrompida
                    self.feature_buffer.clear()
                    self.prev_ankles = None
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

                        # Use the core engine to get ref_dist (torso length)
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
                        self.feature_buffer.append((curr_time, f_list))

                        window = self._resampled_window()
                        if window is not None:
                            scaled = self.scaler.transform(window).reshape(1, self.WINDOW_SIZE, -1)
                            prob = self.predict_fn(tf.constant(scaled, tf.float32)).numpy()[0][0]
                            # O modelo responde "o último frame é um chute?": sem votação entre frames,
                            # que atrasava a detecção. Ele já liga e desliga junto com o chute.
                            current_label = 1 if prob > 0.5 else 0
                        else:
                            current_label = None
                    except Exception as e:
                        print(f"AI Error: {e}")
                        current_label = None
                        avg_speed = 0
            else:
                current_label = None
                avg_speed = 0.0
                self.feature_buffer.clear()
                self.prev_ankles = None

            with self.lock:
                self.current_speed = avg_speed
                if current_label is None:
                    self.current_status, self.current_color = "Analyzing...", (255, 255, 255)
                elif current_label == 1:
                    self.current_status, self.current_color = "KICKING!", (0, 255, 0)
                else:
                    self.current_status, self.current_color = "NO KICK", (0, 0, 255)

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
