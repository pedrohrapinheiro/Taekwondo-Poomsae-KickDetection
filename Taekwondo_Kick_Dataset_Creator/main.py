import cv2 as cv
import numpy as np
from ultralytics import YOLO
import csv
import os
from datetime import datetime

# =========================
# CONFIGURAÇÕES
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = os.path.join(SCRIPT_DIR, "videos")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "poses_dataset_polar.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n-pose.pt")

model = YOLO(MODEL_PATH)

# =========================
# FUNÇÕES MATEMÁTICAS (POLAR & INVARIÂNCIA)
# =========================

def calculate_angle(a, b, c):
    """
    Calcula o ângulo em graus no ponto b dado os pontos a e c.
    Utiliza o produto escalar para garantir invariância de escala.
    """
    # Filtro de confiança: se qualquer ponto for (0,0), retorna 0
    if np.all(a == 0) or np.all(b == 0) or np.all(c == 0):
        return 0.0

    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_normalized_distance(p1, p2, ref_dist):
    """
    Calcula a distância entre p1 e p2 normalizada por uma distância de referência.
    Isso resolve o problema de se afastar da câmera.
    """
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    if ref_dist == 0:
        return 0.0
    return dist / ref_dist

def save_polar_pose_to_csv(filename, video_name, frame_idx, keypoints, label):
    """
    Salva características polares e normalizadas.
    YOLO Indices:
    Hips: 11(L), 12(R) | Knees: 13(L), 14(R) | Ankles: 15(L), 16(R)
    """
    file_exists = os.path.isfile(filename)

    # Extração de pontos (apenas x, y)
    pts = {i: (keypoints[i][0], keypoints[i][1]) for i in range(17)}

    # 1. Cálculo de Ângulos (Invariantes à distância)
    # Joelhos: Quadril -> Joelho -> Tornozelo
    angle_knee_l = calculate_angle(pts[11], pts[13], pts[15])
    angle_knee_r = calculate_angle(pts[12], pts[14], pts[16])
    # Quadris: Ombro(5/6) -> Quadril -> Joelho
    angle_hip_l = calculate_angle(pts[5], pts[11], pts[13])
    angle_hip_r = calculate_angle(pts[6], pts[12], pts[14])

    # 2. Normalização de Distância (Invariante à escala)
    # Referência: Distância entre os quadris (Hip width)
    ref_dist = np.linalg.norm(np.array(pts[11]) - np.array(pts[12]))

    # Distâncias normalizadas (Ex: Quadril para Tornozelo)
    dist_hip_ankle_l = get_normalized_distance(pts[11], pts[15], ref_dist)
    dist_hip_ankle_r = get_normalized_distance(pts[12], pts[16], ref_dist)

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [
                'timestamp', 'video_name', 'frame_idx', 'label',
                'angle_knee_l', 'angle_knee_r', 'angle_hip_l', 'angle_hip_r',
                'dist_hip_ankle_l', 'dist_hip_ankle_r'
            ]
            writer.writerow(header)

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            video_name, frame_idx, label,
            angle_knee_l, angle_knee_r, angle_hip_l, angle_hip_r,
            dist_hip_ankle_l, dist_hip_ankle_r
        ]
        writer.writerow(row)

# =========================
# PROCESSO PRINCIPAL
# =========================

def process_videos():
    if not os.path.exists(VIDEO_FOLDER):
        print(f"Erro: A pasta '{VIDEO_FOLDER}' não foi encontrada.")
        return

    annotations = {}
    ANNOTATIONS_FILE = os.path.join(SCRIPT_DIR, "annotations.csv")
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video = row['video'].strip()
                start = int(row['start_frame'])
                end = int(row['end_frame'])
                if video not in annotations:
                    annotations[video] = []
                annotations[video].append((start, end))
        print(f"Anotações carregadas. Vídeos: {list(annotations.keys())}")
    else:
        print(f"Aviso: {ANNOTATIONS_FILE} não encontrado.")

    files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not files:
        print("Nenhum arquivo de vídeo encontrado.")
        return

    print(f"Iniciando processamento polar de {len(files)} vídeos...")

    for video_file in files:
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        has_annotations = video_file in annotations

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            label = 0
            if has_annotations:
                for start, end in annotations[video_file]:
                    if start <= frame_count <= end:
                        label = 1
                        break

            results = model(frame, verbose=False)
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                xy = results[0].keypoints.xy[0].cpu().numpy()
                conf = results[0].keypoints.conf[0].cpu().numpy()

                current_keypoints = []
                for i in range(17):
                    if i < len(xy):
                        current_keypoints.append((xy[i][0], xy[i][1], conf[i]))
                    else:
                        current_keypoints.append((0, 0, 0))

                save_polar_pose_to_csv(OUTPUT_CSV, video_file, frame_count, current_keypoints, label)

            if frame_count % 100 == 0:
                print(f"  -> {video_file}: Frame {frame_count}/{total_frames}")

        cap.release()
    print("\nProcessamento polar finalizado! Dataset salvo em poses_dataset_polar.csv")

if __name__ == "__main__":
    process_videos()
