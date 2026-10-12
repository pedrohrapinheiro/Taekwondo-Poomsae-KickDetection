import cv2 as cv
import numpy as np
from ultralytics import YOLO
import csv
import os
from datetime import datetime
import sys

# Adicionar a raiz do projeto ao path para importar o core
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.pose_engine import PoseEngine

# =========================
# CONFIGURAÇÕES
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = os.path.join(SCRIPT_DIR, "videos")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "poses_dataset_polar.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n-pose.pt")

model = YOLO(MODEL_PATH)

def save_polar_pose_to_csv(filename, video_name, frame_idx, keypoints, label):
    """
    Salva características polares e normalizadas.
    YOLO Indices:
    Hips: 11(L), 12(R) | Knees: 13(L), 14(R) | Ankles: 15(L), 16(R)
    """
    file_exists = os.path.isfile(filename)

    # Extração de pontos (apenas x, y)
    pts = {i: (keypoints[i][0], keypoints[i][1]) for i in range(17)}

    # Utiliza o PoseEngine do core para extrair features e a distância de referência
    features, ref_dist = PoseEngine.extract_polar_features(pts)

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
            features[0], features[1], features[2], features[3],
            features[4], features[5]
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
                start_val = row['start_frame'].strip()
                end_val = row['end_frame'].strip()

                if not video or not start_val or not end_val:
                    continue

                try:
                    start = int(start_val)
                    end = int(end_val)
                except ValueError:
                    print(f"Aviso: Pulando linha com valores inválidos: {row}")
                    continue

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

    # Recria o CSV do zero: anexar a um arquivo antigo misturaria features/labels de execuções diferentes
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

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
            # Índice 0-based, igual ao annotator.py
            frame_idx = frame_count
            frame_count += 1

            label = 0
            if has_annotations:
                for start, end in annotations[video_file]:
                    if start <= frame_idx <= end:
                        label = 1
                        break

            results = model(frame, verbose=False)
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                person = PoseEngine.select_main_person(results[0])
                xy = results[0].keypoints.xy[person].cpu().numpy()
                conf = results[0].keypoints.conf[person].cpu().numpy()

                current_keypoints = []
                for i in range(17):
                    if i < len(xy):
                        current_keypoints.append((xy[i][0], xy[i][1], conf[i]))
                    else:
                        current_keypoints.append((0, 0, 0))

                save_polar_pose_to_csv(OUTPUT_CSV, video_file, frame_idx, current_keypoints, label)

            if frame_count % 100 == 0:
                print(f"  -> {video_file}: Frame {frame_count}/{total_frames}")

        cap.release()
    print("\nProcessamento polar finalizado! Dataset salvo em poses_dataset_polar.csv")

if __name__ == "__main__":
    process_videos()
