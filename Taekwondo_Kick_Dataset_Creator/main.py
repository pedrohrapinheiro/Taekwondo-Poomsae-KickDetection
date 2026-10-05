import cv2 as cv
import numpy as np
from ultralytics import YOLO
import csv
import os
from datetime import datetime

# =========================
# CONFIGURAÇÕES
# =========================

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Pasta onde estão os vídeos
VIDEO_FOLDER = os.path.join(SCRIPT_DIR, "videos")
# Nome do arquivo de saída
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "poses_dataset.csv")
# Caminho para o modelo local
MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n-pose.pt")

# Inicializa o modelo YOLO Pose usando o arquivo local
model = YOLO(MODEL_PATH)

# =========================
# FUNÇÕES
# =========================

def save_pose_to_csv(filename, video_name, frame_idx, keypoints, deltas, label):
    """
    Salva os keypoints e as variações (deltas) no CSV.
    Estrutura: timestamp, video_name, frame_idx, label, kp0_x, kp0_y, kp0_conf, ..., delta_kp0_x, delta_kp0_y, ...
    """
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Escreve o cabeçalho se o arquivo for novo
        if not file_exists:
            header = ['timestamp', 'video_name', 'frame_idx', 'label']
            # Colunas de Posição
            for i in range(17):
                header += [f'kp{i}_x', f'kp{i}_y', f'kp{i}_conf']
            # Colunas de Movimento (Deltas)
            for i in range(17):
                header += [f'delta_kp{i}_x', f'delta_kp{i}_y']
            writer.writerow(header)

        # Prepara a linha
        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), video_name, frame_idx, label]

        # Adiciona posições atuais
        for kp in keypoints:
            row += list(kp)

        # Adiciona deltas (movimento)
        for d in deltas:
            row += list(d)

        writer.writerow(row)

# =========================
# PROCESSO PRINCIPAL
# =========================

def process_videos():
    # Verifica se a pasta de vídeos existe
    if not os.path.exists(VIDEO_FOLDER):
        print(f"Erro: A pasta '{VIDEO_FOLDER}' não foi encontrada.")
        return

    # Carrega anotações de chutes
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
        print(f"Anotações carregadas de {ANNOTATIONS_FILE}. Vídeos anotados: {list(annotations.keys())}")
    else:
        print(f"Aviso: {ANNOTATIONS_FILE} não encontrado. Todos os frames serão label 0.")


    # Lista todos os arquivos na pasta
    files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not files:
        print("Nenhum arquivo de vídeo encontrado na pasta.")
        return

    print(f"Encontrados {len(files)} vídeos. Iniciando processamento...")

    for video_file in files:
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        cap = cv.VideoCapture(video_path)

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        # Variável para armazenar os pontos do frame anterior (para calcular o delta)
        prev_keypoints = None

        # Debug: Verifica se este vídeo tem anotações
        has_annotations = video_file in annotations
        print(f"Processando: {video_file} | Total Frames: {total_frames} | Tem Anotações: {has_annotations}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Determina o label por frame
            label = 0
            if has_annotations:
                for start, end in annotations[video_file]:
                    if start <= frame_count <= end:
                        label = 1
                        break

            # Inferência YOLO
            results = model(frame, verbose=False)


            # Extração de Keypoints (apenas a primeira pessoa detectada)
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                xy = results[0].keypoints.xy[0].cpu().numpy()
                conf = results[0].keypoints.conf[0].cpu().numpy()

                current_keypoints = []
                for i in range(17):
                    if i < len(xy):
                        current_keypoints.append((xy[i][0], xy[i][1], conf[i]))
                    else:
                        current_keypoints.append((0, 0, 0))

                # Cálculo de Deltas (Movimento)
                deltas = []
                if prev_keypoints is not None:
                    for i in range(17):
                        # Delta X = x_atual - x_anterior
                        # Delta Y = y_atual - y_anterior
                        dx = current_keypoints[i][0] - prev_keypoints[i][0]
                        dy = current_keypoints[i][1] - prev_keypoints[i][1]
                        deltas.append((dx, dy))
                else:
                    # No primeiro frame, o delta é zero
                    deltas = [(0, 0)] * 17

                # Salva no CSV (Posição + Delta)
                save_pose_to_csv(OUTPUT_CSV, video_file, frame_count, current_keypoints, deltas, label)

                # Atualiza o frame anterior para a próxima iteração
                prev_keypoints = current_keypoints

            if frame_count % 100 == 0:
                print(f"  -> Frame {frame_count}/{total_frames} processado...")

        cap.release()
        print(f"Concluído: {video_file}")

    print("\nProcessamento finalizado com sucesso!")

if __name__ == "__main__":
    process_videos()
