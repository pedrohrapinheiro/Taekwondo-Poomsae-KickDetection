import cv2 as cv
import numpy as np
from ultralytics import YOLO
import csv
import os
from datetime import datetime

# =========================
# CONFIGURAÇÕES
# =========================

# Pasta onde estão os vídeos
VIDEO_FOLDER = "videos"
# Nome do arquivo de saída
OUTPUT_CSV = "poses_dataset.csv"

# Inicializa o modelo YOLO Pose
model = YOLO("yolov8n-pose.pt")

# =========================
# FUNÇÕES
# =========================

def save_pose_to_csv(filename, video_name, keypoints, label):
    """
    Salva os keypoints no CSV.
    Estrutura: timestamp, video_name, label, kp0_x, kp0_y, kp0_conf, ..., kp16_x, kp16_y, kp16_conf
    """
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Escreve o cabeçalho se o arquivo for novo
        if not file_exists:
            header = ['timestamp', 'video_name', 'label']
            for i in range(17):
                header += [f'kp{i}_x', f'kp{i}_y', f'kp{i}_conf']
            writer.writerow(header)

        # Prepara a linha
        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), video_name, label]
        for kp in keypoints:
            row += list(kp)

        writer.writerow(row)

# =========================
# PROCESSO PRINCIPAL
# =========================

def process_videos():
    # Verifica se a pasta de vídeos existe
    if not os.path.exists(VIDEO_FOLDER):
        print(f"Erro: A pasta '{VIDEO_FOLDER}' não foi encontrada.")
        return

    # Lista todos os arquivos na pasta
    files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not files:
        print("Nenhum arquivo de vídeo encontrado na pasta.")
        return

    print(f"Encontrados {len(files)} vídeos. Iniciando processamento...")

    for video_file in files:
        # Determina o label com base no prefixo do nome do arquivo
        if video_file.startswith("kick_"):
            label = 1
        elif video_file.startswith("no_kick_"):
            label = 0
        else:
            print(f"Pulando arquivo {video_file} (não começa com 'kick_' ou 'no_kick_')")
            continue

        video_path = os.path.join(VIDEO_FOLDER, video_file)
        cap = cv.VideoCapture(video_path)

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        print(f"Processando: {video_file} | Label: {label} | Total Frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Inferência YOLO
            results = model(frame, verbose=False)

            # Extração de Keypoints (apenas a primeira pessoa detectada)
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                xy = results[0].keypoints.xy[0].cpu().numpy()
                conf = results[0].keypoints.conf[0].cpu().numpy()

                person_keypoints = []
                for i in range(17):
                    # Garante que pegamos x, y, conf mesmo se o modelo retornar menos
                    if i < len(xy):
                        person_keypoints.append((xy[i][0], xy[i][1], conf[i]))
                    else:
                        person_keypoints.append((0, 0, 0))

                # Salva no CSV
                save_pose_to_csv(OUTPUT_CSV, video_file, person_keypoints, label)

            if frame_count % 100 == 0:
                print(f"  -> Frame {frame_count}/{total_frames} processado...")

        cap.release()
        print(f"Concluído: {video_file}")

    print("\nProcessamento finalizado com sucesso!")

if __name__ == "__main__":
    process_videos()
