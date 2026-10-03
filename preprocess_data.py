import pandas as pd
import numpy as np
import os

def calculate_angle(a, b, c):
    """Calcula o ângulo em graus entre três pontos (a, b, c) onde b é o vértice."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def preprocess_poses_dynamic(input_path, output_path):
    print(f"Lendo dados de: {input_path}")
    df = pd.read_csv(input_path)

    # Ordenar por vídeo e timestamp para garantir a sequência temporal
    df = df.sort_values(by=['video_name', 'timestamp'])

    processed_data = []

    # Armazenar a última pose de cada vídeo para calcular a variação (delta)
    last_poses = {}

    for index, row in df.iterrows():
        video = row['video_name']
        try:
            # --- 1. GEOMETRIA ESTÁTICA ---
            hip_center = np.array([
                (row['kp11_x'] + row['kp12_x']) / 2,
                (row['kp11_y'] + row['kp12_y']) / 2
            ])

            relative_coords = []
            for i in range(17):
                px = row[f'kp{i}_x'] - hip_center[0]
                py = row[f'kp{i}_y'] - hip_center[1]
                relative_coords.extend([px, py])

            angle_knee_l = calculate_angle([row['kp11_x'], row['kp11_y']], [row['kp13_x'], row['kp13_y']], [row['kp15_x'], row['kp15_y']])
            angle_knee_r = calculate_angle([row['kp12_x'], row['kp12_y']], [row['kp14_x'], row['kp14_y']], [row['kp16_x'], row['kp16_y']])
            angle_hip_l = calculate_angle([row['kp5_x'], row['kp5_y']], [row['kp11_x'], row['kp11_y']], [row['kp13_x'], row['kp13_y']])
            angle_hip_r = calculate_angle([row['kp6_x'], row['kp6_y']], [row['kp12_x'], row['kp12_y']], [row['kp14_x'], row['kp14_y']])

            dist_foot_l = np.linalg.norm(np.array([row['kp15_x'], row['kp15_y']]) - hip_center)
            dist_foot_r = np.linalg.norm(np.array([row['kp16_x'], row['kp16_y']]) - hip_center)

            current_angles = np.array([angle_knee_l, angle_knee_r, angle_hip_l, angle_hip_r])

            # --- 2. DINÂMICA (VELOCIDADE ANGULAR) ---
            if video in last_poses:
                prev_angles = last_poses[video]
                # Velocidade = Mudança de ângulo / frame
                angle_deltas = current_angles - prev_angles
            else:
                angle_deltas = np.zeros(4)

            last_poses[video] = current_angles

            # Montar a linha final: Pose Estática + Velocidade dos Ângulos
            features = relative_coords + [
                angle_knee_l, angle_knee_r, angle_hip_l, angle_hip_r,
                dist_foot_l, dist_foot_r
            ] + list(angle_deltas)

            full_row = [row['timestamp'], row['video_name'], row['label']] + features
            processed_data.append(full_row)

        except Exception as e:
            continue

    # Colunas
    cols = ['timestamp', 'video_name', 'label']
    for i in range(17):
        cols.extend([f'rel_kp{i}_x', f'rel_kp{i}_y'])
    cols.extend(['angle_knee_l', 'angle_knee_r', 'angle_hip_l', 'angle_hip_r', 'dist_foot_l', 'dist_foot_r'])
    cols.extend(['delta_knee_l', 'delta_knee_r', 'delta_hip_l', 'delta_hip_r'])

    df_new = pd.DataFrame(processed_data, columns=cols)
    df_new.to_csv(output_path, index=False)
    print(f"Dataset Dinâmico salvo em: {output_path}")

if __name__ == "__main__":
    input_file = r'C:\Users\tatir\OneDrive\Documentos\Projetos Pessoais\TaekwondoProject\Taekwondo_Kick_Dataset_Creator\poses_dataset.csv'
    output_file = r'C:\Users\tatir\OneDrive\Documentos\Projetos Pessoais\TaekwondoProject\Taekwondo_Kick_Dataset_Creator\poses_dataset_dynamic.csv'
    preprocess_poses_dynamic(input_file, output_file)
