"""
Extrai os keypoints dos vídeos de referência (Taegeuk 1-8, vistas frontal e lateral) com o
mesmo PoseDetector usado na avaliação, e salva em references/.

Uso:
    python Taekwondo_Poomsae_Evaluator/build_references.py [pasta_do_dataset]

A pasta do dataset é a que contém Data/video/Taegeuk_1st/Frontal/1st_Frontal.mp4 etc.
Padrão: Taekwondo_Poomsae_Evaluator/reference_data
"""
import os
import sys

import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from core.pose_engine import PoseDetector
from poomsae import FORMS, VIEWS, REFERENCE_DIR, extract_keypoints, reference_path

MODEL_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "Taekwondo_Kick_Pose_Predict", "yolov8n-pose.pt")


def main():
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "reference_data")
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    detector = PoseDetector(YOLO(MODEL_PATH))

    for form, ordinal in FORMS.items():
        for view in VIEWS:
            rel_video = os.path.join("Data", "video", f"Taegeuk_{ordinal}", view, f"{ordinal}_{view}.mp4")
            video = os.path.join(dataset_dir, rel_video)
            if not os.path.exists(video):
                print(f"Aviso: {video} não encontrado, pulando.")
                continue
            print(f"Taegeuk {form} ({view})...")
            xy, conf, fps, size = extract_keypoints(
                video, detector, on_progress=lambda i, n: print(f"  -> frame {i}/{n}"))
            np.savez_compressed(
                reference_path(form, view),
                xy=xy.astype(np.float32), conf=conf.astype(np.float16),
                fps=fps, size=np.array(size), video=rel_video.replace(os.sep, "/"))
            print(f"  salvo: {reference_path(form, view)} ({len(xy)} frames)")


if __name__ == "__main__":
    main()
