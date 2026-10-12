import numpy as np
from .geometry import calculate_angle, get_normalized_distance, is_missing

class PoseEngine:
    """
    Motor de processamento de pose que converte keypoints do YOLO
    nas features polares usadas para detecção de chutes.
    """
    @staticmethod
    def extract_polar_features(pts):
        """
        Extrai as 6 features polares:
        - 4 Ângulos (Joelhos L/R, Quadris L/R)
        - 2 Distâncias Normalizadas (Quadril -> Tornozelo L/R)
        """
        # Referência: comprimento do tronco (meio dos ombros -> meio dos quadris).
        # A largura do quadril colapsa para ~0 quando o atleta fica de lado
        # (posição normal de chute), o que explodia as distâncias normalizadas.
        torso_pts = [pts[5], pts[6], pts[11], pts[12]]
        if any(is_missing(p) for p in torso_pts):
            ref_dist = 0.0
        else:
            mid_shoulder = (np.array(pts[5]) + np.array(pts[6])) / 2
            mid_hip = (np.array(pts[11]) + np.array(pts[12])) / 2
            ref_dist = np.linalg.norm(mid_shoulder - mid_hip)

        f_list = [
            calculate_angle(pts[11], pts[13], pts[15]), # Joelho L
            calculate_angle(pts[12], pts[14], pts[16]), # Joelho R
            calculate_angle(pts[5], pts[11], pts[13]),  # Quadril L
            calculate_angle(pts[6], pts[12], pts[14]),  # Quadril R
            get_normalized_distance(pts[11], pts[15], ref_dist), # Dist Hip-Ankle L
            get_normalized_distance(pts[12], pts[16], ref_dist), # Dist Hip-Ankle R
        ]
        return np.array(f_list), ref_dist

    @staticmethod
    def select_main_person(result):
        """
        Retorna o índice da pessoa com a maior bounding box (o atleta em primeiro plano),
        em vez da primeira detecção do YOLO, que pode ser alguém ao fundo.
        """
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return 0
        wh = boxes.xywh[:, 2:4].cpu().numpy()
        return int(np.argmax(wh[:, 0] * wh[:, 1]))
