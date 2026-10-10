import numpy as np
from .geometry import calculate_angle, get_normalized_distance

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
        # Referência: Distância entre os quadris (Hip width)
        ref_dist = np.linalg.norm(np.array(pts[11]) - np.array(pts[12]))

        f_list = [
            calculate_angle(pts[11], pts[13], pts[15]), # Joelho L
            calculate_angle(pts[12], pts[14], pts[16]), # Joelho R
            calculate_angle(pts[5], pts[11], pts[13]),  # Quadril L
            calculate_angle(pts[6], pts[12], pts[14]),  # Quadril R
            get_normalized_distance(pts[11], pts[15], ref_dist), # Dist Hip-Ankle L
            get_normalized_distance(pts[12], pts[16], ref_dist), # Dist Hip-Ankle R
        ]
        return np.array(f_list), ref_dist
