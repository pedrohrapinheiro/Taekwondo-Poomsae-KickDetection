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


class PoseDetector:
    """
    Detecção de pose invariante à distância da câmera.

    Quando o atleta está longe ele ocupa poucos pixels e os keypoints do YOLO ficam
    imprecisos. Aqui a pose é sempre estimada num recorte quadrado ao redor do atleta,
    que o YOLO redimensiona para o tamanho de entrada: o corpo chega ao modelo
    sempre com aproximadamente o mesmo tamanho, perto ou longe.

    O recorte do frame seguinte usa a caixa do frame anterior (tracking), então em regime
    normal é só uma inferência por frame. Se o atleta for perdido, volta ao frame inteiro.
    """
    def __init__(self, model, crop_scale=1.4):
        self.model = model
        # Lado do recorte = crop_scale * maior lado da caixa da pessoa.
        # A folga cobre a perna estendida num chute e o movimento entre frames.
        self.crop_scale = crop_scale
        self.prev_box = None

    def reset(self):
        """Chamar ao trocar de vídeo/câmera."""
        self.prev_box = None

    def _crop_region(self, box, frame_w, frame_h):
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * self.crop_scale / 2
        return (int(max(0, cx - half)), int(max(0, cy - half)),
                int(min(frame_w, cx + half)), int(min(frame_h, cy + half)))

    def _find_person(self, frame):
        result = self.model(frame, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        person = PoseEngine.select_main_person(result)
        return result.boxes.xyxy[person].cpu().numpy()

    def detect(self, frame):
        """
        Retorna (xy, conf) da pessoa principal em coordenadas do frame original,
        ou None se ninguém for encontrado. Keypoints não detectados continuam em (0, 0).
        """
        h, w = frame.shape[:2]
        tracked = self.prev_box is not None
        box = self.prev_box if tracked else self._find_person(frame)
        if box is None:
            return None

        x0, y0, x1, y1 = self._crop_region(box, w, h)
        result = self.model(frame[y0:y1, x0:x1], verbose=False)[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0 or len(result.boxes) == 0:
            self.prev_box = None
            # Caixa do tracking ficou desatualizada: tenta de novo procurando no frame inteiro
            return self.detect(frame) if tracked else None

        person = PoseEngine.select_main_person(result)
        xy = result.keypoints.xy[person].cpu().numpy().copy()
        conf = result.keypoints.conf[person].cpu().numpy()

        missing = np.all(xy == 0, axis=1)
        xy += (x0, y0)
        xy[missing] = 0

        self.prev_box = result.boxes.xyxy[person].cpu().numpy() + (x0, y0, x0, y0)
        return xy, conf
