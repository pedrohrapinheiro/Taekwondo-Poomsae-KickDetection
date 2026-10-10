import numpy as np

def calculate_angle(a, b, c):
    """
    Calcula o ângulo em graus no ponto b dado os pontos a e c.
    Utiliza o produto escalar para garantir invariância de escala.
    """
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
    """
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    if ref_dist == 0:
        return 0.0
    return dist / ref_dist
