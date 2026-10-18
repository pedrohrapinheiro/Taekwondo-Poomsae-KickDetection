"""
Avaliação de Poomsae por comparação com uma execução de referência.

Pipeline:
1. Keypoints extraídos com o mesmo PoseDetector (YOLO + recorte) do detector de chutes.
2. Normalização: centro no quadril e escala pelo comprimento do tronco
   (independe da distância e da posição na quadra).
3. Alinhamento temporal com DTW de subsequência: o vídeo do usuário pode ter tempo
   parado antes/depois e ritmo diferente da referência.
4. A referência é dividida em movimentos (pausas entre técnicas); cada movimento
   recebe nota, comparação de ângulos na postura final e comparação de ritmo.
"""
import os
import sys

import cv2 as cv
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# As 12 juntas do dataset de referência, na ordem usada em todo o módulo
JOINTS = ['L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist',
          'L_hip', 'R_hip', 'L_knee', 'R_knee', 'L_ankle', 'R_ankle']
# Índices correspondentes nos 17 keypoints COCO do YOLO
COCO_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# Espelhamento: troca esquerda <-> direita
MIRROR_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]

# Segmentos usados no feedback: nome -> (junta de origem, junta da ponta) em índices de JOINTS.
# Compara-se a DIREÇÃO de cada segmento (com sinal), não o ângulo interno da articulação:
# ângulos internos não têm sentido e um braço girado 90° pode parecer só 40° diferente.
SEGMENTS = {
    'left upper arm': (0, 2), 'right upper arm': (1, 3),
    'left forearm': (2, 4), 'right forearm': (3, 5),
    'left thigh': (6, 8), 'right thigh': (7, 9),
    'left shin': (8, 10), 'right shin': (9, 11),
}

FORMS = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th', 5: '5th', 6: '6th', 7: '7th', 8: '8th'}
VIEWS = ('Frontal', 'Lateral')

# Taxa usada na análise: metade dos 30 fps originais deixa o DTW 4x mais rápido
# sem perder movimentos (uma técnica dura ~0.5-1 s)
ANALYSIS_FPS = 15
CONF_THRESHOLD = 0.3

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'references')


# ==========================================
# EXTRAÇÃO E LIMPEZA DE KEYPOINTS
# ==========================================
def extract_keypoints(video_path, detector, on_progress=None):
    """Retorna xy (T, 12, 2) em pixels, conf (T, 12), fps e (largura, altura) do vídeo."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    detector.reset()

    xy_list, conf_list = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detection = detector.detect(frame)
        if detection is None:
            xy_list.append(np.full((12, 2), np.nan))
            conf_list.append(np.zeros(12))
        else:
            xy, conf = detection
            xy_list.append(xy[COCO_IDX])
            conf_list.append(conf[COCO_IDX])
        if on_progress and len(xy_list) % 100 == 0:
            on_progress(len(xy_list), total)
    cap.release()
    return np.array(xy_list, dtype=np.float32), np.array(conf_list, dtype=np.float32), fps, size


def clean_keypoints(xy, conf):
    """Descarta juntas com baixa confiança e preenche os buracos interpolando no tempo."""
    xy = xy.astype(np.float64).copy()
    xy[conf < CONF_THRESHOLD] = np.nan
    t = np.arange(len(xy))
    for j in range(xy.shape[1]):
        for k in range(2):
            col = xy[:, j, k]
            valid = ~np.isnan(col)
            if valid.sum() == 0:
                raise ValueError(f"Junta {JOINTS[j]} nunca foi detectada no vídeo")
            xy[:, j, k] = np.interp(t, t[valid], col[valid])
    return xy


def resample(seq, fps, target_fps=ANALYSIS_FPS):
    """Reamostra uma sequência (T, ...) para target_fps por interpolação linear."""
    n_new = max(2, int(round(len(seq) * target_fps / fps)))
    t_old = np.arange(len(seq)) / fps
    t_new = np.arange(n_new) / target_fps
    t_new = t_new[t_new <= t_old[-1]]
    flat = seq.reshape(len(seq), -1)
    out = np.stack([np.interp(t_new, t_old, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape((len(t_new),) + seq.shape[1:])


def mirror(xy, frame_width):
    """Espelha horizontalmente (vídeo de câmera frontal de celular costuma vir espelhado)."""
    out = xy[:, MIRROR_IDX].copy()
    out[..., 0] = frame_width - out[..., 0]
    return out


# ==========================================
# FEATURES
# ==========================================
def pose_frame(xy, fps=ANALYSIS_FPS):
    """
    Centro (meio do quadril) e escala (comprimento do tronco) de cada frame.
    O tronco é suavizado com uma mediana de ~2 s: acompanha a pessoa se aproximando ou
    se afastando da câmera, mas não oscila quando ela inclina o tronco ou gira.
    """
    hip_mid = (xy[:, 6] + xy[:, 7]) / 2
    shoulder_mid = (xy[:, 0] + xy[:, 1]) / 2
    torso = np.linalg.norm(shoulder_mid - hip_mid, axis=1)
    torso = median_filter(torso, size=max(3, int(2 * fps) | 1), mode='nearest')
    return hip_mid, np.maximum(torso, 1e-6)


def normalize_pose(xy, fps=ANALYSIS_FPS):
    """Centraliza no meio do quadril e divide pelo comprimento do tronco."""
    hip_mid, torso = pose_frame(xy, fps)
    norm = (xy - hip_mid[:, None, :]) / torso[:, None, None]
    # Suavização leve contra o tremor do detector (~0.2 s)
    return uniform_filter1d(norm, size=3, axis=0, mode='nearest')


def segment_directions(norm):
    """Direção (graus, atan2 em coordenadas de imagem) de cada segmento: (T, len(SEGMENTS))."""
    vec = np.stack([norm[:, e] - norm[:, o] for o, e in SEGMENTS.values()], axis=1)
    return np.degrees(np.arctan2(vec[..., 1], vec[..., 0]))


def pose_distance_matrix(ref_norm, user_norm):
    """Distância média por junta (em comprimentos de tronco) entre cada par de frames."""
    diff = ref_norm[:, None, :, :] - user_norm[None, :, :, :]
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)


# ==========================================
# ALINHAMENTO (DTW DE SUBSEQUÊNCIA)
# ==========================================
# Passos (avanço na referência, avanço no usuário). Limitar a inclinação entre 1/3 e 3
# permite ritmos de 3x mais lento a 3x mais rápido e deixa cada linha da matriz depender
# só das linhas anteriores, o que permite vetorizar o cálculo com numpy.
DTW_STEPS = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1)]


def _shift(row, s, fill=np.inf):
    """row deslocado s posições para a direita: out[j] = row[j - s]."""
    if s == 0:
        return row
    out = np.full_like(row, fill)
    out[s:] = row[:-s]
    return out


def dtw_subsequence(cost):
    """
    Alinha a referência inteira (linhas) a um trecho contínuo do vídeo do usuário (colunas).
    Retorna (path, custo_médio); path lista TODAS as células (frame_ref, frame_usuario) do
    caminho, inclusive as atravessadas por passos longos.

    Cada passo paga o custo de todas as células que atravessa (peso 2 na primeira, 1 nas
    demais, como no DTW simétrico). Sem isso o alinhamento "pula" frames errados do usuário
    de graça e um movimento mal executado acaba alinhado aos frames corretos vizinhos.
    """
    n, m = cost.shape
    inf = np.inf
    D = np.full((n, m), inf)
    choice = np.full((n, m), -1, dtype=np.int8)
    start = np.zeros((n, m), dtype=np.int64)
    # Começo livre: a referência pode começar em qualquer frame do usuário
    D[0] = 2 * cost[0]
    start[0] = np.arange(m)

    for i in range(1, n):
        c = cost[i]
        best = np.full(m, inf)
        best_k = np.full(m, -1, dtype=np.int8)
        best_start = np.zeros(m, dtype=np.int64)
        for k, (di, dj) in enumerate(DTW_STEPS):
            if i - di < 0:
                continue
            if di == 1:   # avança 1 na referência e dj no usuário: células (i, j-dj+1 .. j)
                step_cost = 2 * _shift(c, dj - 1) + sum(_shift(c, t) for t in range(dj - 1))
            else:         # avança di na referência e 1 no usuário: células (i-di+1 .. i, j)
                step_cost = 2 * cost[i - di + 1] + sum(cost[i - t] for t in range(di - 1))
            cand = _shift(D[i - di], dj) + step_cost
            better = cand < best
            best[better] = cand[better]
            best_k[better] = k
            best_start[better] = _shift(start[i - di], dj, fill=0)[better]
        D[i], choice[i], start[i] = best, best_k, best_start

    # Fim livre: normaliza pelo peso total do caminho para não favorecer trechos curtos
    span = np.arange(m) - start[n - 1]
    normalized = D[n - 1] / (n + 1 + span)
    j = int(np.argmin(normalized))
    if not np.isfinite(normalized[j]):
        raise ValueError("Não foi possível alinhar o vídeo à referência (vídeo curto demais?)")
    total = float(normalized[j])

    path = []
    i = n - 1
    while True:
        if i == 0:
            path.append((0, j))
            break
        di, dj = DTW_STEPS[choice[i, j]]
        if di == 1:
            path.extend((i, j - t) for t in range(dj))
        else:
            path.extend((i - t, j) for t in range(di))
        i, j = i - di, j - dj
    path.reverse()
    return path, total


def ref_to_user_map(path, n_ref):
    """Para cada frame da referência, o frame do usuário alinhado (média quando há vários)."""
    path = np.asarray(path)
    sums = np.bincount(path[:, 0], weights=path[:, 1], minlength=n_ref)
    counts = np.bincount(path[:, 0], minlength=n_ref)
    have = counts > 0
    mapped = np.interp(np.arange(n_ref), np.flatnonzero(have), sums[have] / counts[have])
    return np.round(mapped).astype(int)


# ==========================================
# SEGMENTAÇÃO EM MOVIMENTOS
# ==========================================
def segment_moves(ref_norm, fps=ANALYSIS_FPS, min_move_s=0.8, min_pause_drop=0.15):
    """
    Divide a referência em movimentos nos pontos de pausa: cada técnica de Poomsae termina
    numa postura parada. Retorna lista de (início, fim, frame_da_postura_final).
    Os parâmetros padrão dão uma contagem de movimentos ~igual à oficial de cada Taegeuk
    (ex.: 17 vs 18 no Taegeuk 1, 28 vs 27 no Taegeuk 8).
    """
    speed = np.linalg.norm(np.diff(ref_norm, axis=0), axis=-1).mean(axis=-1) * fps
    speed = uniform_filter1d(np.concatenate([[speed[0]], speed]), size=max(3, int(0.3 * fps)), mode='nearest')

    # Mínimos locais de velocidade separados por pelo menos min_move_s
    win = max(1, int(min_move_s * fps / 2))
    peak = speed.max()
    pauses = []
    for t in range(len(speed)):
        lo, hi = max(0, t - win), min(len(speed), t + win + 1)
        if speed[t] == speed[lo:hi].min() and speed[t] < peak * min_pause_drop:
            if not pauses or t - pauses[-1] >= 2 * win:
                pauses.append(t)

    # A primeira pausa é a posição inicial (joonbi); movimentos vão de uma pausa à próxima
    moves = []
    for a, b in zip(pauses[:-1], pauses[1:]):
        moves.append((a, b, b))
    return moves, speed


# ==========================================
# REFERÊNCIAS
# ==========================================
def reference_path(form, view):
    return os.path.join(REFERENCE_DIR, f'taegeuk_{form}_{view.lower()}.npz')


def load_reference(form, view):
    data = np.load(reference_path(form, view))
    xy = clean_keypoints(data['xy'], data['conf'])
    xy = resample(xy, float(data['fps']))
    return {
        'form': form, 'view': view, 'xy': xy,
        'norm': normalize_pose(xy),
        'size': tuple(int(v) for v in data['size']), 'source_fps': float(data['fps']),
        'video': str(data['video']),
    }


def available_references():
    return [(f, v) for f in FORMS for v in VIEWS if os.path.exists(reference_path(f, v))]


# ==========================================
# AVALIAÇÃO
# ==========================================
# Calibração da nota (distância média por junta, em comprimentos de tronco):
# GOOD_DIST ~ diferença entre dois detectores de pose no MESMO vídeo (execução idêntica)
# BAD_DIST  ~ diferença típica entre Taegeuks diferentes (movimento errado)
GOOD_DIST = 0.20
BAD_DIST = 0.50
# Quantas juntas (as piores) entram na nota de cada movimento
WORST_JOINTS = 3
# Diferença de direção de um segmento (graus) a partir da qual vira um comentário de correção
ANGLE_TOLERANCE = 25.0
MAX_POSE_COMMENTS = 3
# Ritmo: duração do movimento do usuário / da referência fora desta faixa vira comentário
TIMING_RANGE = (0.6, 1.7)


def distance_to_score(dist):
    """0-100: 100 até GOOD_DIST, 0 a partir de BAD_DIST, linear entre os dois."""
    return float(np.clip((BAD_DIST - dist) / (BAD_DIST - GOOD_DIST), 0, 1) * 100)


def _segment_feedback(name, diff_deg, user_end, target_end):
    """
    Texto de correção para um segmento. user_end é onde a ponta do segmento está; target_end
    é onde estaria com a direção da referência (partindo da mesma origem, mesmo comprimento).
    Coordenadas normalizadas: origem no meio do quadril, y cresce para baixo.
    """
    dx, dy = user_end - target_end
    if abs(dy) >= abs(dx):
        hint = "raise it" if dy > 0 else "lower it"
    else:
        hint = "bring it closer to the body" if abs(user_end[0]) > abs(target_end[0]) else "extend it further out"
    return f"{name} is {abs(diff_deg):.0f}° off — {hint}"


def prepare_user(xy, conf, fps):
    """Limpa e reamostra os keypoints do usuário para ANALYSIS_FPS."""
    return resample(clean_keypoints(xy, conf), fps)


def match_references(user_xy, frame_width, references, try_mirror=True):
    """
    Compara o usuário com cada referência (e com o vídeo espelhado).
    Retorna a lista de candidatos ordenada do melhor para o pior.
    """
    variants = [(False, user_xy)]
    if try_mirror:
        variants.append((True, mirror(user_xy, frame_width)))
    variant_norms = [(m, xy, normalize_pose(xy)) for m, xy in variants]

    results = []
    for ref in references:
        for mirrored, xy, norm in variant_norms:
            if len(norm) < len(ref['norm']) // 3:
                continue  # vídeo curto demais para conter este Taegeuk mesmo a 3x de velocidade
            path, cost = dtw_subsequence(pose_distance_matrix(ref['norm'], norm))
            results.append({'ref': ref, 'mirrored': mirrored, 'user_xy': xy, 'user_norm': norm,
                            'path': path, 'cost': cost})
    if not results:
        raise ValueError("Vídeo curto demais para qualquer Taegeuk de referência")
    return sorted(results, key=lambda r: r['cost'])


def evaluate_match(match):
    """Nota e comentários por movimento para o melhor alinhamento encontrado."""
    ref, user_norm = match['ref'], match['user_norm']
    n_ref = len(ref['norm'])
    ref2user = ref_to_user_map(match['path'], n_ref)
    ref_dirs = segment_directions(ref['norm'])
    user_dirs = segment_directions(user_norm)
    # Distâncias em todas as células do caminho: cada frame do usuário entra na conta,
    # inclusive os que o alinhamento comprimiu num mesmo frame da referência
    path = np.asarray(match['path'])
    cell_joint_dist = np.linalg.norm(ref['norm'][path[:, 0]] - user_norm[path[:, 1]], axis=-1)  # (cells, 12)

    moves, _ = segment_moves(ref['norm'])
    report_moves = []
    for k, (a, b, key) in enumerate(moves, start=1):
        # Mediana: chutes rápidos borram a imagem e o detector erra a perna por alguns frames;
        # a mediana ignora esses picos curtos e reflete a execução do movimento como um todo
        in_move = (path[:, 0] >= a) & (path[:, 0] <= b)
        # Por junta: mediana ao longo do movimento (chutes borrados geram picos curtos que a
        # mediana ignora). Nota: média das WORST_JOINTS piores juntas — um erro de técnica
        # costuma estar num braço ou numa perna, e a média das 12 juntas diluiria esse erro.
        joint_med = np.median(cell_joint_dist[in_move], axis=0)
        dist = float(np.sort(joint_med)[-WORST_JOINTS:].mean())
        # Fase final do movimento (a técnica e a postura que os juízes olham): diferença de
        # direção de cada segmento, mediana sobre os últimos 40% do movimento. Usar um único
        # frame alinhado deixaria o comentário à mercê de pequenos deslizes do alinhamento.
        end_phase = in_move & (path[:, 0] >= b - max(2, int(0.4 * (b - a))))
        ri, uj = path[end_phase, 0], path[end_phase, 1]
        diffs = np.median((user_dirs[uj] - ref_dirs[ri] + 180) % 360 - 180, axis=0)
        user_key = ref2user[key]

        comments = []
        for idx in np.argsort(-np.abs(diffs)):
            if abs(diffs[idx]) < ANGLE_TOLERANCE or len(comments) >= MAX_POSE_COMMENTS:
                break
            name = list(SEGMENTS)[idx]
            o, e = SEGMENTS[name]
            seg_len = np.linalg.norm(user_norm[user_key, e] - user_norm[user_key, o])
            ref_vec = ref['norm'][key, e] - ref['norm'][key, o]
            target_end = user_norm[user_key, o] + ref_vec / max(np.linalg.norm(ref_vec), 1e-6) * seg_len
            comments.append(_segment_feedback(name, diffs[idx], user_norm[user_key, e], target_end))

        user_dur = (ref2user[b] - ref2user[a]) / ANALYSIS_FPS
        ref_dur = (b - a) / ANALYSIS_FPS
        ratio = user_dur / ref_dur if ref_dur > 0 else 1.0
        if ratio < TIMING_RANGE[0]:
            comments.append(f"too fast: {user_dur:.1f}s vs {ref_dur:.1f}s in the reference")
        elif ratio > TIMING_RANGE[1]:
            comments.append(f"too slow: {user_dur:.1f}s vs {ref_dur:.1f}s in the reference")

        worst_joint = JOINTS[int(np.argmax(joint_med))]
        report_moves.append({
            'move': k,
            'user_start_s': round(ref2user[a] / ANALYSIS_FPS, 2),
            'user_end_s': round(ref2user[b] / ANALYSIS_FPS, 2),
            'score': round(distance_to_score(dist), 1),
            'distance': round(dist, 3),
            'worst_joint': worst_joint,
            'comments': comments,
        })

    scores = [m['score'] for m in report_moves]
    return {
        'form': ref['form'], 'view': ref['view'], 'mirrored': match['mirrored'],
        'alignment_cost': round(match['cost'], 3),
        'overall_score': round(float(np.mean(scores)), 1) if scores else 0.0,
        'user_start_s': round(match['path'][0][1] / ANALYSIS_FPS, 2),
        'user_end_s': round(match['path'][-1][1] / ANALYSIS_FPS, 2),
        'moves': report_moves,
        'ref2user': ref2user,
        'moves_ref': moves,
    }


def evaluate_video(video_path, detector, form=None, view=None, on_progress=None):
    """Avalia um vídeo de Poomsae. form/view None = detectar automaticamente."""
    raw_xy, conf, fps, size = extract_keypoints(video_path, detector, on_progress)
    user_xy = prepare_user(raw_xy, conf, fps)

    candidates = [(f, v) for f, v in available_references()
                  if (form is None or f == form) and (view is None or v.lower() == view.lower())]
    if not candidates:
        raise FileNotFoundError("Nenhuma referência encontrada. Rode build_references.py primeiro.")
    references = [load_reference(f, v) for f, v in candidates]

    matches = match_references(user_xy, size[0], references)
    best = matches[0]
    report = evaluate_match(best)

    # Confiança da identificação do Taegeuk: diferença para o melhor Taegeuk diferente
    other = next((m for m in matches if m['ref']['form'] != best['ref']['form']), None)
    report['form_ranking'] = [
        {'form': m['ref']['form'], 'view': m['ref']['view'], 'mirrored': m['mirrored'], 'cost': round(m['cost'], 3)}
        for m in matches[:5]]
    report['form_margin'] = round(other['cost'] - best['cost'], 3) if other else None
    report['video'] = {'path': video_path, 'fps': fps, 'size': size, 'frames': len(raw_xy)}
    return report, best
