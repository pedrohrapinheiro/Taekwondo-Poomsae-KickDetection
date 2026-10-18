"""
Avalia a execução de um Poomsae (Taegeuk 1-8) num vídeo, comparando com a referência.

Uso:
    python Taekwondo_Poomsae_Evaluator/evaluate_poomsae.py VIDEO [--form N] [--view frontal|lateral]
                                                           [--output-video saida.mp4] [--json relatorio.json]

Para melhores resultados, filme o corpo inteiro com a câmera parada, posicionada como na
referência: de frente para a direção inicial do Poomsae (frontal) ou de lado (lateral).
"""
import argparse
import json
import os
import sys

import cv2 as cv
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from core.pose_engine import PoseDetector
import poomsae as P

MODEL_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "Taekwondo_Kick_Pose_Predict", "yolov8n-pose.pt")

BONES = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
         (6, 8), (8, 10), (7, 9), (9, 11)]
# Margem de custo para o segundo Taegeuk mais provável a partir da qual a identificação é confiável
CONFIDENT_MARGIN = 0.05
# Custo de alinhamento acima do qual o vídeo provavelmente não contém um Taegeuk completo.
# Na validação: execução correta 0.10-0.18, Taegeuk errado >= 0.22.
NOT_A_FORM_COST = 0.25


def put_label(frame, text, y, scale, color):
    """Texto com fundo escuro para ficar legível sobre qualquer vídeo."""
    font, thick = cv.FONT_HERSHEY_SIMPLEX, max(1, int(2 * scale))
    (tw, th), base = cv.getTextSize(text, font, 0.7 * scale, thick)
    cv.rectangle(frame, (10, y - th - 8), (10 + tw + 16, y + base + 4), (0, 0, 0), -1)
    cv.putText(frame, text, (18, y), font, 0.7 * scale, color, thick, cv.LINE_AA)


def move_notes(m):
    """Texto de observações de um movimento (usado no terminal e no app)."""
    if m['comments']:
        return "; ".join(m['comments'])
    return "" if m['score'] >= 70 else f"check {m['worst_joint'].replace('_', ' ')}"


def fmt_time(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def print_report(report):
    margin = report['form_margin']
    confidence = "confident" if margin is None or margin >= CONFIDENT_MARGIN else "UNCERTAIN, pass --form to choose"
    print()
    print(f"Taegeuk {report['form']} ({report['view'].lower()} view)  |  identification: {confidence}")
    if report['alignment_cost'] > NOT_A_FORM_COST:
        print(f"WARNING: poor match (cost {report['alignment_cost']:.2f}). The video may not contain a complete "
              "Taegeuk, or the camera angle differs a lot from the reference. Scores below are unreliable.")
    if report['mirrored']:
        print("Note: the video looks mirrored (selfie camera?) — or the form was performed on the opposite side.")
    print(f"Form detected from {fmt_time(report['user_start_s'])} to {fmt_time(report['user_end_s'])} in the video")
    print(f"\nOVERALL SCORE: {report['overall_score']:.0f}/100\n")
    print(f"{'Move':>4}  {'Time in video':<15} {'Score':>5}  Notes")
    for m in report['moves']:
        span = f"{fmt_time(m['user_start_s'])}-{fmt_time(m['user_end_s'])}"
        notes = move_notes(m)
        print(f"{m['move']:>4}  {span:<15} {m['score']:>5.0f}  {notes}")
    worst = [m for m in sorted(report['moves'], key=lambda m: m['score']) if m['score'] < 70][:3]
    if worst:
        print("\nMoves to work on: " + ", ".join(f"#{m['move']} ({m['score']:.0f})" for m in worst))
    print("\nOther candidates:", ", ".join(
        f"T{c['form']} {c['view'].lower()}{' mirrored' if c['mirrored'] else ''} (cost {c['cost']:.3f})"
        for c in report['form_ranking'][1:4]))


def render_video(video_path, output_path, report, match, on_progress=None):
    """Vídeo do usuário com o esqueleto dele (verde/vermelho) e o 'fantasma' da referência (branco)."""
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ref = match['ref']
    user_xy = match['user_xy']
    ref2user = report['ref2user']
    hip_mid, torso = P.pose_frame(user_xy)
    move_of_ref = np.zeros(len(ref['norm']), dtype=int)
    for k, (a, b, _) in enumerate(report['moves_ref']):
        move_of_ref[a:b + 1] = k + 1
    # Escala do texto/linhas pelo lado menor: vídeos verticais de celular são estreitos
    scale = max(0.6, min(w, h) / 720)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        u = min(len(user_xy) - 1, int(round(frame_idx * P.ANALYSIS_FPS / fps)))
        user_pts = user_xy[u]
        status = "Waiting for the form to start..."
        if ref2user[0] <= u <= ref2user[-1]:
            t = int(round(np.interp(u, ref2user, np.arange(len(ref2user)))))
            ghost = hip_mid[u] + ref['norm'][t] * torso[u]
            joint_err = np.linalg.norm(ref['norm'][t] - (user_pts - hip_mid[u]) / torso[u], axis=-1)
            if match['mirrored']:
                ghost = ghost.copy(); ghost[:, 0] = w - ghost[:, 0]
            for a, b in BONES:
                cv.line(frame, tuple(map(int, ghost[a])), tuple(map(int, ghost[b])), (255, 255, 255), int(6 * scale), cv.LINE_AA)
            k = move_of_ref[t]
            if k:
                m = report['moves'][k - 1]
                status = f"Taegeuk {report['form']} | move {k}/{len(report['moves'])} | score {m['score']:.0f}"
                for line, comment in enumerate(m['comments'][:2]):
                    put_label(frame, comment, int((80 + 36 * line) * scale), scale, (0, 255, 255))
            else:
                status = f"Taegeuk {report['form']} | ready stance"
        else:
            joint_err = np.zeros(12)
            if u > ref2user[-1]:
                status = f"Finished | overall score {report['overall_score']:.0f}/100"

        pts = user_pts.copy()
        if match['mirrored']:
            pts[:, 0] = w - pts[:, 0]
        for a, b in BONES:
            bad = max(joint_err[a], joint_err[b]) > 0.35
            cv.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])),
                    (0, 0, 255) if bad else (0, 255, 0), int(3 * scale), cv.LINE_AA)
        put_label(frame, status, int(40 * scale), scale, (255, 255, 0))
        writer.write(frame)
        frame_idx += 1
        if on_progress and frame_idx % 30 == 0:
            on_progress(frame_idx, total)
    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Taekwondo Poomsae (Taegeuk 1-8) video against a reference.")
    parser.add_argument("video")
    parser.add_argument("--form", type=int, choices=range(1, 9), help="Taegeuk number (default: detect automatically)")
    parser.add_argument("--view", choices=["frontal", "lateral"], help="camera view (default: detect automatically)")
    parser.add_argument("--output-video", help="save the video with the reference skeleton overlaid")
    parser.add_argument("--json", help="save the full report as JSON")
    args = parser.parse_args()

    detector = PoseDetector(YOLO(MODEL_PATH))
    print("Extracting poses...")
    report, match = P.evaluate_video(args.video, detector, form=args.form, view=args.view,
                                     on_progress=lambda i, n: print(f"  -> frame {i}/{n}"))
    print_report(report)

    if args.output_video:
        render_video(args.video, args.output_video, report, match)
        print(f"\nAnnotated video saved to {args.output_video}")
    if args.json:
        public = {k: v for k, v in report.items() if k not in ('ref2user', 'moves_ref')}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(public, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.json}")


if __name__ == "__main__":
    main()
