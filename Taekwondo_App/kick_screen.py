"""Tela de detecção de chutes em tempo real (webcam)."""
import tkinter as tk
from tkinter import ttk

import cv2 as cv

from widgets import COLORS, FONT, VideoPanel

# Ossos do esqueleto COCO (índices dos 17 keypoints do YOLO), só corpo, sem rosto
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]
STATUS_COLORS = {"KICKING!": COLORS['good'], "NO KICK": COLORS['bad']}


class KickScreen(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.stream = None
        self.processor = None
        self.loop_id = None
        self.kick_count = 0
        self.prev_label = None
        self.current_peak = 0.0
        self._build()

    # ---------- layout ----------
    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill='x', padx=16, pady=(12, 8))
        ttk.Button(top, text="← Back", command=self.app.show_home).pack(side='left')
        ttk.Label(top, text="Kick Detection", font=(FONT, 16, 'bold')).pack(side='left', padx=16)

        body = ttk.Frame(self)
        body.pack(fill='both', expand=True, padx=16, pady=(0, 16))
        self.video = VideoPanel(body, placeholder="Camera off")
        self.video.pack(side='left', fill='both', expand=True)

        side = ttk.Frame(body, style='Panel.TFrame', width=self.app.px(300))
        side.pack(side='right', fill='y', padx=(12, 0))
        side.pack_propagate(False)

        self.status_label = tk.Label(side, text="—", font=(FONT, 28, 'bold'), bg=COLORS['panel'], fg=COLORS['muted'])
        self.status_label.pack(fill='x', pady=(24, 4))
        ttk.Label(side, text="status", style='Muted.TLabel').pack()

        self.count_label = tk.Label(side, text="0", font=(FONT, 40, 'bold'), bg=COLORS['panel'], fg=COLORS['text'])
        self.count_label.pack(pady=(24, 0))
        ttk.Label(side, text="kicks", style='Muted.TLabel').pack()

        self.speed_label = ttk.Label(side, text="Speed: 0.00", style='Panel.TLabel', font=(FONT, 12))
        self.speed_label.pack(pady=(24, 2))
        self.peak_label = ttk.Label(side, text="Last kick peak: —", style='Panel.TLabel', font=(FONT, 12))
        self.peak_label.pack()
        ttk.Label(side, text="speed in torso lengths per second", style='Muted.TLabel', font=(FONT, 8)).pack()

        controls = ttk.Frame(side, style='Panel.TFrame')
        controls.pack(side='bottom', fill='x', padx=16, pady=16)
        cam_row = ttk.Frame(controls, style='Panel.TFrame')
        cam_row.pack(fill='x', pady=(0, 8))
        ttk.Label(cam_row, text="Camera", style='Panel.TLabel').pack(side='left')
        cam = ttk.Combobox(cam_row, textvariable=self.app.camera_var, values=["0", "1", "2", "3"], width=5, state='readonly')
        cam.pack(side='right')
        cam.bind('<<ComboboxSelected>>', lambda e: self._restart_camera())
        self.mirror_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Mirror view", variable=self.mirror_var).pack(anchor='w', pady=(0, 8))
        ttk.Button(controls, text="Reset counter", command=self._reset_counter).pack(fill='x', pady=(0, 8))
        self.toggle_button = ttk.Button(controls, text="Start camera", style='Accent.TButton', command=self._toggle)
        self.toggle_button.pack(fill='x')
        self.info_label = ttk.Label(controls, text="", style='Muted.TLabel', wraplength=self.app.px(260))
        self.info_label.pack(fill='x', pady=(8, 0))

    # ---------- ciclo de vida ----------
    def on_show(self):
        self._start()

    def on_hide(self):
        self._stop()

    def _toggle(self):
        if self.stream:
            self._stop()
        else:
            self._start()

    def _restart_camera(self):
        if self.stream:
            self._stop()
            self._start()

    def _start(self):
        if self.stream:
            return
        self.toggle_button.configure(state='disabled')
        self.video.clear("Loading models and camera...")
        camera = int(self.app.camera_var.get())

        def work(progress):
            from core.camera import VideoStream
            from predict_realtime import AIProcessor  # importa TensorFlow: só quando precisa
            yolo, kick_model, scaler = self.app.models.kick_models()
            stream = VideoStream(src=camera)
            if not stream.is_opened():
                stream.stop()
                raise RuntimeError(f"Could not open camera {camera}. Try another camera number.")
            stream.start()
            return stream, AIProcessor(stream, yolo, kick_model, scaler)

        self.app.tasks.run(work, on_done=self._on_started, on_error=self._on_error)

    def _on_started(self, result):
        self.stream, self.processor = result
        if not self.winfo_ismapped():  # usuário saiu da tela enquanto carregava
            self._stop()
            return
        self.processor.start()
        self.toggle_button.configure(text="Stop camera", style='Danger.TButton', state='normal')
        self.info_label.configure(text="Stand so your whole body, especially hips and feet, is visible.")
        self._loop()

    def _on_error(self, error):
        self.toggle_button.configure(text="Start camera", style='Accent.TButton', state='normal')
        self.video.clear(f"⚠ {error}")

    def _stop(self):
        if self.loop_id:
            self.after_cancel(self.loop_id)
            self.loop_id = None
        if self.processor:
            self.processor.stopped = True
            self.processor.join(timeout=2.0)
            self.processor = None
        if self.stream:
            self.stream.stop()
            self.stream = None
        self.toggle_button.configure(text="Start camera", style='Accent.TButton', state='normal')
        self.video.clear("Camera off")
        self.status_label.configure(text="—", fg=COLORS['muted'])

    def _reset_counter(self):
        self.kick_count = 0
        self.count_label.configure(text="0")
        self.peak_label.configure(text="Last kick peak: —")

    # ---------- atualização ----------
    def _loop(self):
        frame = self.stream.read()
        if frame is not None:
            frame = frame.copy()
            with self.processor.lock:
                status = self.processor.current_status
                label = self.processor.current_label
                speed = self.processor.current_speed
                pose = self.processor.current_pose

            if label == 1:
                if self.prev_label != 1:
                    self.kick_count += 1
                    self.current_peak = 0.0
                self.current_peak = max(self.current_peak, speed)
            elif self.prev_label == 1:
                self.peak_label.configure(text=f"Last kick peak: {self.current_peak:.2f}")
            self.prev_label = label

            if pose is not None:
                self._draw_skeleton(frame, pose, label)
            if self.mirror_var.get():
                frame = cv.flip(frame, 1)
            self.video.show(frame)

            self.status_label.configure(text=status, fg=STATUS_COLORS.get(status, COLORS['muted']))
            self.count_label.configure(text=str(self.kick_count))
            self.speed_label.configure(text=f"Speed: {speed:.2f}")
        self.loop_id = self.after(15, self._loop)

    @staticmethod
    def _draw_skeleton(frame, pose, label):
        xy, conf = pose
        color = (94, 197, 34) if label == 1 else (255, 255, 255)
        thickness = max(2, frame.shape[0] // 240)
        for a, b in BONES:
            if conf[a] > 0.3 and conf[b] > 0.3:
                cv.line(frame, tuple(map(int, xy[a])), tuple(map(int, xy[b])), color, thickness, cv.LINE_AA)
        for i in range(5, 17):
            if conf[i] > 0.3:
                cv.circle(frame, tuple(map(int, xy[i])), thickness + 2, color, -1, cv.LINE_AA)
