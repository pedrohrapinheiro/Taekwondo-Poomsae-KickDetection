"""Tela de avaliação de Poomsae: escolher ou gravar um vídeo, avaliar e ver o resultado."""
import os
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2 as cv
import numpy as np

from widgets import COLORS, FONT, VideoPanel

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
COUNTDOWN_S = 5
MAX_RECORDING_S = 240
FORM_CHOICES = ["Auto-detect"] + [f"Taegeuk {i}" for i in range(1, 9)]
VIEW_CHOICES = ["Auto-detect", "Frontal", "Lateral"]


def score_color(score):
    return COLORS['good'] if score >= 85 else COLORS['warn'] if score >= 70 else COLORS['bad']


def fmt_time(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


class PoomsaeScreen(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.video_path = None
        self.feedback_path = None
        self.report = None
        self.busy = False
        # Gravação
        self.camera = None
        self.rec_state = None      # None | 'countdown' | 'recording'
        self.rec_frames = []       # (timestamp, jpeg)
        self.rec_t0 = 0.0
        self.rec_last_id = -1
        # Player
        self.cap = None
        self.playing = False
        self.player_fps = 30.0
        self.player_total = 0
        self._slider_internal = False
        self.loop_id = None
        self._build()

    # ---------- layout ----------
    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill='x', padx=16, pady=(12, 8))
        self.back_button = ttk.Button(top, text="← Back", command=self.app.show_home)
        self.back_button.pack(side='left')
        ttk.Label(top, text="Poomsae Evaluation", font=(FONT, 16, 'bold')).pack(side='left', padx=16)

        body = ttk.Frame(self)
        body.pack(fill='both', expand=True, padx=16, pady=(0, 16))

        left = ttk.Frame(body)
        left.pack(side='left', fill='both', expand=True)
        self.video = VideoPanel(left, placeholder="Open a video or record one to start")
        self.video.pack(fill='both', expand=True)
        player = ttk.Frame(left)
        player.pack(fill='x', pady=(8, 0))
        self.play_button = ttk.Button(player, text="▶ Play", width=9, command=self._toggle_play, state='disabled')
        self.play_button.pack(side='left')
        self.slider_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(player, from_=0, to=1, variable=self.slider_var, command=self._on_slider)
        self.slider.pack(side='left', fill='x', expand=True, padx=8)
        self.time_label = ttk.Label(player, text="0:00.0 / 0:00.0", width=16)
        self.time_label.pack(side='left')

        px = self.app.px
        side = ttk.Frame(body, style='Panel.TFrame', width=px(460))
        side.pack(side='right', fill='y', padx=(12, 0))
        side.pack_propagate(False)
        pad = {'padx': 16}

        ttk.Label(side, text="1. Video", style='Section.TLabel').pack(anchor='w', pady=(16, 6), **pad)
        row = ttk.Frame(side, style='Panel.TFrame')
        row.pack(fill='x', **pad)
        self.open_button = ttk.Button(row, text="Open video…", command=self._open_video)
        self.open_button.pack(side='left', fill='x', expand=True)
        self.record_button = ttk.Button(row, text="● Record with webcam", command=self._toggle_recording)
        self.record_button.pack(side='left', fill='x', expand=True, padx=(8, 0))
        self.file_label = ttk.Label(side, text="No video selected", style='Muted.TLabel', wraplength=px(420))
        self.file_label.pack(anchor='w', pady=(6, 0), **pad)

        ttk.Label(side, text="2. Options", style='Section.TLabel').pack(anchor='w', pady=(16, 6), **pad)
        opts = ttk.Frame(side, style='Panel.TFrame')
        opts.pack(fill='x', **pad)
        ttk.Label(opts, text="Form", style='Panel.TLabel').grid(row=0, column=0, sticky='w')
        self.form_var = tk.StringVar(value=FORM_CHOICES[0])
        ttk.Combobox(opts, textvariable=self.form_var, values=FORM_CHOICES, state='readonly', width=14).grid(row=0, column=1, padx=(8, 16))
        ttk.Label(opts, text="Camera", style='Panel.TLabel').grid(row=0, column=2, sticky='w')
        self.view_var = tk.StringVar(value=VIEW_CHOICES[0])
        ttk.Combobox(opts, textvariable=self.view_var, values=VIEW_CHOICES, state='readonly', width=12).grid(row=0, column=3, padx=(8, 0))

        self.eval_button = ttk.Button(side, text="Evaluate", style='Accent.TButton', command=self._evaluate, state='disabled')
        self.eval_button.pack(fill='x', pady=(16, 6), **pad)
        self.progress = ttk.Progressbar(side, mode='determinate', maximum=1.0)
        self.progress.pack(fill='x', **pad)
        self.progress_label = ttk.Label(side, text="", style='Muted.TLabel')
        self.progress_label.pack(anchor='w', **pad)

        ttk.Label(side, text="3. Result", style='Section.TLabel').pack(anchor='w', pady=(12, 0), **pad)
        score_row = ttk.Frame(side, style='Panel.TFrame')
        score_row.pack(fill='x', **pad)
        self.score_label = tk.Label(score_row, text="—", font=(FONT, 36, 'bold'), bg=COLORS['panel'], fg=COLORS['muted'])
        self.score_label.pack(side='left')
        self.summary_label = ttk.Label(score_row, text="", style='Panel.TLabel', wraplength=px(300), justify='left')
        self.summary_label.pack(side='left', padx=12)
        self.warning_label = tk.Label(side, text="", bg=COLORS['panel'], fg=COLORS['warn'], wraplength=px(420),
                                      justify='left', font=(FONT, 9))
        self.warning_label.pack(anchor='w', **pad)

        # Rodapé empacotado antes da tabela: a tabela ocupa só o espaço que sobrar
        self.external_button = ttk.Button(side, text="Open feedback video in player", command=self._open_external, state='disabled')
        self.external_button.pack(side='bottom', fill='x', pady=(8, 16), **pad)
        ttk.Label(side, text="Click a move to jump to it in the video.", style='Muted.TLabel').pack(side='bottom', anchor='w', **pad)
        table = ttk.Frame(side, style='Panel.TFrame')
        table.pack(fill='both', expand=True, pady=(8, 0), **pad)
        cols = ('move', 'time', 'score', 'notes')
        self.tree = ttk.Treeview(table, columns=cols, show='headings', selectmode='browse')
        for col, title, width, anchor in [('move', '#', 36, 'center'), ('time', 'Time', 70, 'center'),
                                          ('score', 'Score', 52, 'center'), ('notes', 'Notes', 600, 'w')]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=px(width), anchor=anchor, stretch=(col == 'notes'))
        for tag, color in (('good', COLORS['good']), ('warn', COLORS['warn']), ('bad', COLORS['bad'])):
            self.tree.tag_configure(tag, foreground=color)
        scroll = ttk.Scrollbar(table, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        self.tree.bind('<<TreeviewSelect>>', self._on_move_selected)

    # ---------- ciclo de vida ----------
    def on_show(self):
        pass

    def on_hide(self):
        self._stop_player()
        if self.rec_state:
            self._stop_camera()

    def _set_busy(self, busy):
        self.busy = busy
        state = 'disabled' if busy else 'normal'
        for b in (self.back_button, self.open_button, self.eval_button):
            b.configure(state=state)
        if busy:
            self.record_button.configure(state='disabled')
        else:
            self.record_button.configure(state='normal')
            if not self.video_path:
                self.eval_button.configure(state='disabled')

    # ---------- escolher vídeo ----------
    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Choose a Poomsae video",
            filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")])
        if path:
            self._select_video(path)

    def _select_video(self, path):
        self.video_path = path
        self.file_label.configure(text=os.path.basename(path))
        self.eval_button.configure(state='normal')
        self._clear_result()
        self._load_player(path, autoplay=False)

    # ---------- gravação ----------
    def _toggle_recording(self):
        if self.rec_state:
            self._finish_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        from core.camera import VideoStream
        self._stop_player()
        camera = VideoStream(src=int(self.app.camera_var.get()))
        if not camera.is_opened():
            camera.stop()
            messagebox.showerror("Camera", "Could not open the webcam.")
            return
        self.camera = camera.start()
        self.rec_frames, self.rec_last_id = [], -1
        self.rec_state, self.rec_t0 = 'countdown', time.time()
        self.record_button.configure(text="■ Stop recording", style='Danger.TButton')
        for b in (self.back_button, self.open_button, self.eval_button):
            b.configure(state='disabled')
        self.progress_label.configure(text="Get into position: full body visible, camera facing the form's start direction")
        self._record_loop()

    def _record_loop(self):
        frame, frame_id = self.camera.read_with_id()
        now = time.time()
        if frame is not None:
            preview = frame.copy()
            h, w = preview.shape[:2]
            if self.rec_state == 'countdown':
                remaining = COUNTDOWN_S - (now - self.rec_t0)
                if remaining <= 0:
                    self.rec_state, self.rec_t0 = 'recording', now
                else:
                    text = str(int(np.ceil(remaining)))
                    size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, h / 120, max(2, h // 60))[0]
                    cv.putText(preview, text, ((w - size[0]) // 2, (h + size[1]) // 2), cv.FONT_HERSHEY_SIMPLEX,
                               h / 120, (255, 255, 255), max(2, h // 60), cv.LINE_AA)
            if self.rec_state == 'recording':
                if frame_id != self.rec_last_id:
                    self.rec_last_id = frame_id
                    ok, jpg = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        self.rec_frames.append((now, jpg))
                elapsed = now - self.rec_t0
                cv.circle(preview, (30, 30), 10, (0, 0, 255), -1)
                cv.putText(preview, f"REC {fmt_time(elapsed)}", (50, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                           (0, 0, 255), 2, cv.LINE_AA)
                if elapsed >= MAX_RECORDING_S:
                    self._finish_recording()
                    return
            self.video.show(preview)
        self.loop_id = self.after(10, self._record_loop)

    def _stop_camera(self):
        if self.loop_id:
            self.after_cancel(self.loop_id)
            self.loop_id = None
        if self.camera:
            self.camera.stop()
            self.camera = None
        self.rec_state = None
        self.record_button.configure(text="● Record with webcam", style='TButton')
        self._set_busy(False)

    def _finish_recording(self):
        frames = self.rec_frames if self.rec_state == 'recording' else []
        self._stop_camera()
        self.rec_frames = []
        if len(frames) < 30:
            self.video.clear("Recording too short")
            self.progress_label.configure(text="")
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"poomsae_{datetime.now():%Y%m%d_%H%M%S}.mp4")
        self._set_busy(True)
        self.progress_label.configure(text="Saving recording...")

        def work(progress):
            # FPS real medido: a webcam nem sempre entrega o FPS que anuncia, e o ritmo do
            # Poomsae (comentários de "rápido/lento") depende de o vídeo ter a velocidade certa
            fps = (len(frames) - 1) / max(frames[-1][0] - frames[0][0], 1e-6)
            first = cv.imdecode(frames[0][1], cv.IMREAD_COLOR)
            writer = cv.VideoWriter(path, cv.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
            for i, (_, jpg) in enumerate(frames):
                writer.write(cv.imdecode(jpg, cv.IMREAD_COLOR))
                if i % 30 == 0:
                    progress(i / len(frames))
            writer.release()
            return path

        def done(saved):
            self._set_busy(False)
            self.progress['value'] = 0
            self.progress_label.configure(text="Recording saved. Press Evaluate.")
            self._select_video(saved)

        self.app.tasks.run(work, on_done=done, on_error=self._on_error,
                           on_progress=lambda f: self.progress.configure(value=f))

    # ---------- avaliação ----------
    def _evaluate(self):
        if not self.video_path:
            return
        self._stop_player()
        self._clear_result()
        self._set_busy(True)
        path = self.video_path
        form = None if self.form_var.get() == FORM_CHOICES[0] else int(self.form_var.get().split()[-1])
        view = None if self.view_var.get() == VIEW_CHOICES[0] else self.view_var.get()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(path))[0] + "_feedback.mp4")

        def work(progress):
            import poomsae as P
            import evaluate_poomsae as E
            from core.pose_engine import PoseDetector
            progress("Loading pose model...", 0.0)
            detector = PoseDetector(self.app.models.yolo())
            report, match = P.evaluate_video(
                path, detector, form=form, view=view,
                on_progress=lambda i, n: progress("Analyzing poses...", 0.75 * i / max(n, 1)))
            progress("Rendering feedback video...", 0.75)
            E.render_video(path, out, report, match,
                           on_progress=lambda i, n: progress("Rendering feedback video...", 0.75 + 0.25 * i / max(n, 1)))
            return report, out

        self.app.tasks.run(work, on_done=self._on_evaluated, on_error=self._on_error, on_progress=self._on_progress)

    def _on_progress(self, text, fraction):
        self.progress_label.configure(text=text)
        self.progress['value'] = fraction

    def _on_error(self, error):
        self._set_busy(False)
        self.progress['value'] = 0
        self.progress_label.configure(text="")
        messagebox.showerror("Error", str(error))

    def _on_evaluated(self, result):
        import evaluate_poomsae as E
        report, out = result
        self._set_busy(False)
        self.report, self.feedback_path = report, out
        self.progress['value'] = 1.0
        self.progress_label.configure(text="Done")

        score = report['overall_score']
        self.score_label.configure(text=f"{score:.0f}", fg=score_color(score))
        self.summary_label.configure(
            text=f"Taegeuk {report['form']} · {report['view'].lower()} view\n"
                 f"{len(report['moves'])} moves, from {fmt_time(report['user_start_s'])} to {fmt_time(report['user_end_s'])}")
        warnings = []
        if report['alignment_cost'] > E.NOT_A_FORM_COST:
            warnings.append("⚠ Poor match: the video may not contain a complete Taegeuk, or the camera angle "
                            "differs a lot from the reference. Scores are unreliable.")
        if report['form_margin'] is not None and report['form_margin'] < E.CONFIDENT_MARGIN:
            warnings.append("⚠ Unsure which Taegeuk this is — choose the form in Options and evaluate again.")
        if report['mirrored']:
            warnings.append("Note: the video looks mirrored (selfie camera?), or the form was done on the opposite side.")
        self.warning_label.configure(text="\n".join(warnings))

        for m in report['moves']:
            tag = 'good' if m['score'] >= 85 else 'warn' if m['score'] >= 70 else 'bad'
            self.tree.insert('', 'end', iid=str(m['move']), tags=(tag,),
                             values=(m['move'], fmt_time(m['user_start_s']), f"{m['score']:.0f}", E.move_notes(m)))
        self.external_button.configure(state='normal')
        self._load_player(out, autoplay=True)

    def _clear_result(self):
        self.report = None
        self.score_label.configure(text="—", fg=COLORS['muted'])
        self.summary_label.configure(text="")
        self.warning_label.configure(text="")
        self.tree.delete(*self.tree.get_children())
        self.external_button.configure(state='disabled')
        self.progress['value'] = 0

    def _on_move_selected(self, _event):
        sel = self.tree.selection()
        if not sel or not self.report or not self.cap:
            return
        move = self.report['moves'][int(sel[0]) - 1]
        self._seek_seconds(move['user_start_s'])
        if not self.playing:
            self._toggle_play()

    def _open_external(self):
        if self.feedback_path and os.path.exists(self.feedback_path):
            os.startfile(self.feedback_path)

    # ---------- player ----------
    def _load_player(self, path, autoplay):
        self._stop_player()
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            self.video.clear("Could not open the video")
            return
        self.cap = cap
        self.player_fps = cap.get(cv.CAP_PROP_FPS) or 30.0
        self.player_total = max(1, int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
        self.slider.configure(to=self.player_total - 1)
        self.play_button.configure(state='normal')
        self._show_next_frame()
        if autoplay:
            self._toggle_play()

    def _stop_player(self):
        self.playing = False
        if self.loop_id and not self.rec_state:
            self.after_cancel(self.loop_id)
            self.loop_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.play_button.configure(text="▶ Play", state='disabled')

    def _toggle_play(self):
        if not self.cap:
            return
        self.playing = not self.playing
        self.play_button.configure(text="❚❚ Pause" if self.playing else "▶ Play")
        if self.playing:
            if int(self.cap.get(cv.CAP_PROP_POS_FRAMES)) >= self.player_total - 1:
                self._seek_frame(0)
            self._play_loop()

    def _play_loop(self):
        if not self.playing or not self.cap:
            return
        start = time.time()
        if not self._show_next_frame():
            self.playing = False
            self.play_button.configure(text="▶ Play")
            return
        delay = max(1, int(1000 / self.player_fps - (time.time() - start) * 1000))
        self.loop_id = self.after(delay, self._play_loop)

    def _show_next_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.video.show(frame)
        pos = int(self.cap.get(cv.CAP_PROP_POS_FRAMES)) - 1
        self._slider_internal = True
        self.slider_var.set(pos)
        self._slider_internal = False
        self.time_label.configure(text=f"{fmt_time(pos / self.player_fps)} / {fmt_time(self.player_total / self.player_fps)}")
        return True

    def _seek_frame(self, frame_idx):
        if not self.cap:
            return
        self.cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
        self._show_next_frame()

    def _seek_seconds(self, seconds):
        self._seek_frame(seconds * self.player_fps)

    def _on_slider(self, value):
        if self._slider_internal or not self.cap:
            return
        self._seek_frame(float(value))
