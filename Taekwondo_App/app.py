"""
Taekwondo Trainer: app desktop com duas funções.

- Kick Detection: detecção de chutes em tempo real pela webcam.
- Poomsae Evaluation: nota e correções de um Taegeuk (1-8) a partir de um vídeo gravado ou aberto.

Uso:
    python Taekwondo_App/app.py      (ou dê dois cliques em run_app.bat)
"""
import os
import sys
import threading
import tkinter as tk
import warnings
from tkinter import ttk

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
for path in (ROOT_DIR, os.path.join(ROOT_DIR, "Taekwondo_Kick_Pose_Predict"),
             os.path.join(ROOT_DIR, "Taekwondo_Poomsae_Evaluator"), APP_DIR):
    if path not in sys.path:
        sys.path.append(path)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Nitidez em telas com escala do Windows (ex.: 125%): sem isso o Windows estica a janela
# inteira (fica borrada) e as medidas em pixels não batem com a tela real
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
warnings.filterwarnings("ignore", category=UserWarning)

from widgets import COLORS, FONT, TaskRunner, setup_theme
from kick_screen import KickScreen
from poomsae_screen import PoomsaeScreen

YOLO_PATH = os.path.join(ROOT_DIR, "Taekwondo_Kick_Pose_Predict", "yolov8n-pose.pt")


class ModelStore:
    """Carrega cada modelo uma única vez, na primeira vez que alguma tela precisa dele."""

    def __init__(self):
        self._lock = threading.Lock()
        self._yolo = None
        self._kick = None

    def yolo(self):
        with self._lock:
            if self._yolo is None:
                from ultralytics import YOLO
                self._yolo = YOLO(YOLO_PATH)
            return self._yolo

    def kick_models(self):
        """(yolo, kick_model, scaler). Importa TensorFlow, então só é chamado na tela de chutes."""
        yolo = self.yolo()
        with self._lock:
            if self._kick is None:
                import joblib
                import tensorflow as tf
                import predict_realtime
                model_dir = os.path.join(predict_realtime.SCRIPT_DIR, "model")
                self._kick = (tf.keras.models.load_model(os.path.join(model_dir, "kick_detection_model.h5"), compile=False),
                              joblib.load(os.path.join(model_dir, "scaler.pkl")))
            return (yolo,) + self._kick


class HomeScreen(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        center = ttk.Frame(self)
        center.place(relx=0.5, rely=0.45, anchor='center')
        ttk.Label(center, text="Taekwondo Trainer", style='Title.TLabel').pack()
        ttk.Label(center, text="Choose what you want to train", style='Subtitle.TLabel').pack(pady=(4, 32))

        cards = ttk.Frame(center)
        cards.pack()
        self._card(cards, "🦵", "Kick Detection",
                   "Live webcam: detects kicks as they happen,\ncounts them and measures their speed.",
                   lambda: app.show('kick'), app.px).pack(side='left', padx=12)
        self._card(cards, "🥋", "Poomsae Evaluation",
                   "Record or open a video of a Taegeuk (1–8)\nand get a score and corrections for each move.",
                   lambda: app.show('poomsae'), app.px).pack(side='left', padx=12)

    @staticmethod
    def _card(parent, icon, title, text, command, px):
        card = tk.Frame(parent, bg=COLORS['panel'], width=px(340), height=px(260), cursor='hand2',
                        highlightthickness=2, highlightbackground=COLORS['panel'])
        card.pack_propagate(False)
        widgets = [
            tk.Label(card, text=icon, font=(FONT, 40), bg=COLORS['panel'], fg=COLORS['text']),
            tk.Label(card, text=title, font=(FONT, 17, 'bold'), bg=COLORS['panel'], fg=COLORS['text']),
            tk.Label(card, text=text, font=(FONT, 10), bg=COLORS['panel'], fg=COLORS['muted'], justify='center'),
        ]
        widgets[0].pack(pady=(28, 4))
        widgets[1].pack()
        widgets[2].pack(pady=(10, 0))
        for w in [card] + widgets:
            w.bind('<Button-1>', lambda e: command())
            w.bind('<Enter>', lambda e: card.configure(highlightbackground=COLORS['accent']))
            w.bind('<Leave>', lambda e: card.configure(highlightbackground=COLORS['panel']))
        return card


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Taekwondo Trainer")
        # Fator de escala da tela (1.0 = 96 dpi; 1.25 = escala de 125% do Windows)
        self.ui_scale = self.winfo_fpixels('1i') / 96
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(int(sw * 0.9), self.px(1500)), min(int(sh * 0.85), self.px(900))
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 3)}")
        self.minsize(min(w, self.px(960)), min(h, self.px(600)))
        setup_theme(self, self.ui_scale)

        self.models = ModelStore()
        self.tasks = TaskRunner(self)
        self.camera_var = tk.StringVar(value="0")  # webcam compartilhada pelas duas telas

        container = ttk.Frame(self)
        container.pack(fill='both', expand=True)
        self.screens = {
            'home': HomeScreen(container, self),
            'kick': KickScreen(container, self),
            'poomsae': PoomsaeScreen(container, self),
        }
        self.current = None
        self.show('home')
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def px(self, value):
        """Converte um tamanho pensado para 96 dpi para pixels desta tela."""
        return int(round(value * self.ui_scale))

    def show(self, name):
        if self.current:
            screen = self.screens[self.current]
            if hasattr(screen, 'on_hide'):
                screen.on_hide()
            screen.pack_forget()
        self.current = name
        screen = self.screens[name]
        screen.pack(fill='both', expand=True)
        if hasattr(screen, 'on_show'):
            self.after(50, screen.on_show)  # deixa o layout calcular o tamanho do painel de vídeo antes

    def show_home(self):
        self.show('home')

    def _on_close(self):
        if self.current and hasattr(self.screens[self.current], 'on_hide'):
            self.screens[self.current].on_hide()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
