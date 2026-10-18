"""Componentes compartilhados do app: tema, painel de vídeo e execução em segundo plano."""
import queue
import threading
import tkinter as tk
import traceback
from tkinter import ttk

import cv2 as cv
from PIL import Image, ImageTk

COLORS = {
    'bg': '#1b1d23',
    'panel': '#262932',
    'panel_alt': '#2f333d',
    'text': '#e8e8ea',
    'muted': '#9a9ca5',
    'accent': '#3b82f6',
    'good': '#22c55e',
    'warn': '#eab308',
    'bad': '#ef4444',
}
FONT = 'Segoe UI'


def setup_theme(root, scale=1.0):
    root.configure(bg=COLORS['bg'])
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background=COLORS['bg'], foreground=COLORS['text'], font=(FONT, 10))
    style.configure('TFrame', background=COLORS['bg'])
    style.configure('Panel.TFrame', background=COLORS['panel'])
    style.configure('TLabel', background=COLORS['bg'], foreground=COLORS['text'])
    style.configure('Panel.TLabel', background=COLORS['panel'])
    style.configure('Muted.TLabel', background=COLORS['panel'], foreground=COLORS['muted'])
    style.configure('Title.TLabel', font=(FONT, 26, 'bold'))
    style.configure('Subtitle.TLabel', foreground=COLORS['muted'], font=(FONT, 12))
    style.configure('Section.TLabel', background=COLORS['panel'], font=(FONT, 11, 'bold'))
    style.configure('TButton', background=COLORS['panel_alt'], foreground=COLORS['text'],
                    borderwidth=0, padding=(12, 8), font=(FONT, 10))
    style.map('TButton', background=[('active', '#3a3f4b'), ('disabled', COLORS['panel'])],
              foreground=[('disabled', COLORS['muted'])])
    style.configure('Accent.TButton', background=COLORS['accent'], font=(FONT, 11, 'bold'))
    style.map('Accent.TButton', background=[('active', '#2563eb'), ('disabled', COLORS['panel_alt'])])
    style.configure('Danger.TButton', background=COLORS['bad'], font=(FONT, 11, 'bold'))
    style.map('Danger.TButton', background=[('active', '#dc2626')])
    style.configure('TCheckbutton', background=COLORS['panel'], foreground=COLORS['text'])
    style.map('TCheckbutton', background=[('active', COLORS['panel'])])
    style.configure('TCombobox', fieldbackground=COLORS['panel_alt'], background=COLORS['panel_alt'],
                    foreground=COLORS['text'], arrowcolor=COLORS['text'])
    style.map('TCombobox', fieldbackground=[('readonly', COLORS['panel_alt'])],
              foreground=[('readonly', COLORS['text'])])
    root.option_add('*TCombobox*Listbox.background', COLORS['panel_alt'])
    root.option_add('*TCombobox*Listbox.foreground', COLORS['text'])
    style.configure('Horizontal.TProgressbar', background=COLORS['accent'], troughcolor=COLORS['panel_alt'],
                    borderwidth=0)
    style.configure('Treeview', background=COLORS['panel_alt'], fieldbackground=COLORS['panel_alt'],
                    foreground=COLORS['text'], rowheight=int(26 * scale), borderwidth=0)
    style.configure('Treeview.Heading', background=COLORS['panel'], foreground=COLORS['muted'],
                    font=(FONT, 9, 'bold'), borderwidth=0)
    style.map('Treeview', background=[('selected', COLORS['accent'])])


class VideoPanel(tk.Label):
    """Mostra frames BGR do OpenCV redimensionados para caber no painel, mantendo a proporção."""

    def __init__(self, parent, placeholder="", **kwargs):
        super().__init__(parent, bg='black', fg=COLORS['muted'], font=(FONT, 14), text=placeholder, **kwargs)
        self._photo = None

    def show(self, frame_bgr):
        w, h = max(1, self.winfo_width()), max(1, self.winfo_height())
        fh, fw = frame_bgr.shape[:2]
        scale = min(w / fw, h / fh)
        if scale <= 0:
            return
        size = (max(1, int(fw * scale)), max(1, int(fh * scale)))
        rgb = cv.cvtColor(cv.resize(frame_bgr, size, interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.configure(image=self._photo, text="")

    def clear(self, text=""):
        self._photo = None
        self.configure(image="", text=text)


class TaskRunner:
    """
    Executa funções lentas (carregar modelos, avaliar vídeo) numa thread, e entrega
    progresso/resultado na thread do Tkinter (widgets só podem ser tocados nela).
    """

    def __init__(self, root):
        self.root = root
        self.queue = queue.Queue()
        self.root.after(50, self._poll)

    def run(self, fn, on_done=None, on_error=None, on_progress=None):
        """fn(progress) roda em segundo plano; progress(*args) chama on_progress(*args) na UI."""
        def progress(*args):
            if on_progress:
                self.queue.put((on_progress, args))

        def worker():
            try:
                result = fn(progress)
                self.queue.put((on_done, (result,)))
            except Exception as e:  # mostrado ao usuário pelo on_error
                traceback.print_exc()
                self.queue.put((on_error, (e,)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                callback, args = self.queue.get_nowait()
                if callback:
                    callback(*args)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)
