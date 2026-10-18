import threading

import cv2 as cv


class VideoStream:
    """
    Lê a câmera numa thread separada e guarda sempre o frame mais recente.
    frame_id aumenta a cada leitura, para quem consome saber se o frame é novo.
    """
    def __init__(self, src=0):
        self.stream = cv.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.fps = self.stream.get(cv.CAP_PROP_FPS) or 30.0
        self.frame_id = 0
        self.stopped = False

    def start(self):
        self.T = threading.Thread(target=self.update, args=(), daemon=True)
        self.T.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            self.frame_id += 1

    def read(self):
        return self.frame

    def read_with_id(self):
        return self.frame, self.frame_id

    def is_opened(self):
        return self.stream.isOpened() and self.frame is not None

    def stop(self):
        self.stopped = True
        # Espera a thread de leitura sair antes de liberar a câmera (evita ler de um stream fechado)
        if hasattr(self, 'T'):
            self.T.join(timeout=1.0)
        self.stream.release()
