import platform
import cv2


class Camera:
    def __init__(self, index: int, width: int, height: int):
        # Use V4L2 backend on Linux for lower latency than the default FFMPEG path
        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Buffer of 1 means we always read the most recent frame (no stale frames)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
