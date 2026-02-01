import cv2

class Camera:
    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index)
        # Try to set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
