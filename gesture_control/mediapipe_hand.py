import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, model_complexity=0, det_conf=0.5, track_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )
        self.drawer = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

    def process(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw(self, frame, hand_landmarks):
        self.drawer.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.styles.get_default_hand_landmarks_style(),
            self.styles.get_default_hand_connections_style()
        )

    def get_handedness_label(self, results, i):
        # Uses MediaPipe’s handedness output (reliable)
        if results.multi_handedness and len(results.multi_handedness) > i:
            return results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'
        return "Unknown"
