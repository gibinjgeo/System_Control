import time
import cv2

from config import AppConfig
from camera import Camera
from mediapipe_hand import HandTracker
from model_classifier import TMClassifier
from modes import mouse_control_loop, tm_classifier_loop
from actions import press_key, hotkey
from utils import pick_model


def build_name_actions():
    # Map gesture label NAMES to actions (model-agnostic).
    # Using names instead of indices is correct for all models:
    # CSL/ASL/ISL each assign different indices to the same label name.

    # A-Z letters: each label name maps to its lowercase key press
    actions = {ch: (lambda c=ch.lower(): press_key(c)) for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    actions.update({
        "BAK": lambda: press_key("backspace"),
        "CAP": lambda: press_key("capslock"),
        "SPA": lambda: press_key("space"),
        # Custom shortcut gestures (Ubuntu GNOME / X11)
        "FE1": lambda: hotkey("ctrl", "c"),   # Copy
        "FE2": lambda: hotkey("ctrl", "v"),   # Paste
    })
    return actions


def run_mouse(cfg, cam, tracker):
    return mouse_control_loop(cfg, cam, tracker)


def run_hand_gate(cfg, cam, tracker):
    # Entry point after mouse mode - goes directly to ASL.
    return "asl"


def run_asl(cfg, cam, tracker, asl_model, name_actions):
    return tm_classifier_loop(
        cfg, cam, tracker, asl_model, name_actions,
        window_name="ASL Mode",
        return_on_label="SHF",   # SHIFT gesture -> switch to ISL
        exit_on_label="EXT",     # EXIT gesture -> quit program
        use_two_hands=False,
        next_token="isl",
    )


def run_isl(cfg, cam, tracker, isl_model, name_actions):
    return tm_classifier_loop(
        cfg, cam, tracker, isl_model, name_actions,
        window_name="ISL Mode",
        return_on_label="SHF",   # SHIFT gesture -> switch to Custom mode
        exit_on_label="EXT",
        use_two_hands=True,
        next_token="csl",
    )


def run_csl(cfg, cam, tracker, csl_model, name_actions):
    return tm_classifier_loop(
        cfg, cam, tracker, csl_model, name_actions,
        window_name="Custom Mode",
        return_on_label="SHF",   # SHIFT gesture -> back to Mouse mode
        exit_on_label="EXT",
        use_two_hands=False,
        next_token="mouse",
    )


def start():
    cfg = AppConfig()
    assets = cfg.assets_dir

    cam = Camera(cfg.camera_index, cfg.frame_width, cfg.frame_height)
    tracker = HandTracker()

    csl = TMClassifier(pick_model(assets, "keras_model"),  assets / "labels.txt")
    asl = TMClassifier(pick_model(assets, "keras_model1"), assets / "labels1.txt")
    isl = TMClassifier(pick_model(assets, "keras_model2"), assets / "labels2.txt")

    name_actions = build_name_actions()

    try:
        while True:
            capt = run_mouse(cfg, cam, tracker)
            print("mouse ->", capt)

            if capt == "hand":
                capt = run_hand_gate(cfg, cam, tracker)
                print("hand_gate ->", capt)

                if capt == "asl":
                    time.sleep(0.3)
                    capt = run_asl(cfg, cam, tracker, asl, name_actions)
                    print("asl ->", capt)

                if capt == "isl":
                    time.sleep(0.3)
                    capt = run_isl(cfg, cam, tracker, isl, name_actions)
                    print("isl ->", capt)

                if capt == "csl":
                    time.sleep(0.3)
                    capt = run_csl(cfg, cam, tracker, csl, name_actions)
                    print("csl ->", capt)

                if capt == "mouse":
                    cv2.destroyAllWindows()
                    continue

            elif capt == "exit":
                raise SystemExit

    except SystemExit:
        print("Exiting program.")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start()
