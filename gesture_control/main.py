import time
import cv2

from config import AppConfig
from camera import Camera
from mediapipe_hand import HandTracker
from model_classifier import TMClassifier
from modes import mouse_control_loop, tm_classifier_loop
from actions import press_key
from utils import pick_model


def build_key_actions():
    return {
        0: lambda: press_key("a"),
        1: lambda: press_key("b"),
        3: lambda: press_key("c"),
        4: lambda: press_key("d"),
        6: lambda: press_key("e"),
        8: lambda: press_key("f"),
        11: lambda: press_key("g"),
        12: lambda: press_key("h"),
        13: lambda: press_key("i"),
        14: lambda: press_key("j"),
        15: lambda: press_key("k"),
        16: lambda: press_key("l"),
        17: lambda: press_key("m"),
        18: lambda: press_key("n"),
        19: lambda: press_key("o"),
        20: lambda: press_key("p"),
        21: lambda: press_key("q"),
        22: lambda: press_key("r"),
        23: lambda: press_key("s"),
        26: lambda: press_key("t"),
        27: lambda: press_key("u"),
        28: lambda: press_key("v"),
        29: lambda: press_key("w"),
        30: lambda: press_key("x"),
        31: lambda: press_key("y"),
        32: lambda: press_key("z"),
        2: lambda: press_key("backspace"),
        5: lambda: press_key("capslock"),
        25: lambda: press_key("space"),
        9: lambda: None,
        10: lambda: None,
    }


def run_mouse(cfg, cam, tracker):
    # should return "hand" when user shows “hand mode” gesture
    return mouse_control_loop(cfg, cam, tracker)


def run_hand_gate(cfg, cam, tracker):
    """
    Optional: a small “hand_gesture” gate.
    If you don’t have a separate gate model, just return "asl".
    """
    return "asl"


def run_asl(cfg, cam, tracker, asl_model, key_actions):
    return tm_classifier_loop(
        cfg, cam, tracker, asl_model, key_actions,
        window_name="ASL Mode",
        return_on_label=24,       # SHIFT -> next stage
        mouse_on_label=33,        # BACK TO MOUSE (choose if you have it)
        exit_on_label=7,          # EXIT (if your model has it)
        use_two_hands=False,
        next_token="isl",         # what SHIFT means in this stage
    )


def run_isl(cfg, cam, tracker, isl_model, key_actions):
    return tm_classifier_loop(
        cfg, cam, tracker, isl_model, key_actions,
        window_name="ISL Mode",
        return_on_label=24,
        mouse_on_label=33,
        exit_on_label=7,
        use_two_hands=True,
        next_token="csl",
    )


def run_csl(cfg, cam, tracker, csl_model, key_actions):
    return tm_classifier_loop(
        cfg, cam, tracker, csl_model, key_actions,
        window_name="CSL Mode",
        return_on_label=24,
        mouse_on_label=33,
        exit_on_label=7,
        use_two_hands=False,
        next_token="mouse",       # SHIFT from CSL returns to mouse
    )


def start():
    cfg = AppConfig()
    assets = cfg.assets_dir

    cam = Camera(cfg.camera_index, cfg.frame_width, cfg.frame_height)
    tracker = HandTracker()

    csl = TMClassifier(pick_model(assets, "keras_model"),  assets / "labels.txt")
    asl = TMClassifier(pick_model(assets, "keras_model1"), assets / "labels1.txt")
    isl = TMClassifier(pick_model(assets, "keras_model2"), assets / "labels2.txt")

    key_actions = build_key_actions()

    try:
        while True:
            capt = run_mouse(cfg, cam, tracker)
            print("mouse ->", capt)

            if capt == "hand":
                capt = run_hand_gate(cfg, cam, tracker)
                print("hand_gate ->", capt)

                if capt == "asl":
                    time.sleep(0.3)
                    capt = run_asl(cfg, cam, tracker, asl, key_actions)
                    print("asl ->", capt)

                if capt == "isl":
                    time.sleep(0.3)
                    capt = run_isl(cfg, cam, tracker, isl, key_actions)
                    print("isl ->", capt)

                if capt == "csl":
                    time.sleep(0.3)
                    capt = run_csl(cfg, cam, tracker, csl, key_actions)
                    print("csl ->", capt)

                # if CSL returns mouse, loop repeats naturally
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
