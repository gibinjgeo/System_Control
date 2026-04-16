import time
import cv2

from utils import clamp_bbox, make_square_white, preprocess_for_tm
from actions import move_mouse_to_landmark, press_key, click


# ---------- Mode 1: mouse control (rule-based) ----------
def mouse_control_loop(cfg, cam, tracker):
    """
    Returns:
      - "hand" when user triggers switch to hand ML modes
      - raises SystemExit on exit gesture or ESC
    """
    last_time = time.time() - cfg.gesture_delay_s

    import pyautogui
    screen_w, screen_h = pyautogui.size()

    def count_fingers_rule(lm):
        c = 0

        # Move mouse when index finger is "up"
        if lm.landmark[6].y > lm.landmark[8].y:
            move_mouse_to_landmark(lm.landmark, screen_w, screen_h, lm_index=8)

        # clicks + arrows
        if lm.landmark[8].x > lm.landmark[4].x and lm.landmark[18].y > lm.landmark[20].y:
            c = 2
        if lm.landmark[8].x > lm.landmark[4].x and lm.landmark[18].y > lm.landmark[20].y and lm.landmark[14].y > lm.landmark[16].y:
            c = 3
        if lm.landmark[8].x < lm.landmark[4].x:
            c = 4
        if lm.landmark[8].x < lm.landmark[4].x and lm.landmark[18].y > lm.landmark[20].y:
            c = 5
        if lm.landmark[8].x < lm.landmark[4].x and lm.landmark[18].y > lm.landmark[20].y and lm.landmark[14].y > lm.landmark[16].y:
            c = 6
        if lm.landmark[8].x < lm.landmark[4].x and lm.landmark[18].y > lm.landmark[20].y and lm.landmark[14].y > lm.landmark[16].y and lm.landmark[10].y > lm.landmark[12].y:
            c = 7
        if lm.landmark[6].y > lm.landmark[8].y and lm.landmark[18].y > lm.landmark[20].y and lm.landmark[2].y > lm.landmark[4].y and lm.landmark[8].x < lm.landmark[4].x:
            c = 9
        if lm.landmark[6].y > lm.landmark[8].y and lm.landmark[10].y > lm.landmark[12].y and lm.landmark[14].y > lm.landmark[16].y and lm.landmark[18].y > lm.landmark[20].y and lm.landmark[2].y > lm.landmark[4].y and lm.landmark[8].x < lm.landmark[4].x:
            c = 8

        return c

    def do_action(code):
        if code == 2:
            click("left")
        elif code == 3:
            click("right")
        elif code == 4:
            press_key("up")
        elif code == 5:
            press_key("down")
        elif code == 6:
            press_key("left")
        elif code == 7:
            press_key("right")
        elif code == 8:
            raise SystemExit
        elif code == 9:
            cv2.destroyAllWindows()
            return "hand"
        return None

    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        if cfg.flip_view:
            frame = cv2.flip(frame, 1)

        results = tracker.process(frame)

        if results.multi_hand_landmarks:
            tracker.draw(frame, results.multi_hand_landmarks[0])

            now = time.time()
            if now - last_time >= cfg.gesture_delay_s:
                code = count_fingers_rule(results.multi_hand_landmarks[0])
                out = do_action(code)
                last_time = now

                if out == "hand":
                    return "hand"

        cv2.imshow("Gesture Control - Mouse", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            raise SystemExit


# ---------- Helper: get bounding box from landmarks ----------
def bbox_from_landmarks(hand_landmarks, img_w, img_h, offset):
    xs, ys = [], []
    for lm in hand_landmarks.landmark:
        xs.append(int(lm.x * img_w))
        ys.append(int(lm.y * img_h))

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x = x_min - offset
    y = y_min - offset
    w = (x_max - x_min) + 2 * offset
    h = (y_max - y_min) + 2 * offset
    return x, y, w, h


# ---------- Mode 2/3/4: ML classifier loop ----------
def tm_classifier_loop(
    cfg,
    cam,
    tracker,
    classifier,
    name_to_action: dict,
    window_name: str,
    return_on_label: str | None = None,  # label NAME for SHIFT (e.g. "SHF")
    exit_on_label: str | None = None,    # label NAME for EXIT (e.g. "EXT")
    use_two_hands: bool = False,
    next_token: str | None = None,       # token returned when SHIFT detected
):
    """
    Returns:
      - next_token when return_on_label (SHIFT) is detected
      - raises SystemExit on exit_on_label or ESC
      - otherwise runs name_to_action[label]() when confident
    """
    last_time = time.time() - cfg.ml_delay_s

    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        view = cv2.flip(frame, 1) if cfg.flip_view else frame
        results = tracker.process(view)
        img_h, img_w = view.shape[:2]

        cv2.imshow(window_name, view)

        if cv2.waitKey(1) & 0xFF == 27:
            raise SystemExit

        if not results.multi_hand_landmarks:
            continue

        # ----- Compute bbox (single or combined 2 hands) -----
        if use_two_hands and len(results.multi_hand_landmarks) >= 2:
            all_xmin, all_ymin = 10**9, 10**9
            all_xmax, all_ymax = 0, 0

            for hlm in results.multi_hand_landmarks[:2]:
                tracker.draw(view, hlm)
                x, y, w, h = bbox_from_landmarks(hlm, img_w, img_h, cfg.bbox_offset)
                all_xmin = min(all_xmin, x)
                all_ymin = min(all_ymin, y)
                all_xmax = max(all_xmax, x + w)
                all_ymax = max(all_ymax, y + h)

            x, y = all_xmin, all_ymin
            w, h = all_xmax - all_xmin, all_ymax - all_ymin
        else:
            hlm = results.multi_hand_landmarks[0]
            tracker.draw(view, hlm)
            x, y, w, h = bbox_from_landmarks(hlm, img_w, img_h, cfg.bbox_offset)

        x, y, w, h = clamp_bbox(x, y, w, h, img_w, img_h)

        crop = view[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        # ----- Preprocess for Teachable Machine model -----
        white = make_square_white(crop, cfg.img_size)
        cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.imshow(f"{window_name} - Teachable", white)

        # ----- Predict -----
        x_in = preprocess_for_tm(white)
        idx, label, conf = classifier.predict(x_in)

        # ----- Debounce & confidence gate -----
        now = time.time()
        if conf < cfg.confidence_threshold:
            continue
        if (now - last_time) < cfg.ml_delay_s:
            continue

        # ----- Control gestures (matched by label name, works across all models) -----
        if return_on_label is not None and label == return_on_label:
            cv2.destroyAllWindows()
            last_time = now
            return next_token or "shift"

        if exit_on_label is not None and label == exit_on_label:
            raise SystemExit

        # ----- Normal mapped action -----
        action = name_to_action.get(label)
        if action:
            action()

        last_time = now
