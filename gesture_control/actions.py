import pyautogui

pyautogui.FAILSAFE = True  # move mouse to top-left to stop if needed
pyautogui.PAUSE = 0        # remove default 0.1s delay; debouncing is in modes.py

def move_mouse_to_landmark(landmarks, screen_w, screen_h, lm_index=8):
    # landmark values are normalized [0..1]
    x = int(landmarks[lm_index].x * screen_w)
    y = int(landmarks[lm_index].y * screen_h)
    pyautogui.moveTo(x, y)

def press_key(key: str):
    pyautogui.press(key)

def hotkey(*keys):
    pyautogui.hotkey(*keys)

def click(button="left"):
    pyautogui.click(button=button)
