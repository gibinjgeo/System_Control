import cv2
from keras.models import load_model
import numpy as np
import math
import mediapipe as mp
import pyautogui
import sys
import time

offset = 20
imgsize = 224


def mouse_control():
    def count_fingers(lst):
        c = 0
        if lst.landmark[6].y > lst.landmark[8].y:
            image_width, image_height = pyautogui.size()
            image_height = image_height + 250
            image_width = image_width + 250
            relative_x, relative_y = lst.landmark[8].x, lst.landmark[8].y
            # Multiply by the image dimensions to obtain the denormalized coordinates
            x = int(relative_x * image_width)
            y = int(relative_y * image_height)
            print(x, y)
            pyautogui.moveTo(x, y)
        if lst.landmark[8].x > lst.landmark[4].x and lst.landmark[18].y > lst.landmark[20].y:
            c = 2
        if lst.landmark[8].x > lst.landmark[4].x and lst.landmark[18].y > lst.landmark[20].y and lst.landmark[14].y > \
                lst.landmark[16].y:
            c = 3
        if lst.landmark[8].x < lst.landmark[4].x:
            c = 4
        if lst.landmark[8].x < lst.landmark[4].x and lst.landmark[18].y > lst.landmark[20].y:
            c = 5
        if lst.landmark[8].x < lst.landmark[4].x and lst.landmark[18].y > lst.landmark[20].y and lst.landmark[14].y > \
                lst.landmark[16].y:
            c = 6
        if lst.landmark[8].x < lst.landmark[4].x and lst.landmark[18].y > lst.landmark[20].y and lst.landmark[14].y > \
                lst.landmark[16].y and lst.landmark[10].y > lst.landmark[12].y:
            c = 7
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[18].y > lst.landmark[20].y and lst.landmark[2].y > \
                lst.landmark[4].y and lst.landmark[8].x < lst.landmark[4].x:
            c = 9
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y and lst.landmark[14].y > \
                lst.landmark[16].y and lst.landmark[18].y > lst.landmark[20].y and lst.landmark[2].y > lst.landmark[
            4].y and lst.landmark[8].x < lst.landmark[4].x:
            c = 8
        return c

    def keybord(ctt):
        # Code to be executed in another thread
        if ctt == 2:
            pyautogui.click(button='left')
            print("m-left")
        elif ctt == 3:
            pyautogui.click(button='right')
            print("m-right")
        elif ctt == 4:
            pyautogui.press("up")
            print("k-up")
        elif ctt == 5:
            pyautogui.press("down")
            print("k-down")
        elif ctt == 6:
            pyautogui.press("left")
            print("k-left")
        elif ctt == 7:
            pyautogui.press("right")
            print("k-right")
        elif ctt == 8:
            raise SystemExit
        elif ctt == 9:
            cv2.destroyAllWindows()
            return "hand"

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    # Set the minimum time (in seconds) to wait before recognizing another hand gesture
    gesture_delay = 1
    last_gesture_time = time.time() - gesture_delay

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Wait for the thread to finish)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    # Check if enough time has passed since the last gesture
                    current_time = time.time()
                    ctt = count_fingers(results.multi_hand_landmarks[0])
                    if current_time - last_gesture_time > gesture_delay:
                        print(ctt)
                        shift = keybord(ctt)
                        if shift == "hand":
                            return "hand"
                        last_gesture_time = current_time
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', image)
            # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


def hand_gesture():
    def count_fingers(lst):
        c = 0
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[8].x > lst.landmark[4].x:
            c = 1
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y and lst.landmark[8].x > \
                lst.landmark[4].x:
            c = 2
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y and lst.landmark[14].y > \
                lst.landmark[16].y and lst.landmark[8].x > lst.landmark[4].x:
            c = 3
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y and lst.landmark[14].y > \
                lst.landmark[16].y and lst.landmark[18].y > lst.landmark[20].y and lst.landmark[8].x > lst.landmark[
            4].x:
            c = 4
        if lst.landmark[8].x < lst.landmark[4].x:
            c = 6
        if lst.landmark[8].x < lst.landmark[4].x and lst.landmark[6].y > lst.landmark[8].y:
            c = 7
        if lst.landmark[8].x < lst.landmark[4].x and lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > \
                lst.landmark[12].y:
            c = 8
        if lst.landmark[8].x < lst.landmark[4].x and lst.landmark[6].y > lst.landmark[8].y and lst.landmark[18].y > \
                lst.landmark[20].y:
            c = 10
        if lst.landmark[8].x < lst.landmark[4].x and lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > \
                lst.landmark[12].y and lst.landmark[18].y > lst.landmark[20].y:
            c = 9
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y and lst.landmark[14].y > \
                lst.landmark[16].y and lst.landmark[18].y > lst.landmark[20].y and lst.landmark[2].y > lst.landmark[
            4].y and lst.landmark[8].x < lst.landmark[4].x:
            c = 5
        return c

    def left_rec(lst):
        c = 0
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[8].x < lst.landmark[4].x:
            c = 1
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y and lst.landmark[8].x < \
                lst.landmark[4].x:
            c = 2
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y and lst.landmark[14].y > \
                lst.landmark[16].y and lst.landmark[8].x < lst.landmark[4].x:
            c = 3
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y and lst.landmark[14].y > \
                lst.landmark[16].y and lst.landmark[18].y > lst.landmark[20].y and lst.landmark[8].x < lst.landmark[
            4].x:
            c = 4
        if lst.landmark[2].y > lst.landmark[4].y and lst.landmark[8].x > lst.landmark[4].x:
            c = 6
        if lst.landmark[2].y > lst.landmark[4].y and lst.landmark[8].x > lst.landmark[4].x and lst.landmark[6].y > \
                lst.landmark[8].y:
            c = 7
        if lst.landmark[2].y > lst.landmark[4].y and lst.landmark[8].x > lst.landmark[4].x and lst.landmark[6].y > \
                lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y:
            c = 8
        if lst.landmark[2].y > lst.landmark[4].y and lst.landmark[8].x > lst.landmark[4].x and lst.landmark[6].y > \
                lst.landmark[8].y and lst.landmark[18].y > lst.landmark[20].y:
            c = 10
        if lst.landmark[2].y > lst.landmark[4].y and lst.landmark[8].x > lst.landmark[4].x and lst.landmark[6].y > \
                lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y and lst.landmark[18].y > lst.landmark[
            20].y:
            c = 9
        if lst.landmark[6].y > lst.landmark[8].y and lst.landmark[10].y > lst.landmark[12].y and lst.landmark[14].y > \
                lst.landmark[16].y and lst.landmark[18].y > lst.landmark[20].y and lst.landmark[2].y > lst.landmark[
            4].y and lst.landmark[8].x > lst.landmark[4].x:
            c = 5
        return c

    def keybord(ctt):
        if ctt == 1:
            pyautogui.hotkey('ctrl', 'c')
            print("index finger only-one")
        elif ctt == 2:
            pyautogui.hotkey('ctrl', 'v')
            print("middle finger-two")
        elif ctt == 3:
            pyautogui.press('f11')
            print("3 rd finger raised")
        elif ctt == 4:
            pyautogui.hotkey('win', 'tab')
            print("pingy finger-four")
        elif ctt == 5:
            pyautogui.hotkey('alt', 'tab')
            print("index finger-five")
        elif ctt == 6:
            pyautogui.hotkey('win', 'q')
            print("right thumb-six")
        elif ctt == 7:
            pyautogui.hotkey('ctrl', 'tab')
            print("thum+index-seven")
        elif ctt == 8:
            pyautogui.hotkey('ctrl', 'a')
            print("thumb+index+middle-eight")
        elif ctt == 9:
            pyautogui.hotkey('win', 'e')
            print("open palm with ring finger closed-ten")
        elif ctt == 10:
            cv2.destroyAllWindows()
            print("ilu-nine")
            return "asl"

    def keybordl(ctt):
        if ctt == 1:
            pyautogui.press('1')
            print("index finger only-one")
        elif ctt == 2:
            pyautogui.press('2')
            print("middle finger-two")
        elif ctt == 3:
            pyautogui.press('3')
            print("3 rd finger raised")
        elif ctt == 4:
            pyautogui.press('4')
            print("pingy finger-four")
        elif ctt == 5:
            pyautogui.press('5')
        elif ctt == 6:
            pyautogui.press('6')
            print("right thumb-six")
        elif ctt == 7:
            pyautogui.press('7')
            print("thum+index-seven")
        elif ctt == 8:
            pyautogui.press('8')
            print("thumb+index+middle-eight")
        elif ctt == 9:
            pyautogui.press('9')
            print("ilu-nine")
        elif ctt == 10:
            pyautogui.press('0')
            print("open palm with ring finger closed-ten")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    # Set the minimum time (in seconds) to wait before recognizing another hand gesture
    gesture_delay = 1
    last_gesture_time = time.time() - gesture_delay

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Wait for the thread to finish)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    # Check if enough time has passed since the last gesture
                    current_time = time.time()
                    if current_time - last_gesture_time > gesture_delay:
                        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        # Check if the hand is left or right
                        if landmarks[mp_hands.HandLandmark.WRIST] < landmarks[mp_hands.HandLandmark.THUMB_CMC]:
                            print("Right")
                            ctt = count_fingers(results.multi_hand_landmarks[0])
                            print(ctt)
                            shift = keybord(ctt)
                            if shift == "asl":
                                return "asl"
                            last_gesture_time = current_time
                        else:
                            print("Left")
                            ctt = left_rec(results.multi_hand_landmarks[0])
                            print(ctt)
                            keybordl(ctt)
                            last_gesture_time = current_time
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

def asl_again():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model1.h5", compile=False)

    # Load the labels
    class_names = open("labels1.txt", "r").readlines()
    # Set the minimum time (in seconds) to wait before recognizing another hand gesture
    gesture_delay = 1.5
    last_gesture_time = time.time() - gesture_delay
    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

                # Calculate bounding box coordinates
                x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Draw bounding box
                cv2.rectangle(img, (x_min - offset, y_min - offset), (x_max + offset, y_max + offset), (255, 0, 255), 2)
                x = x_min - offset
                y = y_min - offset
                w = (x_max + offset) - (x_min - offset)
                h = (y_max + offset) - (y_min - offset)
                print(x, y, h, w)
                imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                imgCrop = img[y:y + h, x:x + w]
                height, width, _ = img.shape
                # Calculate the maximum values of x and y
                max_x = width - 1
                max_y = height - 1
                print(max_x, max_y)
                if x > 0 and y > 0 and (y_max + offset < max_y) and (x_max + offset < max_x):
                    ar = h / w  # ar = aspect ratio
                    if ar > 1:
                        const = imgsize / h
                        wCal = math.ceil(const * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        wGap = math.ceil((imgsize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize

                    else:
                        const = imgsize / w
                        hCal = math.ceil(const * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    # cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("Teachable", imgWhite)

            # Make the image a numpy array and reshape it to the models input shape.
            image = np.asarray(imgWhite, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalize the image array
            image = (image / 127.5) - 1

            # Predicts the model
            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
            conf = round(confidence_score, 2) * 100
            newio = int(class_name[0:2])
            print(newio, conf)
            current_time = time.time()
            if current_time - last_gesture_time > gesture_delay:
                # Recognize the hand gesture
                if conf > 95:
                    if newio == 0:
                        pyautogui.press('a')
                    if newio == 1:
                        pyautogui.press('b')
                    if newio == 3:
                        pyautogui.press('c')
                    if newio == 4:
                        pyautogui.press('d')
                    if newio == 6:
                        pyautogui.press('e')
                    if newio == 8:
                        pyautogui.press('f')
                    if newio == 11:
                        pyautogui.press('g')
                    if newio == 12:
                        pyautogui.press('h')
                    if newio == 13:
                        pyautogui.press('i')
                    if newio == 14:
                        pyautogui.press('j')
                    if newio == 15:
                        pyautogui.press('k')
                    if newio == 16:
                        pyautogui.press('l')
                    if newio == 17:
                        pyautogui.press('m')
                    if newio == 18:
                        pyautogui.press('n')
                    if newio == 19:
                        pyautogui.press('o')
                    if newio == 20:
                        pyautogui.press('p')
                    if newio == 21:
                        pyautogui.press('q')
                    if newio == 22:
                        pyautogui.press('r')
                    if newio == 23:
                        pyautogui.press('s')
                    if newio == 26:
                        pyautogui.press('t')
                    if newio == 27:
                        pyautogui.press('u')
                    if newio == 28:
                        pyautogui.press('v')
                    if newio == 29:
                        pyautogui.press('w')
                    if newio == 30:
                        pyautogui.press('x')
                    if newio == 31:
                        pyautogui.press('y')
                    if newio == 32:
                        pyautogui.press('z')
                    if newio == 2:
                        pyautogui.press('backspace')
                    if newio == 5:
                        pyautogui.press('capslock')
                    if newio == 7:
                        raise SystemExit
                    if newio == 24:
                        cv2.destroyAllWindows()
                        return "isl"
                    if newio == 25:
                        pyautogui.press('space')
                    if newio == 9:
                        print("fe1")
                    if newio == 10:
                        print("fe2")
                    last_gesture_time = current_time

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("x"):
            # cap.release()
            cv2.destroyAllWindows()


def asl_new():
    try:
        sh = asl_again()
        if sh == "isl":
            return "isl"
    except SystemExit:
        sys.exit()
    except:
        print("Some unknown error occured restarting the module")
        # time.sleep(0.30)
        asl_new()

def isl_again():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model2.h5", compile=False)

    # Load the labels
    class_names = open("labels2.txt", "r").readlines()
    # Set the minimum time (in seconds) to wait before recognizing another hand gesture
    gesture_delay = 1.5
    last_gesture_time = time.time() - gesture_delay
    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        hand_bboxes = []  # List to store individual hand bounding boxes

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate bounding box coordinates for each hand
                x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Store the bounding box coordinates in the list
                hand_bboxes.append((x_min, y_min, x_max, y_max))

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

            if len(hand_bboxes) > 0:
                # Calculate the overall bounding box for all hands
                x_min_all = min([bbox[0] for bbox in hand_bboxes])
                y_min_all = min([bbox[1] for bbox in hand_bboxes])
                x_max_all = max([bbox[2] for bbox in hand_bboxes])
                y_max_all = max([bbox[3] for bbox in hand_bboxes])

                # Draw bounding box for both hands combined
                cv2.rectangle(img, (x_min_all - offset, y_min_all - offset), (x_max_all + offset, y_max_all + offset),
                              (255, 0, 255), 2)

                # Crop and resize the combined bounding box
                x = x_min_all - offset
                y = y_min_all - offset
                w = (x_max_all + offset) - (x_min_all - offset)
                h = (y_max_all + offset) - (y_min_all - offset)
                imgCrop = img[y:y + h, x:x + w]
                imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                height, width, _ = img.shape
                max_x = width - 1
                max_y = height - 1

                if x > 0 and y > 0 and (y_max_all + offset < max_y) and (x_max_all + offset < max_x):
                    ar = h / w
                    try:
                        if ar > 1:
                            const = imgsize / h
                            wCal = math.ceil(const * w)
                            imgResize = cv2.resize(imgCrop, (wCal, 255))
                            wGap = math.ceil((imgsize - wCal) / 2)
                            imgWhite[:, wGap:wCal + wGap] = imgResize

                        else:
                            const = imgsize / w
                            hCal = math.ceil(const * h)
                            imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                            hGap = math.ceil((imgsize - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, :] = imgResize

                        cv2.imshow("Teachable", cv2.flip(imgWhite, 1))
                        # Make the image a numpy array and reshape it to the models input shape.
                        image = np.asarray(imgWhite, dtype=np.float32).reshape(1, 224, 224, 3)

                        # Normalize the image array
                        image = (image / 127.5) - 1

                        # Predicts the model
                        prediction = model.predict(image)
                        index = np.argmax(prediction)
                        class_name = class_names[index]
                        confidence_score = prediction[0][index]
                        print("Class:", class_name[2:], end="")
                        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
                        conf = round(confidence_score, 2) * 100
                        newio = int(class_name[0:2])
                        print(newio, conf)
                        current_time = time.time()
                        if current_time - last_gesture_time > gesture_delay:
                            # Recognize the hand gesture
                            if conf > 95:
                                if newio == 0:
                                    pyautogui.press('a')
                                if newio == 1:
                                    pyautogui.press('b')
                                if newio == 3:
                                    pyautogui.press('c')
                                if newio == 4:
                                    pyautogui.press('d')
                                if newio == 6:
                                    pyautogui.press('e')
                                if newio == 8:
                                    pyautogui.press('f')
                                if newio == 11:
                                    pyautogui.press('g')
                                if newio == 12:
                                    pyautogui.press('h')
                                if newio == 13:
                                    pyautogui.press('i')
                                if newio == 14:
                                    pyautogui.press('j')
                                if newio == 15:
                                    pyautogui.press('k')
                                if newio == 16:
                                    pyautogui.press('l')
                                if newio == 17:
                                    pyautogui.press('m')
                                if newio == 18:
                                    pyautogui.press('n')
                                if newio == 19:
                                    pyautogui.press('o')
                                if newio == 20:
                                    pyautogui.press('p')
                                if newio == 21:
                                    pyautogui.press('q')
                                if newio == 22:
                                    pyautogui.press('r')
                                if newio == 23:
                                    pyautogui.press('s')
                                if newio == 26:
                                    pyautogui.press('t')
                                if newio == 27:
                                    pyautogui.press('u')
                                if newio == 28:
                                    pyautogui.press('v')
                                if newio == 29:
                                    pyautogui.press('w')
                                if newio == 30:
                                    pyautogui.press('x')
                                if newio == 31:
                                    pyautogui.press('y')
                                if newio == 32:
                                    pyautogui.press('z')
                                if newio == 2:
                                    pyautogui.press('backspace')
                                if newio == 5:
                                    pyautogui.press('capslock')
                                if newio == 7:
                                    raise SystemExit
                                if newio == 24:
                                    cv2.destroyAllWindows()
                                    return "csl"
                                if newio == 25:
                                    pyautogui.press('space')
                                if newio == 9:
                                    print("fe1")
                                if newio == 10:
                                    print("fe2")
                                last_gesture_time = current_time
                    except ValueError:
                        print("could not broadcast input array from shape")
                        continue
                else:
                    continue

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("x"):
            # cap.release()
            cv2.destroyAllWindows()


def isl_new():
    try:
        sh = isl_again()
        if sh == "csl":
            return "csl"
    except SystemExit:
        sys.exit()
    except:
        print("Some unknown error occured restarting the module")
        # time.sleep(0.30)
        isl_new()

def again():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()
    # Set the minimum time (in seconds) to wait before recognizing another hand gesture
    gesture_delay = 1.5
    last_gesture_time = time.time() - gesture_delay
    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

                # Calculate bounding box coordinates
                x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Draw bounding box
                cv2.rectangle(img, (x_min - offset, y_min - offset), (x_max + offset, y_max + offset), (255, 0, 255), 2)
                x = x_min - offset
                y = y_min - offset
                w = (x_max + offset) - (x_min - offset)
                h = (y_max + offset) - (y_min - offset)
                print(x, y, h, w)
                imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                imgCrop = img[y:y + h, x:x + w]
                height, width, _ = img.shape
                # Calculate the maximum values of x and y
                max_x = width - 1
                max_y = height - 1
                print(max_x, max_y)
                if x > 0 and y > 0 and (y_max + offset < max_y) and (x_max + offset < max_x):
                    ar = h / w  # ar = aspect ratio
                    if ar > 1:
                        const = imgsize / h
                        wCal = math.ceil(const * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        wGap = math.ceil((imgsize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize

                    else:
                        const = imgsize / w
                        hCal = math.ceil(const * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    # cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("Teachable", imgWhite)

            # Make the image a numpy array and reshape it to the models input shape.
            image = np.asarray(imgWhite, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalize the image array
            image = (image / 127.5) - 1

            # Predicts the model
            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
            conf = round(confidence_score, 2) * 100
            newio = int(class_name[0:2])
            print(newio, conf)
            current_time = time.time()
            if current_time - last_gesture_time > gesture_delay:
                # Recognize the hand gesture
                if conf > 95:
                    if newio == 0:
                        pyautogui.press('a')
                    if newio == 1:
                        pyautogui.press('b')
                    if newio == 3:
                        pyautogui.press('c')
                    if newio == 4:
                        pyautogui.press('d')
                    if newio == 6:
                        pyautogui.press('e')
                    if newio == 8:
                        pyautogui.press('f')
                    if newio == 11:
                        pyautogui.press('g')
                    if newio == 12:
                        pyautogui.press('h')
                    if newio == 13:
                        pyautogui.press('i')
                    if newio == 14:
                        pyautogui.press('j')
                    if newio == 15:
                        pyautogui.press('k')
                    if newio == 16:
                        pyautogui.press('l')
                    if newio == 17:
                        pyautogui.press('m')
                    if newio == 18:
                        pyautogui.press('n')
                    if newio == 19:
                        pyautogui.press('o')
                    if newio == 20:
                        pyautogui.press('p')
                    if newio == 21:
                        pyautogui.press('q')
                    if newio == 22:
                        pyautogui.press('r')
                    if newio == 23:
                        pyautogui.press('s')
                    if newio == 26:
                        pyautogui.press('t')
                    if newio == 27:
                        pyautogui.press('u')
                    if newio == 28:
                        pyautogui.press('v')
                    if newio == 29:
                        pyautogui.press('w')
                    if newio == 30:
                        pyautogui.press('x')
                    if newio == 31:
                        pyautogui.press('y')
                    if newio == 32:
                        pyautogui.press('z')
                    if newio == 2:
                        pyautogui.press('backspace')
                    if newio == 5:
                        pyautogui.press('capslock')
                    if newio == 7:
                        raise SystemExit
                    if newio == 24:
                        cv2.destroyAllWindows()
                        return "mouse"
                    if newio == 25:
                        pyautogui.press('space')
                    if newio == 9:
                        print("fe1")
                    if newio == 10:
                        print("fe2")
                    last_gesture_time = current_time

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("x"):
            # cap.release()
            cv2.destroyAllWindows()


def new():
    try:
        sh = again()
        if sh == "voice":
            return "voice"
    except SystemExit:
        sys.exit()
    except:
        print("Some unknown error occured restarting the module")
        # time.sleep(0.30)
        new()


def start():
    try:
        capt = mouse_control()
        print(capt)
        if capt == "hand":
            capt = hand_gesture()
            print(capt)
            if capt == "asl":
                time.sleep(1)
                capt = asl_new()
                print(capt)
                if capt == "isl":
                    time.sleep(1)
                    capt = isl_new()
                    print(capt)
                    if capt == "csl":
                        time.sleep(1)
                        capt = new()
                        print(capt)
                        if capt == "mouse":
                            return start()

    except SystemExit:
        print("Exiting The Program")
        sys.exit()
    except:
        print("either some error occured or system restarted")
        start()


start()