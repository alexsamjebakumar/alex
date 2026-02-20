import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
last_action = 0

def allow_action():
    global last_action
    if time.time() - last_action > 1.2:
        last_action = time.time()
        return True
    return False

print("Gesture control started (SAFE MODE)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark
            thumb_up = lm[4].y < lm[3].y
            thumb_down = lm[4].y > lm[3].y
            index = lm[8].y < lm[6].y
            middle = lm[12].y < lm[10].y
            ring = lm[16].y < lm[14].y
            pinky = lm[20].y < lm[18].y

            # ✋ Play
            if index and middle and ring and pinky and allow_action():
                pyautogui.press("space")
                print("PLAY / PAUSE")

            # ☝ Next
            elif index and not middle and not ring and not pinky and allow_action():
                pyautogui.hotkey("ctrl", "right")
                print("NEXT")

            # ✌ Previous
            elif index and middle and not ring and not pinky and allow_action():
                pyautogui.hotkey("ctrl", "left")
                print("PREVIOUS")

            elif thumb_up and not index and not middle and allow_action():
                pyautogui.press("volumeup")
                print("VOLUME UP")

            # 👎 Volume Down
            elif thumb_down and not index and not middle and allow_action():
                pyautogui.press("volumedown")
                print("VOLUME DOWN")

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()