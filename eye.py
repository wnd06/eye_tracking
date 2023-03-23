import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import keyboard
import time

pyautogui.FAILSAFE = False

pyautogui.PAUSE = 0.055 # 마우스 이동 속도 제어 (흔들림 제어)
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) # 얼굴 그물망
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
step = 30 # amount to move mouse by

# 화면 중앙 좌표 계산
initial_mouse_pos = pyautogui.position()

# 초기 마우스 위치를 화면 중앙으로 이동

pyautogui.moveTo(screen_w // 2, screen_h // 2)



while True:
    _, frame = cam.read() # 첫 번째 변수 무시하고 두 번째 변수가 오른쪽 프레임에 카메라를 호출함
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmarks_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmarks_points:
        landmarks = landmarks_points[0].landmark  # 관심을 갖는 얼굴이 하나뿐
        for id, landmark in enumerate(landmarks[474 : 478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0)) # 얼굴 랜드마크 감지
            if id == 1:
                move_x = x - prev_x
                move_y = y - prev_y
                pyautogui.move(move_x * 10, move_y * 10, duration = 0.1) #duration은 마우스 이동에 걸리는 시간(초)을 나타내는 옵션으로, 이 값을 작게 설정할수록 더 빠르게 이동하고, 크게 설정할수록 더 느리게 이동한다.
                prev_x, prev_y = x, y
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))  # 얼굴 랜드마크 감지
        if((left[0].y - left[1].y) < 0.003):
            pyautogui.click()
            pyautogui.sleep(1)

    # Check for arrow key presses
    key = cv2.waitKey(1)
    if key == 27:   # ESC 입력하면 꺼짐
        break
    elif keyboard.is_pressed('up'):  # Up arrow key
        pyautogui.move(0, -step)
    elif keyboard.is_pressed('down'):  # Down arrow key
        pyautogui.move(0, step)
    elif keyboard.is_pressed('left'):  # Left arrow key
        pyautogui.move(-step, 0)
    elif keyboard.is_pressed('right'):  # Right arrow key
        pyautogui.move(step, 0)

    cv2.imshow('Eye Controlled Mouse', frame)

cv2.destroyAllWindows()
cam.release()
