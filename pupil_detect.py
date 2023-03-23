import cv2
import mediapipe as mp
import numpy as np
import pyautogui
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) #얼굴 그물망
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

while True:
    _, frame = cam.read() #첫 번째 변수 무시하고 두 번째 변수가 오른쪽 프레임에 카메라를 호출함
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmarks_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmarks_points:
        landmarks = landmarks_points[0].landmark  #관심을 갖는 얼굴이 하나뿐

        mesh_points = np.array(
            [np.multiply([p.x, p.y], [frame_w, frame_h]).astype(int) for p in output.multi_face_landmarks[0].landmark])

        #(l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_EYE])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_EYE])
        (l_cx, l_cy), a_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        # (r_cx, r_cy), b_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        #cv2.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, center_left, int(a_radius), (255, 0, 255), 1, cv2.LINE_AA)
        # cv2.circle(frame, center_right, int(b_radius), (255, 0, 255), 1, cv2.LINE_AA)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, center_left, int(a_radius), (255, 0, 255), 1, cv2.LINE_AA)
        left_eye = cv2.bitwise_and(rgb_frame, rgb_frame, mask=mask)


        center_left = np.array(([l_cx - 10, l_cy - 10], [l_cx + 10, l_cy + 10]), np.int32)

        min_x = np.min(center_left[:, 0])
        max_x = np.max(center_left[:, 0])
        min_y = np.min(center_left[:, 1])
        max_y = np.max(center_left[:, 1])

        eye = frame[min_y: max_y, min_x: max_x]
        eye = cv2.resize(eye, None, fx=5, fy= 5)
        cv2.imshow("Eye", eye)
        cv2.imshow("mask", left_eye)

        x1 = max_x - min_x
        y1 = max_y - min_y

        #print(center_left)
        # print(pyautogui.position(x1, y1))
        # print(pyautogui.size())
        # pyautogui.FAILSAFE = False
        # pyautogui.moveTo(x1, y1)
    cv2.imshow('Eye Controlled Mouse', frame)
    key = cv2.waitKey(1)
    if key == 27:   # ESC 입력하면 꺼짐
        break