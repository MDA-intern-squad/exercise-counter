import util
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
import cv2 as cv
import time
mp_pose = mp.solutions.pose

csv_loader = util.CSVLoader()

# [ frames: [ landmarks: [ xyz: int, int, int ], [], []... x33 ], [], [].. ]
up = csv_loader('./data/test/up.csv')
down = csv_loader('./data/test/down.csv')

embeder = util.PoseEmbedderByDistance()

# [ embeddings: [ xyz: float, float, float ], [], [] ... x23 ]
embeded_up = np.array([embeder(i) for i in up], dtype=np.float32)
embeded_down = np.array([embeder(i) for i in down], dtype=np.float32)

# classifier = util.PoseClassifierByKNN(
#     {
#         'up': embeded_up,
#         'down': embeded_down
#     },
#     embeder,
#     top_n_by_max_distance=100, 
#     top_n_by_mean_distance=31
# )

classifier = util.PoseClassifierByML('./model.h5', embeder)

cap = cv.VideoCapture('./data/test/test3.mp4')
pose = mp_pose.Pose(model_complexity=1,
                    static_image_mode=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = pose.process(image)
    pose_landmarks = result.pose_landmarks    
    
    if pose_landmarks is not None:
        pose_world_landmarks = result.pose_world_landmarks.landmark
        pose_world_landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_world_landmarks], dtype=np.float32)
        output_frame = frame.copy()
        mp_drawing.draw_landmarks(
            image=output_frame,
            landmark_list=pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS
        )
        t = time.time()
        classification = classifier(pose_world_landmarks) # 분류가 완료된 dict
        print((time.time() - t) * 1000)
        currentState, maxValue = '', -float('inf') # 현재 상태(up, down)를 저장할 변수, 그 상태의 값
        for key, value in classification.items():
            if value > maxValue:
                currentState = key
                maxValue = value
        cv.putText(output_frame, currentState, (50, 350), cv.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 10, cv.LINE_AA)
        cv.imshow('ui', output_frame)
    key = cv.waitKey(20)
