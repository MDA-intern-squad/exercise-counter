from asyncio import sleep
import cv2
import numpy as np
import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import mpCustom as ct
import argparse
import threading
import time
from collections import deque
from queue import Queue

parser = argparse.ArgumentParser()
parser.add_argument('dirName')
parser.add_argument('--cam', '-c', help='웹캠을 테스트 비디오 대신 이용합니다',
                    action='store_true', default=False)
parser.add_argument('--save', '-s', help='비디오를 저장합니다',
                    action='store_true', default=False)
parser.add_argument('--ui', '-u', help='UI를 사용합니다',
                    action='store_true', default=False)
parser.add_argument('--performance', '-pf', help='퍼포먼스를 출력합니다',
                    action='store_true', default=False)
parser.add_argument('--progress', '-pg', help='프로그래스바를 출력합니다',
                    action='store_true', default=False)
args = parser.parse_args()

dirName = args.dirName

video_path = f'./data/{dirName}/video/test.mp4'
out_video_path = f'./data/{dirName}/video/result.mp4'
pose_samples_folder = f'./data/{dirName}/dist'
class_name = 'up'

video_cap = cv2.VideoCapture(0 if args.cam else video_path)
video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # 가로
video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # 세로
video_cap.set(cv2.CAP_PROP_FPS, 60)
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ==================== 초기화 ==================== #
performance = ct.Performance()
pose_tracker = mp_pose.Pose(model_complexity=1)
pose_embedder = ct.FullBodyPoseEmbedder()
pose_classifier = ct.PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10
)
pose_classification_filter = ct.EMADictSmoothing(
    window_size=10,
    alpha=0.2
)
repetition_counter = ct.RepetitionCounter(
    class_name=class_name,
    enter_threshold=6,
    exit_threshold=4
)
pose_classification_visualizer = ct.PoseClassificationVisualizer(
    class_name=class_name,
    plot_x_max=video_n_frames,
    plot_y_max=10
)
out_video = cv2.VideoWriter(
    out_video_path, cv2.VideoWriter_fourcc(*'FMP4'), video_fps, (video_width, video_height)
)

frame_idx = 0
output_frame = None
input_frame_stack = deque([])
output_frame_queue = Queue()
# =======================================================#


def IthreadFunction():
    while True:
        success, frame = video_cap.read()
        if success:
            input_frame_stack.clear() if args.cam else None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = pose_tracker.process(image=frame)
            input_frame_stack.append([frame, landmarks])


IThread = threading.Thread(target=IthreadFunction)
IThread.daemon = True
IThread.start()


while not input_frame_stack:
    pass

def logicThreadFunction():
    global frame_idx
    with tqdm.tqdm(total=video_n_frames, position=0, leave=False) as pbar:
        while True:
            # 영상의 다음 프레임을 가져온다.
            if input_frame_stack:
                input_frame, result = input_frame_stack.popleft()
            else:
                if frame_idx == video_n_frames:
                    break
                else:
                    continue
            pose_landmarks = result.pose_landmarks


            if pose_landmarks is not None:
                # < 1ms | < 1ms
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS
                )
                # 랜드마크를 얻는다.
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in pose_landmarks.landmark], dtype=np.float32)
                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                # 현재 프레임의 포즈를 분류한다.
                pose_classification = pose_classifier(pose_landmarks)

                # EMA를 사용하여 포즈의 분류를 매끄럽게 해 준다.
                pose_classification_filtered = pose_classification_filter(pose_classification)

                # 반복 횟수를 카운트한다.
                repetitions_count = repetition_counter(pose_classification_filtered)

            else:
                pose_classification = None
                pose_classification_filtered = pose_classification_filter(dict())
                pose_classification_filtered = None
                repetitions_count = repetition_counter.n_repeats

            # 6ms | 6ms
            # 분류 그래프와 반복 횟수 카운터를 표시한다.

            # output_frame = pose_classification_visualizer(
            #     frame=output_frame,
            #     pose_classification=pose_classification,
            #     pose_classification_filtered=pose_classification_filtered,
            #     repetitions_count=repetitions_count)
            
            cv2.putText(output_frame, str(repetitions_count), (50, 350), cv2.FONT_HERSHEY_PLAIN, 30, (0, 0, 255), 10, cv2.LINE_AA)

            if args.save or args.ui:
                convert = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)
                output_frame_queue.put(convert)
            if args.progress:
                pbar.update()

            frame_idx += 1

    # 비디오와 mediapipe의 pose모델을 닫는다.
    out_video.release()
    pose_tracker.close()


LogicThread = threading.Thread(target=logicThreadFunction)
LogicThread.daemon = True
LogicThread.start()

while True:
    if output_frame_queue:
        convert = output_frame_queue.get()
        performance.setStartPoint()
        if args.save:
            out_video.write(convert)
        if args.ui:
            cv2.imshow('12345', convert)
            cv2.waitKey(1)
        if args.performance:
            performance.printEndPoint()

cv2.destroyAllWindows()