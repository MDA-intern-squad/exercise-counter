import cv2
import numpy as np
import sys
import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import mpCustom as ct
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--cam', help='웹캠을 테스트 비디오 대신 이용합니다')
parser.add_argument(
    '--clearn', help='sum the integers (default: find the max)')

dirName = sys.argv[1]
video_path = f'./data/{dirName}/video/test.mp4'
out_video_path = f'./data/{dirName}/video/result.mp4'
pose_samples_folder = f'./data/{dirName}/dist'
class_name = 'up'

video_cap = cv2.VideoCapture(0)
# video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
performance = ct.Performance()
video_n_frames = 1000
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ==================== 초기화 ==================== #
pose_tracker = mp_pose.Pose(model_complexity=0)
pose_embedder = ct.FullBodyPoseEmbedder()
pose_classifier = ct.PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)
pose_classification_filter = ct.EMADictSmoothing(
    window_size=10,
    alpha=0.2)
repetition_counter = ct.RepetitionCounter(
    class_name=class_name,
    enter_threshold=6,
    exit_threshold=4)
pose_classification_visualizer = ct.PoseClassificationVisualizer(
    class_name=class_name,
    plot_x_max=video_n_frames,
    plot_y_max=10)
out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
    *'X264'), video_fps, (video_width, video_height))
frame_idx = 0
output_frame = None
# =======================================================#
with tqdm.tqdm(total=video_n_frames, position=0, leave=False) as pbar:
    while True:
        # Get next frame of the video.
        performance.setStartPoint()
        success, input_frame = video_cap.read()
        if not success:
            break

        # Run pose tracker.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        # Draw pose prediction.
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                       for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (
                33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # Classify the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks)

            # Smooth classification using EMA.
            pose_classification_filtered = pose_classification_filter(
                pose_classification)

            # Count repetitions.
            repetitions_count = repetition_counter(
                pose_classification_filtered)

        else:
            pose_classification = None
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None
            repetitions_count = repetition_counter.n_repeats

        # Draw classification plot and repetition counter.

        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count)

        # Save the output frame.

        convert = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)
        out_video.write(convert)
        cv2.imshow('12345', convert)
        cv2.waitKey(1)

        frame_idx += 1
        # pbar.update()
        performance.printEndPoint()

# 비디오와 mediapipe의 pose모델을 닫는다
out_video.release()
pose_tracker.close()
