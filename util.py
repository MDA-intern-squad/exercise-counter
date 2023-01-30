import pandas as pd
import numpy as np
import mediapipe as mp
import cv2 as cv
import os
import tqdm
import csv
import math

mp_pose = mp.solutions.pose

class CSVLoader:
    def __init__(self) -> None:
        pass

    def __call__(self, filename):
        data = pd.read_csv(filename)
        return np.array([item.reshape((33, 3)) for item in np.array([item for idx, item in data.iterrows()], dtype=np.float32)])
    
class CSVSaver:
    def __init__(self) -> None:
        pass

    def __call__(self, filename: str, array: np.ndarray):
        csv.writer(open(f'./{filename}.csv', 'w')).writerows(array)

class Bootstrapper:
    def __init__(self, pose) -> None:
        self.pose = pose

    def __call__(self, filename: str) -> np.ndarray:
        images = Bootstrapper.split_video(filename)
        result: list[np.ndarray] = []
        with tqdm.tqdm(total=len(images), position=0, leave=False) as pbar:
            for image in images:
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                landmarks = self.pose.process(image).pose_world_landmarks.landmark
                tmp_arr = []
                for landmark in mp_pose.PoseLandmark:
                    tmp_arr.append(landmarks[landmark].x)
                    tmp_arr.append(landmarks[landmark].y)
                    tmp_arr.append(landmarks[landmark].z)
                result.append(np.array(tmp_arr, dtype=np.float32))
                pbar.update()
        
        return np.array(result, dtype=np.float32) * 100

            
    @staticmethod
    def split_video(filename: str) -> list:
        files = []
        if os.path.isdir(filename):
            for item in os.listdir(filename):
                files.append(cv.imread(item))
        elif os.path.isfile(filename):
            vcap = cv.VideoCapture(filename)
            while True:
                success, image = vcap.read()
                if success:
                    files.append(image)
                else:
                    break
            vcap.release()
        else:
            print("올바르지 않은 파일 명입니다")
        return files


class AngleEmbeder:
    angles = [
        ["left_elbow", [11, 13, 15]],
        ["right_elbow", [12, 14, 16]],
        ["left_knee", [23, 25, 27]],
        ["right_knee", [24, 26, 28]],
        ["left_hip_y", [25, 23, 24]],
        ["right_hip_y", [26, 24, 23]],
        ["left_hip_x", [11, 23, 25]],
        ["right_hip_x", [12, 24, 26]],
        ["left_shoulder_x", [13, 11, 23]],
        ["right_shoulder_x", [14, 12, 24]],
        ["left_shoulder_y", [13, 11, 12]],
        ["right_shoulder_y", [14, 12, 11]],
        ["left_ankle", [25, 27, 31]],
        ["right_ankle", [26, 28, 32]]
    ]
    
    
    def __init__(self) -> None:
        pass

    def __call__(self, landmark):
        return np.array([round(AngleEmbeder.three_angle(*[landmark[item[1][i]] for i in range(3)]), 5) for item in AngleEmbeder.angles], dtype=np.float32)

    @staticmethod
    def three_angle(a, b, c):
        ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
        bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]
        abVec = math.sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2])
        bcVec = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2])
        abNorm = [ab[0] / abVec, ab[1] / abVec, ab[2] / abVec]
        bcNorm = [bc[0] / bcVec, bc[1] / bcVec, bc[2] / bcVec]
        res = abNorm[0] * bcNorm[0] + abNorm[1] * bcNorm[1] + abNorm[2] * bcNorm[2]
        return math.pi - math.acos(res)
    
    @staticmethod
    def get_angle_names():
        return [item[0] for item in AngleEmbeder.angles]

class DistanceEmbeder:
    _landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]
    def __init__(self):
        pass
    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        return np.array([
            self._get_distance(
                self._get_average_by_names(
                    landmarks, 'left_hip', 'right_hip'
                ),
                self._get_average_by_names(
                    landmarks, 'left_shoulder', 'right_shoulder'
                )
            ),
            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_elbow'
            ),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_elbow'
            ),
            self._get_distance_by_names(
                landmarks, 'left_elbow', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_elbow', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_knee'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_knee'
            ),
            self._get_distance_by_names(
                landmarks, 'left_knee', 'left_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'right_knee', 'right_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_ankle'
            ),
            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_elbow', 'right_elbow'
            ),
            self._get_distance_by_names(
                landmarks, 'left_knee', 'right_knee'
            ),
            self._get_distance_by_names(
                landmarks, 'left_wrist', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'left_ankle', 'right_ankle'
            ),
            # 신체 꺾임의 각도.
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])
    def _get_average_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str) -> np.ndarray:
        lmk_from = landmarks[DistanceEmbeder._landmark_names.index(name_from)]
        lmk_to = landmarks[DistanceEmbeder._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) / 2
    def _get_distance_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str) -> np.ndarray:
        lmk_from = landmarks[DistanceEmbeder._landmark_names.index(name_from)]
        lmk_to = landmarks[DistanceEmbeder._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)
    def _get_distance(self, lmk_from: np.ndarray, lmk_to: np.ndarray) -> np.ndarray:
        return lmk_to - lmk_from
    
class KNNFinder:
    def __init__(self, target: dict[str, np.ndarray], embeder: DistanceEmbeder, axes_weights: tuple=(1.0, 1.0, 0.2), top_n_by_max_distance: int=30, top_n_by_mean_distance: int=9):
        tmp_target: list[np.ndarray] = []
        target_dict: dict[str, int] = {}
        
        for k in target: # ['up', 'down']
            tmp_target.extend(target[k])
            target_dict[k] = len(target[k]) # target_dict에는 해당 클래스의 프레임 개수를 넣어준다.

        self._target = tmp_target
        self._dict = target_dict
        self._pose_embedder = embeder
        self._axes_weights = axes_weights
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance

    def __call__(self, pose_landmarks) -> dict[str, int]:
        pose_embedding = self._pose_embedder(np.array(pose_landmarks * np.array([100, 100, 100])))
        flipped_pose_embedding = self._pose_embedder(np.array(pose_landmarks * np.array([-100, 100, 100])))
        
        max_dist_heap = []

        for sample_idx, sample in enumerate(self._target):
            max_dist = min(
                np.max(np.abs(sample - pose_embedding)),
                np.max(np.abs(sample - flipped_pose_embedding))
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._target[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample - pose_embedding)),
                np.mean(np.abs(sample - flipped_pose_embedding))
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        res = { name: 0 for name in self._dict }
        
        for _, idx in mean_dist_heap:
            for name, ran in self._dict.items():
                idx -= ran
                if idx <= 0:
                    res[name] += 1
                    break
        return res

def pose_flatter(landmarks):
    tmp_arr = []
    for landmark in mp_pose.PoseLandmark:
        tmp_arr.append(landmarks[landmark].x)
        tmp_arr.append(landmarks[landmark].y)
        tmp_arr.append(landmarks[landmark].z)
    return np.array(tmp_arr, dtype=np.float32)