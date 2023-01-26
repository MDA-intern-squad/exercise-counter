import csv
from matplotlib import pyplot as plt
import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import requests
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import time


# Commons
def show_image(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    # plt.show()

# Pose Embedding
class FullBodyPoseEmbedder(object):
    """3D 포즈 랜드마크를 3D 포즈 임베딩으로 변환합니다."""

    def __init__(self, torso_size_multiplier: float=2.5):
        # 몸통에 적용할 multiplier를 곱하여 최소 신체 크기를 얻을 수 있다.
        self._torso_size_multiplier = torso_size_multiplier

        # 예측에 사용되는 랜드마크들의 이름
        self._landmark_names = [
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

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        """포즈의 랜드마크를 표준화 하고 Embedding으로 변환합니다.

        Args:
            landmarks - (N, 3)과 같은 형상의 3D 랜드마크가 있는 Numpy 배열을 인자로 받습니다.

        Result:
            (M, 3)과 같은 형상의 Pose Embedding이 있는 Numpy 배열을 반환합니다.
            여기서 'M'은 `_get_pose_distance_embedding` 에서 정의된 Pairwisde Distances의 개수입니다.
        """
        assert landmarks.shape[0] == len(self._landmark_names), f'예외되지 않은 랜드마크 번호: {landmarks.shape[0]}'

        # 포즈의 랜드마크 얻기
        landmarks = np.copy(landmarks)

        # 랜드마크 표준화
        landmarks = self._normalize_pose_landmarks(landmarks)

        # 임베딩 얻기
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """랜드마크의 translation 및 scale을 표준화합니다."""
        landmarks = np.copy(landmarks)

        # translation 표준화.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # scale 표준화
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # 굳이 100을 곱할 필요는 없으나, 디버깅하기 더 쉬워진다. ( 배열 내 모든 원소에 100을 곱함. )
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks: np.ndarray) -> np.ndarray:
        """엉덩이 사이의 점으로 포즈의 중심을 계산합니다."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) / 2
        return center

    def _get_pose_size(self, landmarks: np.ndarray, torso_size_multiplier: float) -> np.float64:
        """포즈의 크기를 계산합니다.

        이 값은 다음 두 값을 비교하여 더 큰 값을 반환합니다.
            * 몸통 크기(torso_size)에 "torso_size_multiplier" 를 곱한 값
            * 포즈 중심에서 모든 포즈 랜드마크까지의 최대 거리
        """
        # 이 접근법은 포즈의 크기를 계산하기 위해 2D 랜드마크만 사용함
        landmarks = landmarks[:, :2]

        # 엉덩이의 중심
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) / 2

        # 어깨의 중심
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) / 2

        # 몸통 크기는 최소 신체 크기와 같다.
        torso_size = np.linalg.norm(shoulders - hips)

        # 포즈의 중심에 대한 최대 거리
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks: np.ndarray) -> np.ndarray:
        """포즈 랜드마크를 3D 임베딩으로 변환합니다.

        포즈 임베딩을 형성하기 위해 여러 쌍의 3D 거리를 사용합니다. 모든 거리에는
        부호가 있는 X 및 Y가 포함됩니다. 서로 다른 포즈 클래스를 다루기 위해
        서로 다른 유형의 쌍을 가지고 있습니다. 얼마든지 제거하거나 새로 추가할 수 있습니다.

        Args:
            landmarks - (N, 3)과 같은 형상의 3D 랜드마크가 있는 Numpy 배열을 인자로 받습니다.

        Result:
            (M, 3)과 같은 형상의 Pose Embedding이 있는 Numpy 배열을 반환합니다.
            여기서 'M'은 Pairwisde Distances의 개수입니다.
        """
        # 0 [hip center - shoulder center], 몸통의 세로길이
        # 1 [left_shoulder - left_elbow], 왼쪽 상완
        # 2 [right_shoulder - right_elbow], 오른쪽 상완
        # 3 [left_elbow - left_wrist], 왼쪽 전완
        # 4 [right_elbow - right_wrist], 오른쪽 전완
        # 5 [left_hip - left_knee], 왼쪽 허벅지
        # 6 [right_hip - right_knee], 오른쪽 허벅지
        # 7 [left_knee - left_ankle], 왼쪽 종아리
        # 8 [right_knee - right_ankle], 오른쪽 종아리
        # 9 [left_shoulder - left_wrist], 왼쪽 팔
        # 10 [right_shoulder - right_wrist], 오른쪽 팔
        # 11 [left_hip - left_ankle], 왼쪽 다리
        # 12 [right_hip - right_ankle], 오른쪽 다리
        # 13 [left_hip - left_wrist], 왼쪽 엉덩이 - 손목
        # 14 [right hip - right_wrist], 오른쪽 엉덩이 - 손목
        # 15 [left_shoulder - left_ankle], 왼쪽 어깨 - 발목
        # 16 [right_shoulder - right_ankle], 오른쪽 어깨 - 발목
        # 17 [left_hip - left_wrist], 왼쪽 엉덩이 - 손목
        # 18 [right_hip - right_wrist], 오른쪽 엉덩이 - 손목
        # 19 [left_elbow - right_elbow], 팔꿈치 거리
        # 20 [left_knee - right_knee], 무릎 거리
        # 21 [left_wrist - right_wrist], 손목 거리
        # 22 [left_ankle - right_ankle], 발목 거리

        embedding = np.array([

            # One joint.


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


            # Two joints.


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


            # Four joints.


            self._get_distance_by_names(
                landmarks, 'left_hip', 'left_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'right_wrist'
            ),


            # Five joints.


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


            # Cross body.


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

            # 신체 꺾임의 방향.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str) -> np.ndarray:
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) / 2

    def _get_distance_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str) -> np.ndarray:
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from: np.ndarray, lmk_to: np.ndarray) -> np.ndarray:
        return lmk_to - lmk_from

# Pose Classification

class PoseSample(object):

    def __init__(self, name: str, landmarks: np.ndarray, class_name: str, embedding: np.ndarray):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name
        self.embedding = embedding


class PoseSampleOutlier(object):

    def __init__(self, sample: PoseSample, detected_class: list[str], all_classes: dict[str, int]):
        self.sample = sample
        self.detected_class = detected_class
        self.all_classes = all_classes


class PoseClassifier(object):
    """랜드마크로 포즈를 분류합니다."""

    def __init__(self, pose_samples_folder: str, pose_embedder: FullBodyPoseEmbedder, file_extension: str='csv', file_separator: str=',', n_landmarks: int=33, n_dimensions: int=3, top_n_by_max_distance: int=30, top_n_by_mean_distance: int=10, axes_weights: tuple | list=(1.0, 1.0, 0.2)):
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights

        self._pose_samples = self._load_pose_samples(pose_samples_folder, file_extension, file_separator, n_landmarks, n_dimensions, pose_embedder)

    def _load_pose_samples(self, pose_samples_folder: str, file_extension: str, file_separator: str, n_landmarks: int, n_dimensions: int, pose_embedder: FullBodyPoseEmbedder) -> list[PoseSample]:
        """주어진 폴더로부터 포즈 샘플을 가져옵니다.

        요구되는 파일 구조:
            neutral_standing.csv
            pushups_down.csv
            pushups_up.csv
            squats_down.csv
            ...

        요구되는 CSV 구조:
            sample_00001,x1,y1,z1,x2,y2,z2,....
            sample_00002,x1,y1,z1,x2,y2,z2,....
            ...
        """
        # 폴더 내의 각 파일은 하나의 포즈 클래스를 나타낸다.
        file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

        pose_samples = []
        for file_name in file_names:
            # 파일 이름을 포즈 클래스 이름으로 사용한다.
            class_name = file_name[:-(len(file_extension) + 1)]

            # CSV 파싱
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=file_separator)
                for row in csv_reader:
                    assert len(row) == n_landmarks * n_dimensions + 1, f'Wrong number of values: {len(row)}'
                    landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
                    pose_samples.append(
                        PoseSample(
                            name=row[0],
                            landmarks=landmarks,
                            class_name=class_name,
                            embedding=pose_embedder(landmarks)
                        )
                    )

        return pose_samples

    def find_pose_sample_outliers(self) -> list[PoseSampleOutlier]:
        """전체 데이터베이스에 대하여 각 샘플을 분류합니다."""
        # 타겟 포즈에서 이상치를 찾는다.
        outliers = []
        for sample in self._pose_samples:
            # 대상에 가장 가까운 포즈를 찾는다.
            pose_landmarks = sample.landmarks.copy()
            pose_classification = self.__call__(pose_landmarks)
            class_names = [class_name for class_name, count in pose_classification.items() if count == max(pose_classification.values())]

            # 가장 가까운 포즈의 클래스가 다르거나 둘 이상의 포즈 클래스가 가장 가까운 포즈로 탐지된 경우 표본이 이상치(outlier)이다.
            if sample.class_name not in class_names or len(class_names) != 1:
                outliers.append(PoseSampleOutlier(sample, class_names, pose_classification))

        return outliers

    def __call__(self, pose_landmarks: np.ndarray) -> dict[str, int]:
        """주어진 포즈를 분류합니다.

        분류 작업은 두 단계로 이루어져 있습니다:
            * 먼저, MAX 거리별로 상위 N개의 샘플을 선택합니다. 주어진 포즈와 거의 똑같은 샘플은 삭제가 가능하지만,
            반대 방향으로 구부러진 joint는 거의 없습니다.
            * 그런 다음, 평균 거리를 기준으로 상위 N개의 샘플을 선택합니다. 이전 단계에서 이상치를 제거한 후
            평균적으로 가까운 표본을 선택할 수 있습니다.

        Args:
            pose_landmarks: (N, 3)과 같은 형상의 3D 랜드마크가 있는 Numpy 배열을 인자로 받습니다.

        Returns:
            데이터베이스에서 가장 가까운 포즈 샘플 수가 포함된 Dictionary를 반환합니다. 예시:
                {
                    'pushups_down': 8,
                    'pushups_up': 2,
                }
        """
        # 제공된 포즈와 대상 포즈의 모양이 동일한지 확인한다.
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

        # 포즈 임베딩을 얻는다.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1])) # 랜드마크가 좌우반전된 임베딩

        # 최대 거리로 분류한다.
        #
        # 이것은 이상치를 제거하는 데 도움을 준다. -> 주어진 포즈와 거의 동일하지만
        # 관절 하나가 다른 방향으로 꺾여있고 실제로는 다른 포즈 클래스를 나타낸다.
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights)
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        # 평균 거리로 분류한다.
        #
        # 결측치를 제거한 후엔 평균 거리로 가장 가까운 포즈를 찾을 수 있다.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights)
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        # 결과를 dict에 집어넣는다. (class_name -> n_samples)
        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        result = { class_name: class_names.count(class_name) for class_name in set(class_names) }

        return result


class EMADictSmoothing(object):
    """포즈 분류를 매끄럽게 해 줍니다."""

    def __init__(self, window_size: int=10, alpha: float=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data: dict[str, int]) -> dict[str, float]:
        """주어진 포즈의 분류를 매끄럽게 해 줍니다.

        Smoothing은 지정된 시간 창에서 관찰된 모든 포즈 클래스에 대한 지수 평균 이동(Expotential Moving Average, EMA)을
        계산하여 수행됩니다. 누락된 포즈 클래스는 0으로 계산됩니다.

        Args:
            data: 포즈 분류가 포함된 Dictionary. 예시:
                {
                    'pushups_down': 8,
                    'pushups_up': 2,
                }

        Result:
            위와 동일한 형식의 Dictionary 형식이나, 정수 값 대신
            Smooth 작업이 완료된 실수 값이 포함된 Dictionary를 반환합니다. 예시:
                {
                    'pushups_down': 8.3,
                    'pushups_up': 1.7,
                }
        """
        # 간단한 코드를 위하여 창 시작 부분에 새 데이터를 추가한다.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[:self._window_size]

        # 모든 키들을 얻는다.
        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # 부드럽게 한 데이터를 얻는다.
        smoothed_data = dict()
        for key in keys:
            factor, top_sum, bottom_sum = 1.0, 0.0, 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # factor 재갱신
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data


class RepetitionCounter(object):
    """지정된 대상 포즈 클래스의 반복 횟수를 카운트합니다."""

    def __init__(self, class_name: str, enter_threshold: int = 6, exit_threshold: int = 4):
        self._class_name = class_name

        # 포즈 카운터가 지정된 임계값(threshold)을 통과한다면, 포즈를 입력한다.
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # 주어진 자세를 취했는지 아닌지 여부를 저장하는 변수.
        self._pose_entered = False

        # 포즈를 종료한 횟수를 저장하는 변수.
        self._n_repeats = 0

    @property
    def n_repeats(self):
        return self._n_repeats

    def __call__(self, pose_classification: dict[str, float]) -> int:
        """지정된 프레임까지 발생한 반복 횟수를 카운트합니다.

        두 가지의 임계값을 사용합니다. 첫번째는 높은 쪽으로 올라가야 자세에 들어가고,
        그 다음에 낮은 쪽으로 내려가야 자세에서 나옵니다. 임계값 간의 차이로 인해
        jittering을 예측할 수 있습니다. ( 임계값이 하나만 있는 경우에는 
        잘못된 카운트가 발생합니다. )

        Args:
            pose_classification: 현재 프레임의 포즈 분류 딕셔너리를 받습니다. 예시:
                {
                    'pushups_down': 8.3,
                    'pushups_up': 1.7,
                }

        Returns:
            반복 횟수를 정수형으로 반환합니다.
        """
        # 포즈의 신뢰도를 가져옵니다.
        pose_confidence = 0.0
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]

        # 아직 포즈를 취하지 않았다면 현재 포즈 신뢰도가 포즈 임계점을 넘었는지 확인하고,
        # 임계점을 넘었을 시 상태를 변경합니다.
        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            return self._n_repeats

        # 포즈를 취한 후 포즈를 마치면 카운터를 늘리고 상태를 업데이트합니다.
        if pose_confidence < self._exit_threshold:
            self._n_repeats += 1
            self._pose_entered = False

        return self._n_repeats


class PoseClassificationVisualizer(object):
    """모든 프레임에 대한 분류를 추적하고 렌더링합니다."""

    def __init__(
        self,
        class_name,
        plot_location_x=0.05,
        plot_location_y=0.05,
        plot_max_width=0.4,
        plot_max_height=0.4,
        plot_figsize=(9, 4),
        plot_x_max=None,
        plot_y_max=None,
        counter_location_x=0.85,
        counter_location_y=0.05,
        counter_font_path='https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true',
        counter_font_color='red',
        counter_font_size=0.15
    ):
        self._class_name = class_name
        self._plot_location_x = plot_location_x
        self._plot_location_y = plot_location_y
        self._plot_max_width = plot_max_width
        self._plot_max_height = plot_max_height
        self._plot_figsize = plot_figsize
        self._plot_x_max = plot_x_max
        self._plot_y_max = plot_y_max
        self._counter_location_x = counter_location_x
        self._counter_location_y = counter_location_y
        self._counter_font_path = counter_font_path
        self._counter_font_color = counter_font_color
        self._counter_font_size = counter_font_size

        self._counter_font = None

        self._pose_classification_history = []
        self._pose_classification_filtered_history = []

    def __call__(self, frame, pose_classification, pose_classification_filtered, repetitions_count):
        """주어진 프레임까지 포즈 분류 및 카운터를 렌더합니다."""
        # classification 기록 확장
        # self._pose_classification_history.append(pose_classification)
        # self._pose_classification_filtered_history.append(
        #     pose_classification_filtered)

        # classification plot과 counter가 있는 프레임 출력
        output_img = Image.fromarray(frame)

        output_width = output_img.size[0]
        output_height = output_img.size[1]

        # plot 그리기
        # img = self._plot_classification_history(output_width, output_height)
        # img.thumbnail((int(output_width * self._plot_max_width),
        #                int(output_height * self._plot_max_height)),
        #               Image.ANTIALIAS)
        # output_img.paste(
        #     img, (
        #         int(output_width * self._plot_location_x),
        #         int(output_height * self._plot_location_y)
        #     )
        # )

        # count 그리기
        output_img_draw = ImageDraw.Draw(output_img)
        if self._counter_font is None:
            font_size = int(output_height * self._counter_font_size)
            font_request = requests.get(self._counter_font_path, allow_redirects=True)
            self._counter_font = ImageFont.truetype(io.BytesIO(font_request.content), size=font_size)

        output_img_draw.text(
            (output_width * self._counter_location_x, output_height * self._counter_location_y),
            str(repetitions_count),
            font=self._counter_font,
            fill=self._counter_font_color
        )

        return output_img

    def _plot_classification_history(self, output_width, output_height):
        fig = plt.figure(figsize=self._plot_figsize)

        for classification_history in [self._pose_classification_history, self._pose_classification_filtered_history]:
            y = []
            for classification in classification_history:
                if classification is None:
                    y.append(None)
                elif self._class_name in classification:
                    y.append(classification[self._class_name])
                else:
                    y.append(0)
            plt.plot(y, linewidth=7)

        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Frame')
        plt.ylabel('Confidence')
        plt.title('Classification history for `{}`'.format(self._class_name))
        # plt.legend(loc='upper right')

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)

        # plot을 이미지로 변환
        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1])
        )
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img


class BootstrapHelper(object):
    """분류를 위해 이미지를 부트스트랩하고 포즈 샘플을 필터링하는 데 도움을 줍니다."""

    def __init__(self, images_in_folder, images_out_folder, csvs_out_folder, pose_class_name):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_folder = csvs_out_folder
        self._pose_class_name = pose_class_name

    def bootstrap(self, per_pose_class_limit=None):
        """주어진 폴더에 포함된 부트스트랩 이미지

        폴더 내에 요구되는 이미지 예시 ( 이미지 out 폴더와 동일한 사용 ):
            pushups_up/
                image_001.jpg
                image_002.jpg
                ...
            pushups_down/
                image_001.jpg
                image_002.jpg
                ...
            ...

        생산되는 CSV 출력 폴더:
            pushups_up.csv
            pushups_down.csv

        포즈 3D 랜드마크를 사용한 CSV 구조 제작 예시:
            sample_00001,x1,y1,z1,x2,y2,z2,....
            sample_00002,x1,y1,z1,x2,y2,z2,....
        """
        # CSV를 위한 폴더를 만든다.
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)

        pose_class_name = self._pose_class_name
        print('Bootstrapping ', pose_class_name, file=sys.stderr)

        # 포즈 클래스의 경로
        images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
        images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
        csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')

        if not os.path.exists(images_out_folder): # out 폴더가 존재하지 않을 경우
            os.makedirs(images_out_folder) # 자동 생성

        with open(csv_out_path, 'w') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            # 이미지들의 리스트를 얻는다.
            image_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
            if per_pose_class_limit is not None:
                image_names = image_names[:per_pose_class_limit]

            # 모든 이미지를 Bootstrap 한다.
            for image_name in tqdm.tqdm(image_names):
                # 이미지를 불러온다.
                input_frame = cv2.imread(os.path.join(images_in_folder, image_name))
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                # 새 포즈 트래커를 초기화하고 실행한다.
                with mp_pose.Pose(model_complexity=2) as pose_tracker:
                    result = pose_tracker.process(image=input_frame)
                    pose_landmarks = result.pose_landmarks

                # 포즈 예측을 사용하여 이미지를 저장한다. ( 포즈가 감지된 경우 )
                output_frame = input_frame.copy()
                if pose_landmarks is not None:
                    mp_drawing.draw_landmarks(image=output_frame, landmark_list=pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

                # 포즈가 감지된 경우 랜드마크를 저장한다.
                if pose_landmarks is not None:
                    # 랜드마크를 얻는다.
                    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                    pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in pose_landmarks.landmark], dtype=np.float32)

                    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                    csv_out_writer.writerow([image_name] + pose_landmarks.flatten().astype(np.str_).tolist())

                # XZ 투영(projection) 을 그려 이미지와 연결(concatenate)한다.
                projection_xz = self._draw_xz_projection(output_frame=output_frame, pose_landmarks=pose_landmarks)
                output_frame = np.concatenate((output_frame, projection_xz), axis=1)

    def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
        img = Image.new('RGB', (frame_width, frame_height), color='white')

        if pose_landmarks is None:
            return np.asarray(img)

        # 영상 너비에 따라 반지름을 조정한다.
        r *= frame_width * 0.01

        draw = ImageDraw.Draw(img)
        for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS:
            # Z를 뒤집고 힙을 영상의 중앙으로 이동시킵니다.
            x1, y1, z1 = pose_landmarks[idx_1] * \
                [1, 1, -1] + [0, 0, frame_height / 2]
            x2, y2, z2 = pose_landmarks[idx_2] * \
                [1, 1, -1] + [0, 0, frame_height / 2]

            draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
            draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
            draw.line([x1, z1, x2, z2], width=int(r), fill=color)

        return np.asarray(img)

    def align_images_and_csvs(self, print_removed_items=False):
        """이미지 폴더와 CSV의 샘플이 동일한지 확인합니다.

        이미지 폴더와 CSV 모두에 샘플의 교차점만 남깁니다.
        """
        pose_class_name = self._pose_class_name
        # 포즈 클래스의 경로
        images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
        csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')

        # CSV를 메모리로 읽는다. ( 원문 : Read CSV into memory )
        rows = []
        with open(csv_out_path) as csv_out_file:
            csv_out_reader = csv.reader(csv_out_file, delimiter=',')
            for row in csv_out_reader:
                rows.append(row)

        # CSV 에 남아 있는 이미지 이름을 저장할 배열
        image_names_in_csv = []

        # 해당 영상 없이 CSV 제거 라인을 재작성한다.
        with open(csv_out_path, 'w') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for row in rows:
                image_name = row[0]
                image_path = os.path.join(images_out_folder, image_name)
                if os.path.exists(image_path):
                    image_names_in_csv.append(image_name)
                    csv_out_writer.writerow(row)
                elif print_removed_items:
                    print(f'Removed image from CSV: {image_path}')

        # CSV에서 해당하는 라인이 없는 이미지를 제거한다.
        for image_name in os.listdir(images_out_folder):
            if image_name not in image_names_in_csv:
                image_path = os.path.join(images_out_folder, image_name)
                os.remove(image_path)
                if print_removed_items:
                    print('Removed image from folder: ', image_path)

    def analyze_outliers(self, outliers):
        """이상치를 찾기 위해 각 표본을 다른 모든 표본과 비교하여 분류합니다.

        샘플이 원래 클래스와 다르게 분류된 경우 -> 삭제하거나 유사한 샘플을 추가해야 합니다.
        """
        for outlier in outliers:
            image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)

            print('Outlier')
            print('  sample path =    ', image_path)
            print('  sample class =   ', outlier.sample.class_name)
            print('  detected class = ', outlier.detected_class)
            print('  all classes =    ', outlier.all_classes)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            show_image(img, figsize=(20, 20))

    def remove_outliers(self, outliers):
        """이미지 폴더에서 이상치를 제거합니다."""
        for outlier in outliers:
            image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)
            os.remove(image_path)

    def print_images_in_statistics(self):
        """입력 이미지 폴더에서 통계를 출력합니다."""
        self._print_images_statistics(self._images_in_folder, self._pose_class_names)

    def print_images_out_statistics(self):
        """출력 이미지 폴더에서 통계를 출력합니다."""
        self._print_images_statistics(self._images_out_folder, self._pose_class_names)

    def _print_images_statistics(self, images_folder, pose_class_names):
        print('Number of images per pose class:')
        for pose_class_name in pose_class_names:
            n_images = len([n for n in os.listdir(os.path.join(images_folder, pose_class_name)) if not n.startswith('.')])
            print('  {}: {}'.format(pose_class_name, n_images))


class Performance:
    def __init__(self) -> None:
        self.time = 0

    def setStartPoint(self) -> None:
        self.time = time.time()

    def getEndPoint(self) -> float:
        return (time.time() - self.time) * 1000

    def printEndPoint(self) -> None:
        print("WorkingTime: {} ms".format(self.getEndPoint()))
