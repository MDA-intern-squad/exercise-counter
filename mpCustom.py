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


def show_image(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    # plt.show()


class FullBodyPoseEmbedder(object):
    """3D 포즈 랜드마크를 3D 포즈 임베딩으로 변환합니다."""

    def __init__(self, torso_size_multiplier: float = 2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
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

    def __call__(self, landmarks: np.ndarray):
        """포즈의 랜드마크를 표준화 하고 Embedding으로 변환합니다.

        Args:
            landmarks - (N, 3)과 같은 형상의 3D 랜드마크가 있는 Numpy 배열을 인자로 받습니다.

        Result:
            (M, 3)과 같은 형상의 Pose Embedding이 있는 Numpy 배열을 반환합니다.
            여기서 'M은' `_get_pose_distance_embedding` 에서 정의된 Pairwisde Distances의 개수입니다.
        """
        assert landmarks.shape[0] == len(self._landmark_names), '예외되지 않은 랜드마크 번호: {}'.format(landmarks.shape[0])

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
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks: np.ndarray, torso_size_multiplier: float) -> np.float64:
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # 이 접근법은 포즈의 크기를 계산하기 위해 2D 랜드마크만 사용함
        landmarks = landmarks[:, :2]

        # 엉덩이의 중심
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # 어깨의 중심
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks: np.ndarray) -> np.ndarray:
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
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

            # 신체 꺾임의 각도.

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
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from: np.ndarray, lmk_to: np.ndarray) -> np.ndarray:
        return lmk_to - lmk_from


class PoseSample(object):

    def __init__(self, name: str, landmarks: np.ndarray, class_name: str, embedding: np.ndarray):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name
        self.embedding = embedding


class PoseSampleOutlier(object):

    def __init__(self, sample, detected_class, all_classes):
        self.sample = sample
        self.detected_class = detected_class
        self.all_classes = all_classes


class PoseClassifier(object):
    """포즈의 랜드마크를 분류합니다."""

    def __init__(self, pose_samples_folder: str, pose_embedder: FullBodyPoseEmbedder, file_extension: str = 'csv', file_separator: str = ',', n_landmarks: int = 33, n_dimensions: int = 3, top_n_by_max_distance: int = 30, top_n_by_mean_distance: int = 10, axes_weights: tuple | list = (1.0, 1.0, 0.2)) -> None:
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
                    assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
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

            # 가장 가까운 포즈의 클래스가 다르거나 둘 이상의 포즈 클래스가 가장 가까운 포즈로 탐지된 경우 표본이 특이치(outlier)이다.
            if sample.class_name not in class_names or len(class_names) != 1:
                outliers.append(PoseSampleOutlier(sample, class_names, pose_classification))

        return outliers

    def __call__(self, pose_landmarks):
        """주어진 포즈를 분류합니다.

        분류 작업은 두 단계로 이루어져 있습니다:
            * 먼저, MAX 거리별로 상위 N개의 샘플을 선택합니다. 주어진 포즈와 거의 똑같은 샘플은 삭제가 가능하지만,
            반대 방향으로 구부러진 joint는 거의 없습니다.
            * 그런 다음, 평균 거리를 기준으로 상위 N개의 샘플을 선택합니다. 이전 단계에서 특이치를 제거한 후
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
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        # 최대 거리로 분류한다.
        #
        # 이것은 특이치를 제거하는 데 도움을 준다. -> 주어진 포즈와 거의 동일하지만
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

        # Collect results into map: (class_name -> n_samples)
        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        result = { class_name: class_names.count(class_name) for class_name in set(class_names) }

        return result


class EMADictSmoothing(object):
    """포즈 분류를 매끄럽게 해 줍니다."""

    def __init__(self, window_size: int = 10, alpha: float = 0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        """주어진 포즈의 분류를 매끄럽게 해 줍니다.

        Smoothing은 지정된 시간 창에서 관찰된 모든 포즈 클래스에 대한 지수 이동 평균을
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
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
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

    def __call__(self, pose_classification):
        """지정된 프레임까지 발생한 반복 횟수를 카운트합니다.

        두 가지의 임계값을 사용합니다. 첫번째는 높은 쪽으로 올라가야 자세가 들어가고,
        그 다음에 낮은 쪽으로 내려가야 자세가 나옵니다. 임계값 간의 차이로 인해
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

        # On the very first frame or if we were out of the pose, just check if we
        # entered it on this frame and update the state.
        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            return self._n_repeats

        # 포즈를 취한 후 포즈를 마치면 카운터를 늘리고 상태를 업데이트합니다.
        if pose_confidence < self._exit_threshold:
            self._n_repeats += 1
            self._pose_entered = False

        return self._n_repeats


class PoseClassificationVisualizer(object):
    """Keeps track of claassifcations for every frame and renders them."""

    def __init__(self,
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
                 counter_font_size=0.15):
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
        self._pose_classification_history.append(pose_classification)
        self._pose_classification_filtered_history.append(
            pose_classification_filtered)

        # classification plot과 counter가 있는 프레임 출력
        output_img = Image.fromarray(frame)

        output_width = output_img.size[0]
        output_height = output_img.size[1]

        # plot 그리기
        img = self._plot_classification_history(output_width, output_height)
        img.thumbnail((int(output_width * self._plot_max_width),
                       int(output_height * self._plot_max_height)),
                      Image.ANTIALIAS)
        output_img.paste(
            img, (
                int(output_width * self._plot_location_x),
                int(output_height * self._plot_location_y)
            )
        )

        # count 그리기
        output_img_draw = ImageDraw.Draw(output_img)
        if self._counter_font is None:
            font_size = int(output_height * self._counter_font_size)
            font_request = requests.get(
                self._counter_font_path, allow_redirects=True)
            self._counter_font = ImageFont.truetype(
                io.BytesIO(font_request.content), size=font_size)
        output_img_draw.text((output_width * self._counter_location_x,
                              output_height * self._counter_location_y),
                             str(repetitions_count),
                             font=self._counter_font,
                             fill=self._counter_font_color)

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
        plt.legend(loc='upper right')

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)

        # plot을 이미지로 변환
        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1]))
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img


class BootstrapHelper(object):
    """Helps to bootstrap images and filter pose samples for classification."""

    def __init__(self,
                 images_in_folder,
                 images_out_folder,
                 csvs_out_folder):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_folder = csvs_out_folder

        # Get list of pose classes and print image statistics.
        self._pose_class_names = sorted([n for n in os.listdir(
            self._images_in_folder) if not n.startswith('.')])

    def bootstrap(self, per_pose_class_limit=None):
        """Bootstraps images in a given folder.

        Required image in folder (same use for image out folder):
          pushups_up/
            image_001.jpg
            image_002.jpg
            ...
          pushups_down/
            image_001.jpg
            image_002.jpg
            ...
          ...

        Produced CSVs out folder:
          pushups_up.csv
          pushups_down.csv

        Produced CSV structure with pose 3D landmarks:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
        """
        # Create output folder for CVSs.
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)

        for pose_class_name in self._pose_class_names:
            print('Bootstrapping ', pose_class_name, file=sys.stderr)

            # Paths for the pose class.
            images_in_folder = os.path.join(
                self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(
                self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(
                self._csvs_out_folder, pose_class_name + '.csv')
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            with open(csv_out_path, 'w') as csv_out_file:
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                # Get list of images.
                image_names = sorted([n for n in os.listdir(
                    images_in_folder) if not n.startswith('.')])
                if per_pose_class_limit is not None:
                    image_names = image_names[:per_pose_class_limit]

                # Bootstrap every image.
                for image_name in tqdm.tqdm(image_names):
                    # Load image.
                    input_frame = cv2.imread(
                        os.path.join(images_in_folder, image_name))
                    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                    # Initialize fresh pose tracker and run it.
                    with mp_pose.Pose(model_complexity=2) as pose_tracker:
                        result = pose_tracker.process(image=input_frame)
                        pose_landmarks = result.pose_landmarks

                    # Save image with pose prediction (if pose was detected).
                    output_frame = input_frame.copy()
                    if pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image=output_frame,
                            landmark_list=pose_landmarks,
                            connections=mp_pose.POSE_CONNECTIONS)
                    output_frame = cv2.cvtColor(
                        output_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(
                        images_out_folder, image_name), output_frame)

                    # Save landmarks if pose was detected.
                    if pose_landmarks is not None:
                        # Get landmarks.
                        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                        pose_landmarks = np.array(
                            [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                             for lmk in pose_landmarks.landmark],
                            dtype=np.float32)
                        assert pose_landmarks.shape == (
                            33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
                        csv_out_writer.writerow(
                            [image_name] + pose_landmarks.flatten().astype(np.str_).tolist())

                    # Draw XZ projection and concatenate with the image.
                    projection_xz = self._draw_xz_projection(
                        output_frame=output_frame, pose_landmarks=pose_landmarks)
                    output_frame = np.concatenate(
                        (output_frame, projection_xz), axis=1)

    def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
        img = Image.new('RGB', (frame_width, frame_height), color='white')

        if pose_landmarks is None:
            return np.asarray(img)

        # Scale radius according to the image width.
        r *= frame_width * 0.01

        draw = ImageDraw.Draw(img)
        for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS:
            # Flip Z and move hips center to the center of the image.
            x1, y1, z1 = pose_landmarks[idx_1] * \
                [1, 1, -1] + [0, 0, frame_height * 0.5]
            x2, y2, z2 = pose_landmarks[idx_2] * \
                [1, 1, -1] + [0, 0, frame_height * 0.5]

            draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
            draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
            draw.line([x1, z1, x2, z2], width=int(r), fill=color)

        return np.asarray(img)

    def align_images_and_csvs(self, print_removed_items=False):
        """Makes sure that image folders and CSVs have the same sample.

        Leaves only intersetion of samples in both image folders and CSVs.
        """
        for pose_class_name in self._pose_class_names:
            # Paths for the pose class.
            images_out_folder = os.path.join(
                self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(
                self._csvs_out_folder, pose_class_name + '.csv')

            # Read CSV into memory.
            rows = []
            with open(csv_out_path) as csv_out_file:
                csv_out_reader = csv.reader(csv_out_file, delimiter=',')
                for row in csv_out_reader:
                    rows.append(row)

            # Image names left in CSV.
            image_names_in_csv = []

            # Re-write the CSV removing lines without corresponding images.
            with open(csv_out_path, 'w') as csv_out_file:
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for row in rows:
                    image_name = row[0]
                    image_path = os.path.join(images_out_folder, image_name)
                    if os.path.exists(image_path):
                        image_names_in_csv.append(image_name)
                        csv_out_writer.writerow(row)
                    elif print_removed_items:
                        print('Removed image from CSV: ', image_path)

            # Remove images without corresponding line in CSV.
            for image_name in os.listdir(images_out_folder):
                if image_name not in image_names_in_csv:
                    image_path = os.path.join(images_out_folder, image_name)
                    os.remove(image_path)
                    if print_removed_items:
                        print('Removed image from folder: ', image_path)

    def analyze_outliers(self, outliers):
        """Classifies each sample agains all other to find outliers.

        If sample is classified differrrently than the original class - it sould
        either be deleted or more similar samples should be aadded.
        """
        for outlier in outliers:
            image_path = os.path.join(
                self._images_out_folder, outlier.sample.class_name, outlier.sample.name)

            print('Outlier')
            print('  sample path =    ', image_path)
            print('  sample class =   ', outlier.sample.class_name)
            print('  detected class = ', outlier.detected_class)
            print('  all classes =    ', outlier.all_classes)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            show_image(img, figsize=(20, 20))

    def remove_outliers(self, outliers):
        """Removes outliers from the image folders."""
        for outlier in outliers:
            image_path = os.path.join(
                self._images_out_folder, outlier.sample.class_name, outlier.sample.name)
            os.remove(image_path)

    def print_images_in_statistics(self):
        """Prints statistics from the input image folder."""
        self._print_images_statistics(
            self._images_in_folder, self._pose_class_names)

    def print_images_out_statistics(self):
        """Prints statistics from the output image folder."""
        self._print_images_statistics(
            self._images_out_folder, self._pose_class_names)

    def _print_images_statistics(self, images_folder, pose_class_names):
        print('Number of images per pose class:')
        for pose_class_name in pose_class_names:
            n_images = len([
                n for n in os.listdir(
                    os.path.join(images_folder, pose_class_name)
                )
                if not n.startswith('.')])
            print('  {}: {}'.format(pose_class_name, n_images))
