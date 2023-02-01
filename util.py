import pandas as pd
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2 as cv
import os
import tqdm
import csv
import math
import io
<<<<<<< HEAD

from PIL import Image, ImageFont, ImageDraw
import requests
import cv2
from matplotlib import pyplot as plt
=======
import requests
>>>>>>> 9595b6665bb7e051bf77a0b74b189bb3cb37cca0

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Dropout
from keras.optimizers.legacy.sgd import SGD
from keras.optimizers.legacy.adam import Adam
from keras import models
from PIL import Image, ImageFont, ImageDraw

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
    def __init__(self, pose):
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
        
        return np.array(result, dtype=np.float32)
    
    @staticmethod
    def pose_flatter(landmarks):
        tmp_arr = []
        for landmark in mp_pose.PoseLandmark:
            tmp_arr.append(landmarks[landmark].x)
            tmp_arr.append(landmarks[landmark].y)
            tmp_arr.append(landmarks[landmark].z)
        return np.array(tmp_arr, dtype=np.float32)

            
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


class PoseEmbedderByAngle:
    angles = [
        ['left_elbow', [11, 13, 15]],
        ['right_elbow', [12, 14, 16]],
        ['left_knee', [23, 25, 27]],
        ['right_knee', [24, 26, 28]],
        ['left_hip_y', [25, 23, 24]],
        ['right_hip_y', [26, 24, 23]],
        ['left_hip_x', [11, 23, 25]],
        ['right_hip_x', [12, 24, 26]],
        ['left_shoulder_x', [13, 11, 23]],
        ['right_shoulder_x', [14, 12, 24]],
        ['left_shoulder_y', [13, 11, 12]],
        ['right_shoulder_y', [14, 12, 11]],
        ['left_ankle', [25, 27, 31]],
        ['right_ankle', [26, 28, 32]]
    ]
    
    
    def __init__(self):
        pass

    def __call__(self, landmark):
        return np.array([round(PoseEmbedderByAngle.three_angle(*[landmark[item[1][i]] for i in range(3)]), 5) for item in PoseEmbedderByAngle.angles], dtype=np.float32)

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
        return [item[0] for item in PoseEmbedderByAngle.angles]

class PoseEmbedderByDistance:
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
                landmarks, 'left_hip', 'right_wrist'
            ),
            self._get_distance_by_names(
                landmarks, 'right_hip', 'left_wrist'
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
        ]).flatten()
    def _get_average_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str) -> np.ndarray:
        lmk_from = landmarks[PoseEmbedderByDistance._landmark_names.index(name_from)]
        lmk_to = landmarks[PoseEmbedderByDistance._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) / 2
    def _get_distance_by_names(self, landmarks: np.ndarray, name_from: str, name_to: str) -> np.ndarray:
        lmk_from = landmarks[PoseEmbedderByDistance._landmark_names.index(name_from)]
        lmk_to = landmarks[PoseEmbedderByDistance._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)
    def _get_distance(self, lmk_from: np.ndarray, lmk_to: np.ndarray) -> np.ndarray:
        return lmk_to - lmk_from
    
class PoseClassifierByKNN:
    def __init__(self, target: dict[str, np.ndarray], embeder: PoseEmbedderByAngle | PoseEmbedderByDistance, axes_weights: tuple=(1.0, 1.0, 0.2), top_n_by_max_distance: int=30, top_n_by_mean_distance: int=9):
        tmp_target: list[np.ndarray] = []
        target_dict: dict[str, int] = dict()
        
        for k in target: # ['up', 'down']
            tmp_target.extend(target[k]) # tmp_target에 up, down 임베딩들을 모두 넣어준다.
            target_dict[k] = len(target[k]) # target_dict에는 해당 클래스의 프레임 개수를 넣어준다.

        self._target = tmp_target
        self._dict = target_dict
        self._pose_embedder = embeder
        self._axes_weights = axes_weights
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance

    def __call__(self, pose_landmarks: np.ndarray) -> dict[str, int]:
        """주어진 포즈를 분류합니다.

        분류 작업은 두 단계로 이루어져 있습니다:
            * 먼저, MAX 거리별로 상위 N개의 샘플을 선택합니다.
            * 그런 다음, 평균 거리를 기준으로 상위 N개의 샘플을 선택합니다.
            이전 단계에서 이상치를 제거하였기 때문에 평균적으로 가까운 표본을 선택할 수 있습니다.

        Args:
            pose_landmarks: (N, 3)과 같은 shape의 3D 랜드마크가 있는 Numpy 배열을 인자로 받습니다.

        Returns:
            데이터베이스에서 가장 가까운 포즈 샘플 수가 포함된 Dictionary를 반환합니다. 예시:
                {
                    'down': 7,
                    'up': 2,
                }
        """
        
        # 포즈 임베딩을 얻는다.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1])) # 랜드마크가 좌우반전된 임베딩

        # 최대 거리로 분류한다.
        #
        # 이것은 이상치를 제거하는 데 도움을 준다. -> 주어진 포즈와 거의 동일하지만
        # 관절 하나가 다른 방향으로 꺾여있고 실제로는 다른 포즈 클래스를 나타낸다.
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._target):
            max_dist = min(
                np.max(np.abs(sample - pose_embedding)),
                np.max(np.abs(sample - flipped_pose_embedding))
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        # 평균 거리로 분류한다.
        #
        # 결측치를 제거한 후엔 평균 거리로 가장 가까운 포즈를 찾을 수 있다.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            mean_dist = min(
                np.mean(np.abs(sample - pose_embedding)),
                np.mean(np.abs(sample - flipped_pose_embedding))
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        res = { name: 0 for name in self._dict }
        for _, idx in mean_dist_heap:
            for name, frames in self._dict.items():
                # idx에 올 수 있는 가장 최댓값은 dict에 들어있는 두 frames들의 합이다.
                # 각 차례마다 현재 차례의 힙 원소의 idx에서 현재 클래스의 frames 값을 빼 본다.
                # 만약 0 이하로 내려가는 순간 현재 차례의 클래스이다.
                idx -= frames
                if idx <= 0:
                    res[name] += 1
                    break
        return res

class ModelComplier:
    def __init__(self) -> None:
        self._model = Sequential([
            Dense(128, activation='tanh'),
            Dropout(0.1),
            Dense(32, activation='tanh'),
            Dropout(0.1),
            Dense(8, activation='tanh'),
            Dense(1, activation='sigmoid')
        ])
    def compile(self, datasets: dict[int | float, np.ndarray]):
        self._model.compile(optimizer=Adam(),
          loss=tf.keras.losses.binary_crossentropy,
          metrics=['accuracy'])

        xdata = []
        ydata = []

        for k, v in datasets.items():
            xdata.extend(v)
            ydata.extend([k for _ in range(len(v))])
            
        return self._model.fit(np.array(xdata), np.array(ydata), epochs=64, batch_size=32)
        
    def save(self, filename: str):
        self._model.save(filename)


class PoseClassifierByML:
    def __init__(self, modelfile: str, embeder: PoseEmbedderByAngle | PoseEmbedderByDistance, posedict: dict[int, str]):
        self._model = models.load_model(modelfile)
        self._model.summary()
        self._embeder = embeder
        self._posedict = posedict

    def __call__(self, pose_landmarks: np.ndarray):
        predict_result = self._model.predict(np.array([self._embeder(pose_landmarks * np.array([100, 100, 100]))], dtype=np.float32), verbose=0)[0][0]
        # return {"up": 10, "down": 0}
        return { 'up': 10, 'down': 0 } if predict_result > 0.1 else { 'up': 0, 'down': 10 }

<<<<<<< HEAD

class EMADictSmoothing(object):
    """포즈 분류를 매끄럽게 해 줍니다."""

    def __init__(self, window_size: int=10, alpha: float=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window: list[dict] = []

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
        # 창 시작 부분에 새 데이터를 추가한다.
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


=======
>>>>>>> 9595b6665bb7e051bf77a0b74b189bb3cb37cca0
class PoseClassificationVisualizer(object):
    """모든 프레임에 대한 분류를 추적하고 렌더링합니다."""

    def __init__(
        self,
        class_name: str,
        plot_location_x: int | float=0.05,
        plot_location_y: int | float=0.05,
        plot_max_width: int | float=0.4,
        plot_max_height: int | float=0.4,
        plot_figsize: tuple=(9, 4),
        plot_x_max: int | float | None=None,
        plot_y_max: int | float | None=None,
        counter_location_x: int | float=0.85,
        counter_location_y: int | float=0.05,
        counter_font_path: str='https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true',
        counter_font_color: str='red',
        counter_font_size: int | float=0.15
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

        self._counter_font: ImageFont.FreeTypeFont | None = None

        self._pose_classification_history = []
        self._pose_classification_filtered_history = []

    def __call__(self, frame, pose_classification, pose_classification_filtered, repetitions_count):
        """주어진 프레임까지 포즈 분류 및 카운터를 렌더합니다."""
        # classification 기록 확장
        self._pose_classification_history.append(pose_classification)
<<<<<<< HEAD
        self._pose_classification_filtered_history.append(
            pose_classification_filtered)
=======
        # self._pose_classification_filtered_history.append(
        #     pose_classification_filtered)
>>>>>>> 9595b6665bb7e051bf77a0b74b189bb3cb37cca0

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

<<<<<<< HEAD
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
=======
        # # count 그리기
        # output_img_draw = ImageDraw.Draw(output_img)
        # if self._counter_font is None:
        #     font_size = int(output_height * self._counter_font_size)
        #     font_request = requests.get(self._counter_font_path, allow_redirects=True)
        #     self._counter_font = ImageFont.truetype(io.BytesIO(font_request.content), size=font_size)

        # output_img_draw.text(
        #     (output_width * self._counter_location_x, output_height * self._counter_location_y),
        #     str(repetitions_count),
        #     font=self._counter_font,
        #     fill=self._counter_font_color
        # )
>>>>>>> 9595b6665bb7e051bf77a0b74b189bb3cb37cca0

        return output_img

    def _plot_classification_history(self, output_width, output_height):
        fig = plt.figure(figsize=self._plot_figsize)

<<<<<<< HEAD
        for classification_history in [self._pose_classification_history, self._pose_classification_filtered_history]:
=======
        for classification_history in [self._pose_classification_history]:
>>>>>>> 9595b6665bb7e051bf77a0b74b189bb3cb37cca0
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