import mediapipe
import cv2 as cv
import math
import matplotlib.pyplot as plt

mp_pose = mediapipe.solutions.pose
angleCategories = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'left_hip_y', 'right_hip_y', 'left_hip_x', 'right_hip_x', 'left_shoulder_x', 'right_shoulder_x', 'left_shoulder_x', 'right_shoulder_x', 'left_ankle', 'right_ankle']
plotData = { category: [] for category in angleCategories }

class PoseAngle:
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
        ['right_ankle', [26, 28, 32]],
    ]
    def __init__(self) -> None:
        self.pose = mp_pose.Pose(
            model_complexity=2,
            static_image_mode=False,
            min_detection_confidence=0.5
        )
    
    def __call__(self, image) -> dict[str, list[float]]:
        result = self.pose.process(image)
        return {} if not result.pose_landmarks else { angleCategories[idx]: 
            PoseAngle.ThreeDegree(
                result.pose_landmarks.landmark[angle[1][0]],
                result.pose_landmarks.landmark[angle[1][1]],
                result.pose_landmarks.landmark[angle[1][2]]
            ) * 180 / math.pi for idx, angle in enumerate(PoseAngle.angles) 
        }

    @staticmethod
    def ThreeDegree(a, b, c) -> float:
        ab = [b.x - a.x, b.y - a.y, b.z - a.z]
        bc = [c.x - b.x, c.y - b.y, c.z - b.z]
        abVec = math.sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2])
        bcVec = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2])
        abNorm = [ab[0] / abVec, ab[1] / abVec, ab[2] / abVec]
        bcNorm = [bc[0] / bcVec, bc[1] / bcVec, bc[2] / bcVec]
        res = abNorm[0] * bcNorm[0] + abNorm[1] * bcNorm[1] + abNorm[2] * bcNorm[2]
        return math.pi - math.acos(res)
    
getAngle = PoseAngle()
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        cv.imshow('webcam', frame)
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        for key, value in getAngle(image=image).items():
            plotData[key].append(value)

        key = cv.waitKey(1000 // 60)
        if key & 0xFF == ord('x'):
            break

for key, value in plotData.items():
    plt.plot(value)
plt.show()