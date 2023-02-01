import util
import mediapipe as mp
import argparse
import os

mp_pose = mp.solutions.pose

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=argparse.FileType())
parser.add_argument('--out', '-o', help='부트스트래핑한 CSV파일을 내보낼 디렉토리를 입력받습니다')
parser.add_argument('--complexity', '-c', help='모델 복잡도를 설정합니다', default=1)
# parser.add_argument('--progress', '-p', help='프로그래스바를 출력합니다', action='store_true', default=False)

args = parser.parse_args()
args.filename = args.filename.name

bootstrapper = util.Bootstrapper(mp_pose.Pose(            
    model_complexity=1,
    static_image_mode=True))
up = bootstrapper(args.filename)
csv_saver = util.CSVSaver()
csv_saver(os.path.splitext(args.filename)[0], up)