import util
import mediapipe as mp
mp_pose = mp.solutions.pose

bootstrapper = util.Bootstrapper(mp_pose.Pose(model_complexity=1))
up = bootstrapper('./data/test/up.mp4')
down = bootstrapper('./data/test/down.mp4')

csv_saver = util.CSVSaver()
csv_saver('./data/test/up', up)
csv_saver('./data/test/down', down)