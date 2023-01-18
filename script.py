import subprocess
import os
import sys


exerciseDir = f'./data/{sys.argv[1]}/'

videos = [video for video in os.listdir(exerciseDir) if video[0] != '.']

for directory in ['dist', 'in', 'out', 'video']:
    os.makedirs(exerciseDir + directory)

for video in videos:
    os.replace(exerciseDir + video, exerciseDir + 'video/' + video)
    if video.find('test') == -1 and video.find('result') == -1:
        os.mkdir(exerciseDir + 'in/' + video.split('.')[0])
        os.mkdir(exerciseDir + 'out/' + video.split('.')[0])
        subprocess.call(
            ['python', './videosplitter.py', sys.argv[1], video], shell=False)

subprocess.call(['python', './main.py', sys.argv[1]], shell=False)
subprocess.call(['python', './run.py', sys.argv[1]], shell=False)
