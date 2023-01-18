import os
import subprocess
for i in os.listdir('./data/'):
    try:
        subprocess.call(['python', './run.py', i], shell=False)
        os.rename(f'./data/{i}/video/result.mp4', f'./dist/{i}.mp4')
    except:
        pass
