import cv2
import sys
dirName = sys.argv[1]
fileName = sys.argv[2]
vidcap = cv2.VideoCapture(f'./data/{dirName}/video/{fileName}')
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(f'./data/{dirName}/in/{fileName.split(".")[0]}/%06d.jpg' %
                count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

print("finish! convert video to frame")

f'{cv2}'
