import cv2


cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
    print('Unable to read camera feed')

while True:
    ret, color = cap.read()
    if color != []:
        cv2.imshow('uvc', color)
        image = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        cv2.waitKey(1)