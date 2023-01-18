import cv2


cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
    print('Unable to read camera feed')

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('uvc', frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        key = cv2.waitKey(1000 // 60)
        if key == 113:
            break
