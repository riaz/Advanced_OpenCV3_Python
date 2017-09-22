import numpy as np
import cv2

cap = cv2.VideoCapture("/Users/rmunshi/Downloads/videoplayback.mp4") # enter a file name

# Press command + Q to exit

while True:
    ret, frame = cap.read()

    #color = cv2.cvtColor(frame, cv2.COLOR_RGB2)

    cv2.imshow("Video Streaming", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


