# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2

# initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")
leftStream = VideoStream(src=0).start()
rightStream = VideoStream(src=1).start()
time.sleep(2.0)

# loop over frames from the video streams
while True:
    # grab the frames from their respective video streams
    left = leftStream.read()
    right = rightStream.read()
    cv2.imshow("Left Frame", left)
    cv2.imshow("Right Frame", right)


    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
leftStream.stop()
rightStream.stop()