# import the necessary packages
from __future__ import print_function
from panorama import Stitcher
from imutils.video import VideoStream
import datetime
import imutils
import time
import cv2

# initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")
leftStream = VideoStream(src=0).start()
rightStream = VideoStream(src=1).start()
time.sleep(2.0)


# initialize the image stitcher, motion detector, and total
# number of frames read
stitcher = Stitcher()
total = 0

# loop over frames from the video streams
while True:
    # grab the frames from their respective video streams
    left = leftStream.read()
    right = rightStream.read()

    # resize the frames
    left = imutils.resize(left, width=640, height=480)
    right = imutils.resize(right, width=640, height=480)

    # stitch the frames together to form the panorama
    # IMPORTANT: you might have to change this line of code
    # depending on how your cameras are oriented; frames
    # should be supplied in left-to-right order
    result = stitcher.stitch([left, right])

    # no homograpy could be computed
    if result is None:
        print("[INFO] homography could not be computed")
        break



    # increment the total number of frames read and draw the
    # timestamp on the image
    total += 1
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(result, ts, (10, result.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the output images
    cv2.imshow("Result", result)
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