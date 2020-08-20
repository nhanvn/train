import numpy as np;
import cv2 as cv;
from backgroundSubract import createBackgroundSubtractMask;
from correctContour import correctContour;

capVideo = cv.VideoCapture(0);

ret, frameBegin = capVideo.read();
frameBegin = cv.flip(frameBegin, 1);

# variables to store history
initRectArea = (400, 100, 200, 200);
contourRectArea = None;

while (True):
  # Capture Video
  ret, frameOrigin = capVideo.read();
  frameOrigin = cv.flip(frameOrigin, 1);
  
  bgSubtractMask = createBackgroundSubtractMask(frameOrigin, frameBegin);

  # read more: https://docs.opencv.org/4.3.0/d4/d73/tutorial_py_contours_begin.html
  contours, hierarchy = cv.findContours(bgSubtractMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  # Get convex Hull
  hulls = [];
  for i in range(len(contours)):
    # read more: https://docs.opencv.org/4.3.0/dd/d49/tutorial_py_contour_features.html
    hull = cv.convexHull(contours[i], returnPoints=True);
    hulls.append(hull);
  
  # Area for detect hand
  hullsLen, hulls = correctContour(hulls, initRectArea, contourRectArea);
  if (hullsLen > 0):
    contourRectArea = cv.boundingRect(hulls[0]);
    # draw Green reactangle
    cv.rectangle(frameOrigin, (contourRectArea[0], contourRectArea[1]), (contourRectArea[0] + contourRectArea[2], contourRectArea[1] + contourRectArea[3]), (0, 255, 0), 1);
  else:
    contourRectArea = None;
    # draw Blue reactangle
    cv.rectangle(frameOrigin, (initRectArea[0], initRectArea[1]), (initRectArea[0] + initRectArea[2], initRectArea[1] + initRectArea[3]), 255, 1);
  
  # Create a ROI of hand contour
  contourROI = None;
  if (contourRectArea != None):
    contourROI = bgSubtractMask[contourRectArea[1] : contourRectArea[1] + contourRectArea[3], contourRectArea[0] : contourRectArea[0] + contourRectArea[2]];
  else:
    contourROI = bgSubtractMask[initRectArea[1] : initRectArea[1] + initRectArea[3], initRectArea[0] : initRectArea[0] + initRectArea[2]]

  # read more: https://docs.opencv.org/4.3.0/d4/d73/tutorial_py_contours_begin.html
  cv.drawContours(frameOrigin, hulls, -1, (255, 0, 0), 1, cv.LINE_AA);

  # show result
  cv.imshow("Video Capture", frameOrigin);
  cv.imshow("BgSubtract", bgSubtractMask);
  cv.imshow("contourROI", contourROI);

  # if user press q => exit
  key = cv.waitKey(100);
  if (key == ord("q")):
    break;

capVideo.release();
cv.destroyAllWindows();
hgjjgh
