import cv2
import numpy as np
import matplotlib.pyplot as plt
from autoKeyPoints import moveDetectionByKeyPoints
from manualKeyPoints import manualKeyPoints

imagePath1 = "../images/kuiguangta/image1.jpg"
imagePath2 = "../images/kuiguangta/image3.jpg"

image1 = cv2.imread(imagePath1)
image2 = cv2.imread(imagePath2)

moveDetectionByKeyPoints(image1,image2)