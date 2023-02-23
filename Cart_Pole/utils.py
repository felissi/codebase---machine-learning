import cv2 as cv
import numpy as np

WIDTH  = 240
HEIGHT = 160
def preprocess_image(image: np.ndarray, width=WIDTH, height=HEIGHT):
    _, gray = cv.threshold(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 200, 255,cv.THRESH_BINARY)
    # gray:np.ndarray = gray[(gray.shape[0]-WIDTH)//2:(gray.shape[0]-WIDTH)//2+WIDTH,(gray.shape[1]-HEIGHT)//2:(gray.shape[1]-HEIGHT)//2+HEIGHT]
    gray = cv.resize(gray, (width, height))
    return gray