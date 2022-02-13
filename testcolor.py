import cv2
import numpy as np
from components import image


def empty(a):
    pass


cv2.namedWindow("TarckBars")
cv2.resizeWindow("TarckBars", 640, 240)
cv2.createTrackbar("Hue Min", "TarckBars", 0, 180, empty)
cv2.createTrackbar("Hue Max", "TarckBars", 180, 180, empty)
cv2.createTrackbar("Sat Min", "TarckBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TarckBars", 255, 255, empty)
cv2.createTrackbar("Lig Min", "TarckBars", 0, 255, empty)
cv2.createTrackbar("Lig Max", "TarckBars", 255, 255, empty)

img = cv2.imread("./image/result/all.jpg")
# 将图片的颜色空间转换到HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
while True:
    h_min = cv2.getTrackbarPos("Hue Min", "TarckBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TarckBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TarckBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TarckBars")
    v_min = cv2.getTrackbarPos("Lig Min", "TarckBars")
    v_max = cv2.getTrackbarPos("Lig Max", "TarckBars")
    # 颜色区间 [lower ~ upper]
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    # 分割出颜色区间
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)
    # 显示
    cv2.imshow('img', img)
    cv2.imshow("img_mask", mask)
    cv2.imshow("img_res", res)
    cv2.waitKey(1)
