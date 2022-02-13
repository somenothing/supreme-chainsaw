import cv2
from components import image
from components.physics import Table

empty = cv2.imread(r"D:\Code\BilliardsAI\image\empty.jpg")
target = cv2.imread(r"D:\Code\BilliardsAI\image\special.jpg")

x = image.extract_table(target, empty)
r = image.extract_ball(target, empty, [437, 1898, 237, 970])

cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)
cv2.imshow('result', r)
cv2.waitKey(0)
cv2.destroyAllWindows()
