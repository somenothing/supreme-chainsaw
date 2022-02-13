import cv2
import numpy as np


def resize(image, scale):
    """
    调整图像大小，按比例缩放
    :param image: cv2图像对象
    :param scale: 缩放比例
    :return: 缩放后的图像
    """
    size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return cv2.resize(image, dsize=size)


def extract_table(target_image, empty_image):
    """
    提取球桌边缘位置
    :param target_image: 目标图像
    :param empty_image: 空桌面图像
    :return: 球桌边缘的坐标
    """
    target_size = target_image.shape[:2][::-1]  # 目标图像大小，转换为(宽度, 高度)
    empty_image = cv2.resize(empty_image, target_size)  # 空桌面大小自动缩放

    # hsv范围
    hsv_min = np.array([70, 150, 0])
    hsv_max = np.array([90, 255, 255])

    image_hsv = cv2.cvtColor(empty_image, cv2.COLOR_BGR2HSV)  # 图片转换为hsv
    mask = cv2.inRange(image_hsv, hsv_min, hsv_max)
    table_image = cv2.bitwise_and(empty_image, empty_image, mask=mask)
    table_blurred = cv2.GaussianBlur(table_image, (15, 15), 0)  # 模糊处理
    table_canny = cv2.Canny(table_blurred, 50, 50)  # 边缘提取

    lines = cv2.HoughLinesP(table_canny, 1.0, np.pi/180, 100, minLineLength=500, maxLineGap=60)  # 霍夫直线检测

    result = [None, None, None, None]  # 顺序：x1, x2, y1, y2
    for line in lines:
        for x1, y1, x2, y2 in line:
            # cv2.line(target_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if abs(x1 - x2) < 2 and x1 + x2 < target_size[0]:  # 左侧竖直线
                result[0] = int((x1 + x2) / 2) if result[0] is None or int((x1 + x2) / 2) > result[0] else result[0]
            if abs(x1 - x2) < 2 and x1 + x2 > target_size[0]:  # 右侧竖直线
                result[1] = int((x1 + x2) / 2) if result[1] is None or int((x1 + x2) / 2) < result[1] else result[1]
            if abs(y1 - y2) < 2 and y1 + y2 < target_size[1]:  # 上方水平线
                result[2] = int((y1 + y2) / 2) if result[2] is None or int((y1 + y2) / 2) > result[2] else result[2]
            if abs(y1 - y2) < 2 and y1 + y2 > target_size[1]:  # 下方水平线
                result[3] = int((y1 + y2) / 2) if result[3] is None or int((y1 + y2) / 2) < result[3] else result[3]
    # cv2.line(target_image, (result[0], result[2]), (result[1], result[3]), (0, 0, 255), 2)
    if None in result:
        print('[x错误x] 球桌边缘检测失败：%s\n位于./components/image/image.py' % result)
    else:
        print('球桌边缘检测：', result)
    return result


def extract_ball(target_image, empty_image):
    target_size = target_image.shape[:2][::-1]  # 目标图形大小
    empty_image = cv2.resize(empty_image, target_size)  # 将空桌面大小自动缩放

    subtract_image = cv2.subtract(target_image, empty_image)  # 相减
    gray_image = cv2.cvtColor(subtract_image, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    ret, thresh = cv2.threshold(gray_image, 5, 255, cv2.THRESH_BINARY_INV)  # 二值化
    cv2.imshow('thresh', thresh)

    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 10, param1=20, param2=12, minRadius=20, maxRadius=30)
    print('共找到%s个' % len(circles[0]))
    for circle in circles[0]:
        x, y, r = [int(i) for i in circle]
        cv2.circle(target_image, (x, y), r, (0, 0, 255), -1)
    return target_image


def border(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    blurred_image = cv2.GaussianBlur(gray_image, (17, 17), 0)  # 模糊处理
    contour_image = cv2.Canny(blurred_image, 50, 50)  # 边缘提取
    return contour_image


def draw_circle(target, mark=None):
    mark = target if mark is None else mark
    circles = cv2.HoughCircles(target, cv2.HOUGH_GRADIENT, 1, 1, param1=20, param2=15, minRadius=5, maxRadius=20)
    print('共找到%s个' % len(circles[0]))
    for circle in circles[0]:
        x, y, r = [int(i) for i in circle]
        cv2.circle(mark, (x, y), r, (0, 0, 255), -1)
