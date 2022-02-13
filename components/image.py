import copy
import cv2
import numpy as np

hsv_dict = {
    '0': (np.array([0, 0, 100]), np.array([180, 30, 255])),
    '1': (np.array([20, 50, 80]), np.array([40, 255, 245])),
    '2': (np.array([100, 50, 120]), np.array([125, 255, 255])),
    '3': (np.array([160, 100, 100]), np.array([180, 255, 255])),
    '4': (np.array([125, 45, 50]), np.array([155, 255, 255])),
    '5': (np.array([10, 90, 100]), np.array([15, 255, 255])),
    '6': (np.array([50, 50, 50]), np.array([70, 255, 255])),
    '7': (np.array([2, 150, 80]), np.array([9, 255, 255])),
    '8': (np.array([0, 0, 15]), np.array([180, 255, 60])),
}


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

    # 图片预处理
    image_hsv = cv2.cvtColor(empty_image, cv2.COLOR_BGR2HSV)  # 图片转换为hsv
    mask = cv2.inRange(image_hsv, hsv_min, hsv_max)  # 设置阈值
    table_image = cv2.bitwise_and(empty_image, empty_image, mask=mask)  # 去除背景部分
    table_blurred = cv2.GaussianBlur(table_image, (15, 15), 0)  # 模糊处理
    table_canny = cv2.Canny(table_blurred, 50, 50)  # 边缘提取

    lines = cv2.HoughLinesP(table_canny, 1.0, np.pi / 180, 100, minLineLength=500, maxLineGap=60)  # 霍夫直线检测

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
    # 如检测正常，下方语句可显示自左上至右下的对角线
    # cv2.line(target_image, (result[0], result[2]), (result[1], result[3]), (0, 0, 255), 2)
    if None in result:
        print('[x错误x] 球桌边缘检测失败：%s\n位于./components/image.py' % result)
        return None
    else:
        print('球桌边缘检测：', result)
        return result


def extract_ball(target_image, empty_image, table_range=None):
    target_size = target_image.shape[:2][::-1]  # 目标图形大小，转换为(宽度, 高度)
    empty_image = cv2.resize(empty_image, target_size)  # 将空桌面大小自动缩放

    # 图片预处理
    subtract_image = cv2.subtract(target_image, empty_image)  # 相减
    gray_image = cv2.cvtColor(subtract_image, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    ret, thresh = cv2.threshold(gray_image, 5, 255, cv2.THRESH_BINARY_INV)  # 二值化

    # 霍夫圆形检测
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 8, param1=20, param2=12, minRadius=20, maxRadius=30)

    if circles is None:
        print('[x错误x] 台球第一次检测失败：%s\n位于./components/image.py' % circles)
        return None

    # 圆形过滤
    draw_image = copy.deepcopy(target_image)
    if table_range is None:  # 未提供过滤参数
        for circle in circles[0]:
            x, y, r = [int(i) for i in circle]
            cv2.circle(draw_image, (x, y), r, (0, 0, 255), -1)
        print('台球检测：共找到%s个' % len(circles[0]))
        print('[!警告!] 不提供参考范围参数仅用于调试，考虑给函数提供参考范围参数以提高准确度！\n位于./components/image.py')
        return None
    else:
        filter_num = 0
        filter_results = []
        for circle in circles[0]:
            if table_range[0] < circle[0] < table_range[1] and table_range[2] < circle[1] < table_range[3]:  # 圆形在球桌范围内
                filter_num += 1
                x, y, r = [int(i) for i in circle]
                filter_results.append([x, y, r])
                cv2.circle(draw_image, (x, y), r, (0, 0, 255), -1)
        print('台球检测：共找到%s个' % filter_num)
        print(filter_results)

    # 精细圆形检测
    radius = int(np.ceil(np.mean([result[2] for result in filter_results])))  # 向上取整半径平均值

    print('第二次台球检测：')
    ball_result = {}
    for result in filter_results:
        # 裁剪图像
        border = int(np.ceil(radius * 1.5))
        x1, x2, y1, y2 = result[0] - border, result[0] + border, result[1] - border, result[1] + border
        ball_cropped = target_image[y1:y2, x1:x2]
        empty_cropped = empty_image[y1:y2, x1:x2]

        # 预处理
        subtract_ball_image = cv2.subtract(ball_cropped, empty_cropped)  # 相减
        gray_ball_image = cv2.cvtColor(subtract_ball_image, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
        ret_ball, thresh_ball = cv2.threshold(gray_ball_image, 8, 255, cv2.THRESH_BINARY_INV)  # 二值化

        # 霍夫圆形检测
        circles = cv2.HoughCircles(thresh_ball, cv2.HOUGH_GRADIENT, 1, 10,
                                   param1=20, param2=10, minRadius=20, maxRadius=30)

        if circles is None:
            print('[?提示?] 第二次检测存在区域无有效圆形？\n位于./components/image.py')
            continue

        if len(circles[0]) == 1:  # 只检测到一个圆形
            x, y, r = [int(i) for i in circles[0][0]]
            position_filtered = [x, y, r]
            # cv2.circle(ball_cropped, (x, y), r, (0, 0, 255), 5)
        else:  # 检测到多个圆形
            ball_possible = [0, radius * 2]  # 可能值，顺序为[序号, 距图中心距离]
            # 遍历圆形，寻找距中心最近的圆
            # todo: 此段代码可能存在缺陷，可能返回最外侧的圆形而不是中间的圆形
            for index, circle in enumerate(circles[0]):
                x, y, r = [int(i) for i in circle]
                horizontal_distance, vertical_distance = abs(x - radius), abs(y - radius)
                distance = np.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)
                if distance < ball_possible[1]:
                    ball_possible = [index, distance]
            x, y, r = [int(i) for i in circles[0][ball_possible[0]]]
            position_filtered = [x, y, r]

        # 裁剪出每个球的圆形图像
        mask_circle = np.zeros(ball_cropped.shape[:2], np.uint8)  # 新建蒙版
        mask_circle = cv2.circle(mask_circle, (x, y), r, (255, 255, 255), -1)  # 设置最大内接圆为不透明
        ball_image = cv2.bitwise_and(ball_cropped, ball_cropped, mask=mask_circle)

        # 对每个球进行进一步检测判断球号
        ball_hsv = cv2.cvtColor(ball_image, cv2.COLOR_BGR2HSV)  # 图片转换为hsv
        cv2.imshow('ball', ball_image)
        color_result = {}
        ball_area = np.pi * (r ** 2)
        for num, color in hsv_dict.items():
            mask = cv2.inRange(ball_hsv, color[0], color[1])  # 设置阈值
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
            area = 0
            for contour in contours:
                area += cv2.contourArea(contour)  # 面积累加
            if area > 0.1 * ball_area and color != '0':  # 过滤小面积误差
                color_result[num] = area
            elif area > 0.5 * ball_area and color == '0':
                color_result[num] = area
        color_result = sorted(color_result.items(), key=lambda kv: (kv[1], kv[0]))  # 排序，取面积最大的两个
        if len(color_result) > 1:
            if color_result[1][1] > color_result[0][1] * 5:  # 如果大的面积远大于小的面积，则为主要颜色
                color_result = [color_result[1]]
        color_result = [result[0] for result in color_result]
        print(color_result)

        # 合并结果
        position_trans = np.array(position_filtered) + np.array([x1, y1, 0])  # 将局部坐标转换为全局坐标
        position_trans = position_trans.tolist()
        if len(color_result) == 1:  # 全色球
            ball_result[color_result[0]] = position_trans
        elif len(color_result) == 2 and '0' in color_result and '8' not in color_result:  # 花色球
            color_result = [str(int(c) + 8) for c in color_result if c != '0']  # 计算球号
            ball_result[color_result[0]] = position_trans
        print(ball_result)
        cv2.waitKey(0)

    return draw_image
