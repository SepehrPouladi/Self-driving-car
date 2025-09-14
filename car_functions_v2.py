from math import hypot
from typing import Sequence

import cv2 as cv
import numpy as np

# Color of road lines
YELLOW_LINE_COLOR = [22, 30, 102, 255, 122, 255]
WHITE_LINE_COLOR = [0, 179, 0, 49, 184, 255]

LOWER_YELLOW = np.array(
    [YELLOW_LINE_COLOR[0], YELLOW_LINE_COLOR[2], YELLOW_LINE_COLOR[4]])
UPPER_YELLOW = np.array(
    [YELLOW_LINE_COLOR[1], YELLOW_LINE_COLOR[3], YELLOW_LINE_COLOR[5]])

LOWER_WHITE = np.array(
    [WHITE_LINE_COLOR[0], WHITE_LINE_COLOR[2], WHITE_LINE_COLOR[4]])
UPPER_WHITE = np.array(
    [WHITE_LINE_COLOR[1], WHITE_LINE_COLOR[3], WHITE_LINE_COLOR[5]])

# Shape of input image is 512 x 512
W, H = 512, 512

# Centers of car
CAR_CENTER = (260, 400)
CAR_CENTER_WARP_FRAME = (250, 210)

# Race points
top_left = (160, 230)
top_right = (352, 230)
bottom_right = (W - 30, H - 120)
bottom_left = (60, H - 120)

# Contours
CONTOUR_MIN_SIZE = 250  # todo: find the best
APPROX_MAX_SIZE = 16


def warp_frame(frame):
    src_points = np.float32(
        [[top_left], [top_right], [bottom_right], [bottom_left]])
    dst_points = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    matrix = cv.getPerspectiveTransform(src_points, dst_points)
    warped_frame = cv.warpPerspective(frame, matrix, (W, H))
    return warped_frame


def create_mask(warped_frame, low, up):
    img_hsv = cv.cvtColor(warped_frame, cv.COLOR_BGR2HSV)
    return cv.inRange(img_hsv, low, up)


def find_line(frame, mask):
    white_image = np.ones_like(frame, dtype=np.uint8) * 255

    # stop_line: bool = False todo: Working on stop line
    line_center_x = None


    # Find best results of contours:
    contours = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
    if len(contours) != 0:
        # white_image = cv.drawContours(white_image, contours, -1, (0, 0, 0), 1)
        contours_distances_list = find_contours_distances(contours)

        # Find the biggest area of contours
        sorted_distances = sorted(
            contours_distances_list, key=lambda x: x[-1])

        if len(sorted_distances) > 0:
            # Define 2 biggest contours
            first_contour = sorted_distances[0][0]

            # Draw contours
            white_image = cv.drawContours(
                white_image, [first_contour], 0, (255, 0, 255), -1)

            # Define moments and cx, cy for each contour
            # todo: remove this part. (it is just for visualizing)

            line_center_x = sorted_distances[0][1][0]
    return white_image, line_center_x  # , stop_line


def find_contours_distances(contours):
    """
    returns: A list of tuples containing each contour with its center point and the contour distance form the center of car
    """
    distances_of_contours = []

    for contour in contours:
        if cv.contourArea(contour) > CONTOUR_MIN_SIZE:
            epsilon = 0.01 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            if len(approx) < APPROX_MAX_SIZE:
                moments = cv.moments(contour)
                if moments['m00'] != 0:
                    contour_cx, contour_cy = int(moments['m10'] / moments['m00']), int(
                        moments['m01'] / moments['m00'])

                    distance_from_car_center = find_distance_from_car_center(
                        (contour_cx, contour_cy))

                    distances_of_contours.append((
                        contour,
                        (contour_cx, contour_cy),
                        distance_from_car_center
                    ))

    return distances_of_contours


def find_distance_from_car_center(point):
    x1, y1 = CAR_CENTER
    x2, y2 = point
    return hypot(x2 - x1, y2 - y1)


def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def calc_steering(frame):
    warped_frame = warp_frame(frame)
    yellow_mask = create_mask(warped_frame, LOWER_YELLOW, UPPER_YELLOW)
    white_mask = create_mask(warped_frame, LOWER_WHITE, UPPER_WHITE)
    yellow_img, yellow_center_x = find_line(frame, yellow_mask)
    white_img, white_img_center_x = find_line(frame, white_mask)
    
    if yellow_center_x is not None and white_img_center_x is not None:
        avg = (yellow_center_x + white_img_center_x) // 2
    else:
        avg = 0

    result = np.zeros(yellow_img.shape)
    
    result += yellow_img
    result += white_img

    result /= 2
    result = result.astype(np.uint8)

    return result, avg


if __name__ == '__main__':
    frame = cv.imread('../assets/frame_screenshot_11.07.2024.png')
    img, avg = calc_steering(frame)

    print(f'{avg = }')
    cv.imshow('img', img)
    cv.waitKey(0)
