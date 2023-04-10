"""
작성자 정보: 황승현 / 전남대학교 전자컴퓨터공학부 전자정보통신공학전공 / ggg06139@gmail.com
코드 기능: 직선 검출에 관한 함수들이 구현되어 있습니다.
최종 수정 시간: 2023년 4월 10일 20:54
"""


import cv2
import numpy as np

# 선을 그리는 함수
def draw_lines(img, lines, color=[0, 0, 255], thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Hough Transform 함수
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold,
                            np.array([]), min_line_len, max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
