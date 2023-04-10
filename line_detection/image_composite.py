"""
작성자 정보: 황승현 / 전남대학교 전자컴퓨터공학부 전자정보통신공학전공 / ggg06139@gmail.com
코드 기능: 두 이미지를 합성하는 함수가 구현되어 있습니다.
최종 수정 시간: 2023년 4월 10일 20:55
"""

import cv2

def weighted_img(img, initial_img, a, b, theta_w):
    return cv2.addWeighted(initial_img, a, img, b, theta_w)
