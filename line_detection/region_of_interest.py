"""
작성자 정보: 황승현 / 전남대학교 전자컴퓨터공학부 전자정보통신공학전공 / ggg06139@gmail.com
코드 기능: ROI(Region of Interest)를 설정하는 함수가 구현되어 있습니다.
최종 수정 시간: 2023년 4월 10일 20:56
"""

import cv2
import numpy as np

# ROI 마스크를 생성하는 함수
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
