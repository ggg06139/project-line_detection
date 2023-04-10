import cv2
import numpy as np

# 그레이스케일 이미지로 변환하는 함수
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 가우시안 필터링 함수
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 2)

# Canny edge detection 함수
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# Canny edge detection 임계값을 자동 설정하는 함수
def auto_canny(img, sigma):
    v = np.median(img)
    lower = int(max(0, (1.0-sigma)*v))
    upper = int(min(255, (1.0+sigma)*v))
    return cv2.Canny(img, lower, upper)