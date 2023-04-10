import cv2
import numpy as np
from line_detection import *

# 동영상 파일 경로 지정
path = "./source/highway.mp4"

# 동영상 파일 열기
cap = cv2.VideoCapture(path)

# 동영상 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# 동영상 저장을 위한 설정
codec = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('result.mp4', codec,
                      30.0, (int(width), int(height)))

# Canny edge detection 알고리즘에 필요한 변수
kernel_size = 5
low_threshold = 50
high_threshold = 200

# Hough Transform 알고리즘에 필요한 변수
rho = 2
theta = np.pi/180
threshold = 90
min_line_len = 100
max_line_gap = 50

# 이미지 합성에 대한 변수
a = 0.8
b = 1
theta_w = 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # 이미지 전처리
    gray = grayscale(img)
    blur_gray = gaussian_blur(gray, kernel_size)
    edges = canny(blur_gray, 15, 100)
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    imshape = img.shape

    # ROI 설정
    vertices = np.array(
        [[(680, 530), (815, 530), (1060, 720), (390, 720)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = region_of_interest(edges, vertices)

    # 직선 검출
    lines = hough_lines(masked_image, rho, theta,
                        threshold, min_line_len, max_line_gap)

    # 검출된 직선과 원본 이미지 합성
    lines_edges = weighted_img(lines, img, a, b, theta)

    cv2.imshow('frame1', lines_edges)
    out.write(lines_edges)

    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
