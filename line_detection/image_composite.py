import cv2

def weighted_img(img, initial_img, a, b, theta_w):
    return cv2.addWeighted(initial_img, a, img, b, theta_w)