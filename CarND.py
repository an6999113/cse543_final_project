#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """Applies an image mask."""
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """Draws lines on the image"""
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def slope_lines(image, lines):
    img = image.copy()
    poly_vertices = []
    order = [0,1,3,2]
    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
            if m < 0:
                left_lines.append((m, c))
            else:
                right_lines.append((m, c))

    left_line = np.mean(left_lines, axis=0) if left_lines else (0, 0)
    right_line = np.mean(right_lines, axis=0) if right_lines else (0, 0)

    for slope, intercept in [left_line, right_line]:
        rows, cols = image.shape[:2]
        y1 = int(rows)
        y2 = int(rows * 0.6)
        x1 = int((y1 - intercept) / slope) if slope != 0 else cols // 2
        x2 = int((y2 - intercept) / slope) if slope != 0 else cols // 2
        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))
        draw_lines(img, np.array([[[x1, y1, x2, y2]]]))

    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(img, pts=np.array([poly_vertices], 'int32'), color=(0, 255, 0))
    return cv2.addWeighted(image, 0.7, img, 0.4, 0.)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return slope_lines(line_img, lines)

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows]
    top_right = [cols * 0.6, rows * 0.6]
    return np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

def lane_finding_pipeline(image):
    gray_img = grayscale(image)
    smoothed_img = gaussian_blur(img=gray_img, kernel_size=5)
    canny_img = canny(img=smoothed_img, low_threshold=50, high_threshold=150)
    masked_img = region_of_interest(img=canny_img, vertices=get_vertices(image))
    houghed_lines = hough_lines(img=masked_img, rho=1, theta=np.pi / 180, threshold=30, min_line_len=15, max_line_gap=50)
    return weighted_img(img=houghed_lines, initial_img=image, α=0.8, β=1., γ=0.)
