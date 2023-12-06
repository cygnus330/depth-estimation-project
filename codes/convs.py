import cv2
import numpy as np

def binary(src, val=255/2):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(gray, 255/2, 255, cv2.THRESH_BINARY)

    return dst

def edge_sobel(src, dx=1, dy=0, dst=3, ksize=3):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # noinspection PyTypeChecker
    dst = cv2.Sobel(gray, cv2.CV_8U, dx, dy, dst)

    return dst

def edge_laplacian(src, ksize=3):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(gray, cv2.CV_8U, ksize=ksize)

    return dst

def edge_canny(src, threshold1=100, threshold2=255):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(gray, threshold1, threshold2)

    return dst