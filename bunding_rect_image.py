import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

src_img = None
th_img_3 = None

def nothing(x):
    pass

def showThresholdImage(*arg):
    global src_img, th_img_3

    f = cv2.getTrackbarPos("1. Threshold inverse\n0 : OFF\n1 : ON", "image")
    # src = cv2.imread(sys.argv[1], -1)
    print(f)
    img_t = createThrsholdImage(f)
    th_img_3 = cv2.cvtColor(img_t, cv2.COLOR_GRAY2RGBA)
    # kernel = np.ones((20,20), np.uint8)
    # dilation = cv2.dilate(img_t, kernel)
    # cv2.imshow("threshold", dilation)

def createThrsholdImage(img, isBinaryInv):
    global src_img

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    threshold_param = cv2.THRESH_OTSU
    if isBinaryInv:
        threshold_param += cv2.THRESH_BINARY_INV
    else:
        threshold_param += cv2.THRESH_BINARY
        
    ret, thshold_img = cv2.threshold(gray, 0, 255, threshold_param)

    return thshold_img

def drawContoursRectImage(img, contours, current_min_width, current_max_width, current_min_height, current_max_height):
    d_img = img.copy()

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # print(w, current_min_width, current_max_width)

        if (w >= current_min_width) and (w <= current_max_width) and h >= current_min_height and h <= current_max_height:
            # print('draw')
            cv2.rectangle(d_img, (x, y), (x + w, y + h), (255, 0, 0), 10)

    return d_img

def extractContours(img):
    pass

def getMinValMaxVlaTrackBar(src_img, contours):
    min_h, min_w = src_img.shape[:2]
    max_w = 0
    max_h = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if(w < min_w):
            min_w = w
        elif(w > max_w):
            max_w = w
        if(h < min_h):
            min_h = h
        elif(h > max_h):
            max_h = h
    return min_w, max_w, min_h, max_h

def createBoundingRectImage(img_path, threshold_img_path=None):
    global src_img, th_img_3

    imgName = "image"

    switchThreshold = "1. Threshold inverse\n0 : OFF\n1 : ON"

    labelMinValWidth = "2. minValWidth"
    labelMaxValWidth = "3. maxValWidth"
    labelMinValHeight = "4. minValHeight"
    labelMaxValHeight = '5. maxValHeight'

    # 元画像の読み込み
    src_img = cv2.imread(img_path, -1)
    src_img_h, src_img_w = src_img.shape[:2]
    src_img_min_h = 0
    src_img_min_w = 0
    # マスク画像の読み込み
    # if threshold_img_path is None:
    #     th_img = createThrsholdImage(src_img, True)
    # else:
    # th_img = cv2.imread(threshold_img_path, -1)
    # th_img_3 = cv2.cvtColor(th_img, cv2.COLOR_GRAY2RGBA)

    # マスク画像（二値画像）から特徴抽出
    # contours, hierachy = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # trackbar_min_w, trackbar_max_w, trackbar_min_h, trackbar_max_h = getMinValMaxVlaTrackBar(src_img, contours)
    prev_min_width = None
    prev_max_width = None
    prev_min_height = None
    prev_max_height = None

    prev_threshold_inv = None

    min_w = src_img_min_w
    max_w = src_img_w
    min_h = src_img_min_h
    max_h = src_img_h

    threshold_inv = 0

    cv2.namedWindow(imgName)

    cv2.createTrackbar(labelMinValWidth, imgName, src_img_min_w, src_img_w, nothing)
    # cv2.setTrackbarMin(labelMinValWidth, imgName, trackbar_min_w)
    # cv2.setTrackbarPos(labelMinValWidth, imgName, trackbar_min_w)

    cv2.createTrackbar(labelMaxValWidth, imgName, src_img_w, src_img_w, nothing)
    # cv2.setTrackbarMin(labelMinValWidth, imgName, trackbar_min_w)

    cv2.createTrackbar(labelMinValHeight, imgName, src_img_min_h, src_img_h, nothing)
    # cv2.setTrackbarMin(labelMinValHeight, imgName, trackbar_min_w)
    # cv2.setTrackbarPos(labelMinValHeight, imgName, trackbar_min_h)

    cv2.createTrackbar(labelMaxValHeight, imgName, src_img_h, src_img_h, nothing)
    # cv2.setTrackbarMin(labelMinValHeight, imgName, trackbar_min_w)

    cv2.createTrackbar(switchThreshold, imgName, 0, 1, createThrsholdImage)

    # d_img = drawContoursRectImage(src_img, contours, src_img_min_w, src_img_w, src_img_min_h, src_img_h)

    while(1):
        if(prev_threshold_inv != threshold_inv):
            th_img = createThrsholdImage(src_img, threshold_inv)
            th_img_3 = cv2.cvtColor(th_img, cv2.COLOR_GRAY2RGBA)
            prev_threshold_inv = threshold_inv

            prev_min_width = None
            prev_max_width = None
            prev_min_height = None
            prev_max_height = None

            min_w = src_img_min_w
            max_w = src_img_w
            min_h = src_img_min_h
            max_h = src_img_h

            contours, hierachy = cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if(min_w != prev_min_width) or (max_w != prev_max_width) or (min_h != prev_min_height) or (max_h != prev_max_height):
            d_img = drawContoursRectImage(src_img, contours, min_w, max_w, min_h, max_h)

            prev_min_width = min_w
            prev_max_width = max_w
            prev_min_height = min_h
            prev_max_height = max_h
    
        imgs = cv2.hconcat([d_img, th_img_3])
        # cv2.imshow(imgName, d_img)
        cv2.imshow(imgName, imgs)

        k = cv2.waitKey(0) & 0xff
        if(k == 27):
            break

        min_w = cv2.getTrackbarPos(labelMinValWidth, imgName)
        max_w = cv2.getTrackbarPos(labelMaxValWidth, imgName)
        min_h = cv2.getTrackbarPos(labelMinValHeight, imgName)
        max_h = cv2.getTrackbarPos(labelMaxValHeight, imgName)

        threshold_inv = cv2.getTrackbarPos(switchThreshold, imgName)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # if len(sys.argv) == 1:
    #     createBoundingRectImage(sys.argv[1])
    # elif len(sys.argv) == 2:
    createBoundingRectImage(sys.argv[1], sys.argv[2])
    # else:
    #     print('Invalid argument')

    # img = cv2.imread(sys.argv[1])
    # th = createThrsholdImage(img, True)
    # cv2.imshow('img', th)
    # cv2.waitKey(0) & 0xff
    # cv2.destroyAllWindows()