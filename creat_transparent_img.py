import sys
import cv2
import numpy as np

def create_transparent_img(img_path):
    source_window = "source_image"
    gray_window = "gray"
    otsu_window = "otsu_threshold"
    edge_window = "edge"

    img = cv2.imread(img_path, -1)
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    # cv2.imshow("rgb", rgba)

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # cv2.imshow("img", img)
    # cv2.imshow("gray", gray_img)

    # color_lower = np.array([255, 0, 0, 255])
    # color_upper = np.array([255, 0, 0, 255])

    # img_mask = cv2.inRange(img, color_lower, color_upper)
    # img_bool = cv2.bitwise_not(img, img, mask=img_mask)

    ret, otsu_inv_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    # cv2.imshow(otsu_window, otsu_inv_img)

    threshold1 = 100
    threshold2 = 100
    edge_img = cv2.Canny(otsu_inv_img, threshold1, threshold2)
    # cv2.imshow(edge_window, edge_img)

    mask = otsu_inv_img


    contours, hierachy = cv2.findContours(otsu_inv_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(otsu_inv_img)
    mask_a = cv2.cvtColor(otsu_inv_img, cv2.COLOR_GRAY2RGBA)
    # cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
    # cv2.cvtColor(mask_a, cv2.COLOR_BGR2RGBA)
    cv2.imshow("a", mask_a)
    # cv2.drawContours(mask, contours, -1, color=255, thickness=-1)

    # rgba[..., 3] = np.where(np.all(img == 255, axis=-1), 0, 255)
    # cv2.imshow("mask", rgba)

    # cv2.imwrite("out.png", img_bool)
    rgba[..., 3] = np.where(np.all(mask_a == 0), 0, 255)
    cv2.imshow("rgb", rgba)
    cv2.imwrite("out.png", rgba)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

create_transparent_img(sys.argv[1])