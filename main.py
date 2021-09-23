import sys
import numpy as np
import cv2

def main():
    source_window = "source_image"
    gray_window = "gray"
    otsu_window = "otsu_threshold"
    edge_window = "edge"

    gray_img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    threshold1 = 100
    threshold2 = 100

    ret, otsu_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    cv2.imshow(otsu_window, otsu_img)

    edge_img = cv2.Canny(otsu_img, threshold1, threshold2)
    cv2.imshow(edge_window, edge_img)

    contours, hierachy = cv2.findContours(otsu_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    cv2.drawContours(gray_img, contours, -1, (0,255,0), 3)
    cv2.imshow(gray_window, gray_img)
    
    # cv2.imshow("test", findCon_img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()