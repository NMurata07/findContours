import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_transparent_img(img_path):
    filename = os.path.splitext(img_path)[0]
    dirname = filename
    data_dirname = os.path.join(dirname, "data")
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    if not os.path.exists(data_dirname):
        os.mkdir(data_dirname)
    
    img = cv2.imread(img_path, -1)

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    ret, otsu_inv_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    ret, otsu_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # cv2.imshow(otsu_window, otsu_inv_img)
    mask_file_path = os.path.join(data_dirname, filename + '_mask.png')
    mask_inv_file_path = os.path.join(data_dirname, filename + '_mask_inv.png')
    cv2.imwrite(mask_file_path, otsu_inv_img)
    cv2.imwrite(mask_inv_file_path, otsu_img)


    im_a1 = Image.open(mask_file_path).convert('L')
    im_a2 = Image.open(mask_inv_file_path).convert('L')
    im_rgba1 = Image.open(img_path)
    im_rgba2 = Image.open(img_path)
    im_rgba1.putalpha(im_a1)
    im_rgba2.putalpha(im_a2)

    img_file_path = os.path.join(dirname, filename + ".png")
    img_inv_file_path = os.path.join(dirname, filename + "_inv.png")
    im_rgba1.save(img_file_path)
    im_rgba2.save(img_inv_file_path)

create_transparent_img(sys.argv[1])