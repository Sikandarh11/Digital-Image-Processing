import cv2
import numpy as np

path = "D:\\2. DIP\\2. LAB\\cc.png"
img = cv2.imread(path, 0)
#img = img.reshape(255,255)
row, col = img.shape
row_s = row//16
col_s = col//16
# for 15
level = 0
'''for i in range(16):
    for j in range(16):
        row_s = i*16
        row_e = (i+1)*16
        col_s = j*16
        col_end = (j+1)*16
        img[row_s:row_e,col_s:col_end]=level
        level += 15
        '''
for i in range(3):
    for j in range(3):
        row_s = i*64
        row_e = (i+1)*64
        col_s = j*64
        col_end = (j+1)*64
        if level > 255:
            img[row_s:row_e,col_s:col_end] = 255
        else:
            img[row_s:row_e,col_s:col_end]=level
        level += 40


cv2.imshow('win', img)
cv2.waitKey()