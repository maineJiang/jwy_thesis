import cv2
import numpy as np
img = cv2.imread('ours/mask_0.png')
edge = cv2.Canny(img,50,100)
# edgd = cv2.rectangle(edge,(330,0),(368,30),(255,0,0),thickness=3)
a = 1
O = img * float(a)
O[O>255] =255
O = np.round(O)
O = O.astype(np.uint8)
# range_img = img.max()-img.min()
# img = (img-img.min())
# img = img/range_img
# # img = img*255
# cv2.imshow("img",edge)
# cv2.waitKey(0)
cv2.imwrite('ours/mask_0_new.png',edge)
print(' ')