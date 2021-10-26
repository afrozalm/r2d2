import cv2
import numpy as np

sv = np.load('./imgs/boat.png.r2d2')
kpts = sv['keypoints']
dpts = sv['descriptors']
sz = sv['imsize']

img = cv2.imread('./imgs/boat.png', 0)
#img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

kpts = [cv2.KeyPoint(k[0], k[1], size=k[2]) for k in kpts]

kpt_img = cv2.drawKeypoints(img, kpts,outImage=np.array([]))

cv2.imwrite("imgs/boat-kpts.png", kpt_img)
