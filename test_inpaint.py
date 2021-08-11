#MacGregor Wineagrd, August 2021
#This file is a quick demonstration of the Criminisi Algorithm on a very simple test. 

import cv2 as cv
import numpy as np
import criminisi

orig_img = cv.imread(cv.samples.findFile('simple_sample.png'))

mask = np.zeros(orig_img.shape, dtype = int)
mask[np.where(orig_img == 255)] = 1
mask = mask[...,0]


result = criminisi.inpainting.inpainting(orig_img, mask, 25)

cv.imshow("temp", result)
cv.waitKey(0)
