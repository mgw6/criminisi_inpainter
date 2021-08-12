#MacGregor Wineagrd, August 2021
#This file is a quick demonstration of the Criminisi Algorithm on a very simple test. 

import cv2 as cv
import numpy as np
import criminisi.criminisi as criminisi
from imageio import mimsave
from skimage.io import imread, imsave, imshow, show


orig_img = cv.imread(cv.samples.findFile('images/simple_sample.png'))


mask = np.zeros(orig_img.shape, dtype = int)
mask[np.where(orig_img == 255)] = 1
mask = mask[...,0]

result, movie = criminisi.Inpainting.inpaint(orig_img, mask, 21, return_movie = True)

cv.imshow("temp", result)
cv.waitKey(0)

mimsave("simple_filling.gif", movie, duration = .25) 
cv.imwrite("filling_result.png", result)