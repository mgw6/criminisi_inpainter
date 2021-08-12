import numpy as np
import criminisi.criminisi as criminisi
from imageio import mimsave
import cv2 as cv

orig_img = cv.imread(cv.samples.findFile('images/simple_sample.png'))

mask = np.zeros(orig_img.shape, dtype = int)
mask[np.where(orig_img == 255)] = 1
mask = mask[...,0]

result, movie = criminisi.Inpainting.inpaint(orig_img, mask, 21, return_movie = True)

mimsave("simple_filling.gif", movie, duration = .25) 
cv.imwrite("filling_result.png", result) 
