#August 10, 2021
#Program by MacGregor Winegard
#Implementation of the algorithm proposed by Criminisi et al. (2004)

import cv2 as cv
import numpy as np
from scipy.signal import convolve2d 
from scipy.ndimage.filters import convolve
import time


class TVA: #TIME VARIANCE AUTHORITY
    #Matlab tic and toc functions, inspired by Stack Overflow
    def tic():
        startTime_for_tictoc = time.time()
        print("Timer started at: " + time.strftime("%H:%M:%S"))
        return startTime_for_tictoc

    def toc(tic_time):
        elapsed_time = f"{(time.time() - tic_time)//60} minutes, " + \
        f"{((time.time() - tic_time)%60):5.1f} seconds."
        return elapsed_time   

class inpainting: 
    def get_patch_slice(point, psz): #TODO: account for edge cases
        half_psz = psz//2
        return (slice(point[0] - half_psz, point[0] + half_psz+1),
            slice(point[1] - half_psz, point[1] + half_psz+1))
    
    def get_patch_list(point, psz): #TODO: account for edge cases
        halfPatch = psz//2 
        rows = np.arange( (point[0] - halfPatch), (point[0] + halfPatch+1))
        cols = np.transpose(np.arange( (point[1] - halfPatch), (point[1] + halfPatch+1)))
        rows = (rows * np.ones((psz, psz))).astype(int)
        cols = (np.transpose(cols *np.ones((psz, psz)))).astype(int) 
        return [rows, cols]
    
    def get_boundary(mask):
        laplacian = [[1,1,1], [1,-8,1], [1,1,1]] #Laplacian Matrix
        double_mask = np.double(mask)
        conv2dHolder = np.rot90(convolve2d(np.rot90(double_mask, 2), np.rot90(laplacian, 2), mode='same'), 2) 
        return list(map(tuple, np.argwhere(conv2dHolder>0)))
    
    def get_data(image, mask, boundary, psz):
        #norm of the mask
        row_sobel = np.array([[1, 0, -1], [1, 0, -2], [1, 0, -1]])
        col_sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        norm_mask = np.array(mask, dtype = float)
        
        normal = np.array([
            convolve(norm_mask, row_sobel), 
            convolve(norm_mask, col_sobel)
        ])
        normal_bottom = np.sqrt(
            normal[0]**2 + normal[1]**2
        )
        normal_bottom[normal_bottom==0] = 1
        
        norm_list = []
        for point in boundary:
            norm_list.append([
                (normal[0][point]/normal_bottom[point]),
                (normal[1][point]/normal_bottom[point]), 
            ])
        
        flat_image = np.copy(image).astype(float)
        if flat_image.ndim == 3:
            flat_image = np.sum(flat_image, axis = 2)
        flat_image[mask] = None
        
        
        Ix, Iy = np.nan_to_num(np.array(np.gradient(flat_image)))
        total_gradient = np.sqrt(Ix**2 + Iy**2)
        gradient_list = []
        for point in boundary:
            patch = inpainting.get_patch_slice(point, psz)
            row_gradient = Ix[patch]
            col_gradient = Iy[patch]
            patch_total_gradient = total_gradient[patch]

            max_gradient = np.unravel_index(
                patch_total_gradient.argmax(),
                patch_total_gradient.shape
            )

            gradient_list.append([  
                row_gradient[max_gradient],
                col_gradient[max_gradient]
            ])
        
        data = (np.array(gradient_list)*np.array(norm_list))**2
        return np.sqrt(np.sum(data, axis = 1)) + .001
    
    
    
    
    def get_confidence(confidence, boundary, psz):
        confidence_list = []
        for point in boundary:
            confidence_list.append(
                np.sum(confidence[inpainting.get_patch_slice(point, psz)])
            )
        return np.array(confidence_list) 
    
    
    def bestexemplar(working_image, to_fill, target_pixel, psz):
        
        patchErr = 0.0
        bestErr = 1000000000.0 
        best = None
        source_region = ~to_fill
        
        half_psz = psz//2
        
        target_patch = inpainting.get_patch_slice(target_pixel, psz)
        target_patch_data = working_image[target_patch]
        target_patch_known = source_region[target_patch]
        
        for row in range(half_psz, working_image.shape[0]-half_psz-1):
            for col in range(half_psz, working_image.shape[1]-half_psz-1):
                source_patch = inpainting.get_patch_slice((row,col), psz)
                if not source_region[source_patch].all():
                    continue
                
                SSD = np.sum(
                    ((target_patch_data - working_image[source_patch])[target_patch_known] ) **2
                )
                if(SSD < bestErr):
                        bestErr = SSD
                        best = (row, col)
        return inpainting.get_patch_list(best, psz)

    
    """ Copied from the matlab code:
    % Inputs: 
    %   - original_image: original image or corrupted image
    %   - mask:           implies target region (1 denotes target region) 
    %   - psz:           patch size (odd scalar). If psz=5, patch size is 5x5.
    %
    % Outputs:
    %   - working_image   The inpainted image; an MxNx3 matrix of doubles. 
    """   
    def inpainting(original_image, mask, psz):
        if psz % 2 == 0:
            raise Exception("Patch size psz must be odd")
        if not np.any(original_image):
            raise Exception("Original image invalid")
        if not np.any(mask):
            raise Exception("Mask invalid")  
        if np.setdiff1d(mask, [0,1]).size:
            raise Exception("Mask should only have 0's and 1's.")
        if original_image.shape[:2] != mask.shape:
            raise Exception("Mask and image should be the same shape.")
        
        start_time = TVA.tic()
        
        working_image = np.copy(original_image)
        to_fill = ~np.logical_not(mask)
        confidence = np.array(~to_fill, dtype = float)
        
        
        iter_count = 0
        while True in to_fill:
        
            iter_count +=1
            print("\nStarting iteration " + str(iter_count))
            cv.imshow('Working image', working_image)
            cv.waitKey(5)
            
            boundary_list = inpainting.get_boundary(to_fill)
            
            data_list = inpainting.get_data(working_image, to_fill, boundary_list, psz)
            confidence_list = inpainting.get_confidence(confidence, boundary_list, psz)
            #print(np.unique(data_list))
            highest_priority = np.argmax(confidence_list*data_list)
            #exit()
            
            target_pixel = boundary_list[highest_priority]
            target_patch = inpainting.get_patch_list(target_pixel, psz)
            target_patch_known = to_fill[tuple(target_patch)]
            
            source_patch = inpainting.bestexemplar(working_image, to_fill, target_pixel, psz)
            
            #update fill region
            to_fill[tuple(target_patch)] = False
            
            #Update confidences
            confidence[target_patch[0][target_patch_known], target_patch[1][target_patch_known]]  = \
                confidence[target_pixel]
            
            #Copy image data from source to target  
            working_image[target_patch[0][target_patch_known], target_patch[1][target_patch_known]] = \
                working_image[source_patch[0][target_patch_known], source_patch[1][target_patch_known]]
            
        print("Time: " + TVA.toc(start_time))
        return working_image