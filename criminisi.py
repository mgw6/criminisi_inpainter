#June 28nd, 2021
#Program MacGregor Winegard
#This program takes crim3.py and makes it so that the size of the comparison patch is different
#than the fill in patch


import cv2 as cv
import numpy as np
from scipy.signal import convolve2d
import time
import math
import warnings
import os
warnings.filterwarnings("ignore")

np.seterr(divide='ignore', invalid='ignore') # https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
#Issue is in the for loop where I find the norms of N
np.set_printoptions(precision=8)


class TVA: #Loki Reference
    #Homemade version of matlab tic and toc functions
    def tic():   
        global startTime_for_tictoc
        startTime_for_tictoc = time.time()
        print("Timer started at: " + time.strftime("%H:%M:%S"))

    def toc():
        if 'startTime_for_tictoc' in globals():
            #elapsed_time = str((time.time() - startTime_for_tictoc)//60) + " minutes, " + str((time.time() - startTime_for_tictoc)%60) + " seconds." 
            #print ("Elapsed time is: " + elapsed_time)
            return (time.time() - startTime_for_tictoc)
            
        else:
            print ("Toc: start time not set")
            return None

class imagePrep:
    def getMask(inImg):
        mask = np.zeros_like(inImg)
        mask[np.where(inImg==0)] = 255
        return mask
        
    def display_image(inMtrx):
        cv.imshow("Display Window", inMtrx)
        cv.waitKey(0) #Keeps window open until the user presses a key



class inpainting: 
    def getPatch(p, psz):
        halfPatch = (psz-1)/2 
        rows = np.arange( (p[0] - halfPatch), (p[0] + halfPatch+1))
        cols = np.transpose(np.arange( (p[1] - halfPatch), (p[1] + halfPatch+1)))
        rows = (rows * np.ones((psz, psz))).astype(int)
        cols = (np.transpose(cols *np.ones((psz, psz)))).astype(int) 
        
        return [rows, cols]
    
    
    def bestexemplar(img, Ip, ctr, toFill, sourceRegion): 
        
        patchErr = 0.0
        bestErr = 1000000000.0 
        best = None
        toFill = np.logical_not(toFill)
        
        patchSz = Ip.shape[0]
        halfPatch = (np.floor(patchSz/2)).astype(int)  
        hyp = np.floor((img.shape[0] + img.shape[1])/(2*5)).astype(int) #search a 5th of the image around the damage
        iStart = (ctr[0] -hyp).astype(int)
        iEnd = ctr[0] + hyp.astype(int)
        
        if iStart < halfPatch:
            iStart = halfPatch.astype(int)
        if iEnd > img.shape[0] - halfPatch:
           iEnd =  (img.shape[0] - halfPatch).astype(int)
        
        for i in range(iStart, iEnd):
            jLen = (hyp **2 - ((ctr[0] -i)**2)) ** .5
            jStart = (ctr[1] - jLen).astype(int)
            jEnd = (ctr[1] + jLen).astype(int)
            if jStart < halfPatch:
                jStart = halfPatch.astype(int)
            if jEnd > img.shape[1] - halfPatch:
               jEnd =  (img.shape[1] - halfPatch).astype(int)
               
        
            for j in range(jStart, jEnd): #We've arrived at our destination, lets set up shop  
                [tempRows, tempCols] = inpainting.getPatch([i,j], patchSz)
                if sourceRegion[tempRows, tempCols].all():
                    tempPatch = img[tempRows, tempCols]
                    patchErr = 0.0
                    pixErr = 0.0                
                    for m in range(0, patchSz):
                        for n in range(0, patchSz):
                            if toFill[m,n].all():
                                for g in range(img.shape[2]):
                                    pixErr = Ip[m,n,g] - tempPatch[m,n,g]
                                    patchErr += pixErr*pixErr   
                    if(patchErr < bestErr):
                        bestErr = patchErr
                        best = [i,j]
        if best == None: #If this happens you probably restricted the image too much
            raise Exception("No best match") 
        return best

    def img2ind(img):
        s = img.shape
        arr = np.arange(s[0]*s[1]) #https://stackoverflow.com/questions/11892358/matlab-vs-python-reshapenewArr = np.asarray(arr)
        ind = arr.reshape(s[0], s[1])
        return np.transpose(ind)
        
    def ind2img(ind, img):
        img2 = np.zeros((ind.shape[0],ind.shape[0], img.shape[2]) )
        for i in range(0, img.shape[2]):
            temp = img[:,:,i]
            [y,x] = np.unravel_index(ind, temp.shape)
            #x-=1
            img2[:,:,i] = temp[x,y]
        return img2


    """ Copied from the matlab code:
    % Inputs: 
    %   - origImg        original image or corrupted image
    %   - mask           implies target region (1 denotes target region) 
    %   - psz:           patch size (odd scalar). If psz=5, patch size is 5x5.
    %
    % Outputs:
    %   - inpaintedImg   The inpainted image; an MxNx3 matrix of doubles. 
    """   
    def inpainting(origImg, mask, psz):
        if psz % 2 == 0:
            raise Exception("Patch size psz must be odd")
        if not np.any(origImg):
            raise Exception("Original image invalid")
        if not np.any(mask):
            raise Exception("Mask invalid")
        
        diff = np.setdiff1d(mask, [0,255])
        if diff.size:
            raise Exception("Issue with mask!")
        
        if len(mask.shape) == 2:
            temp = np.zeros((mask.shape[0],mask.shape[0], origImg.shape[2]))
            for x in range(origImg.shape[2]):
                temp[:,:,x] = mask
            mask = np.copy(temp)
        
        
        fillRegion = np.sum(mask, axis =2)/3
        inpainted_img = np.copy(origImg)
        
        ind = inpainting.img2ind(inpainted_img) 
        sz = [inpainted_img.shape[0], inpainted_img.shape[1]]
        sourceRegion = np.logical_not(fillRegion) 
        
        Ix = np.zeros(inpainted_img.shape)
        Iy = np.zeros_like(Ix)
        
        for i in range(inpainted_img.shape[2]): #finds the boundaries of the damaged region
            [Iy[:,:,i], Ix[:,:,i] ]= np.gradient(mask[:,:,i])
        Ix = np.sum(Ix, axis =2)/(mask.shape[2]*255)
        Iy = np.sum(Iy, axis =2)/(mask.shape[2]*255)
        gradSum = sum(abs(Ix) + abs(Iy))
        
        #Initialize confidence and data terms
        C = np.double(sourceRegion)
        D = np.tile(-.1, sz) 
        Laplacian = [ [1,1,1],[1,-8,1],[1,1,1] ] #Laplacian Matrix
        
        numWhiles = 1
        
        while 255 in fillRegion:  
            print("Iteration: " + str(numWhiles))
            cv.imshow('live progress', inpainted_img)
            cv.waitKey(5)
            
            fillRegionD = np.double(fillRegion)
            
            conv2dHolder = np.rot90(convolve2d(np.rot90(fillRegionD, 2), np.rot90(Laplacian, 2), mode='same'), 2) # this is right
            dR = np.argwhere(conv2dHolder>0)
            
            [Nx, Ny] = np.gradient(np.double(np.logical_not(fillRegionD)))
            N = []  
            for x in range(len(dR)):
                N.append(np.array([Nx[dR[x,0],dR[x,1] ], Ny[dR[x,0],dR[x,1]] ])) 
            N = np.flip(np.array(N),1)
            
            
            for x in range(len(N)): #See if you can find a better way to do this
                norm = np.linalg.norm(N[x]) 
                N[x] = N[x]/norm 
            N[np.isnan(N)] = 0 
            
              
            for k in dR:
                [rows, cols] = inpainting.getPatch(k,psz)
                q = np.logical_not(fillRegion[rows, cols].astype(int))
                C[k[0], k[1]] = sum(C[rows[q], cols[q]])/rows.size
            
            for x in range(len(dR)):
                D[dR[x,0], dR[x,1]] = abs(Ix[dR[x,0], dR[x,1]]*N[x,0]+Iy[dR[x,0], dR[x,1]]*N[x,1]) + 0.001 
            
            priorities = []
            for x in dR:
                priorities.append(C[x[0], x[1]] * D[x[0], x[1]])
            priorities = np.array(priorities)
            ndx = np.rot90(np.where(priorities == np.max(priorities)))
            
            p = dR[ndx[0]]
            p = p[0]
            
            [pRows, pCols] = inpainting.getPatch(p, psz)
            Hp = inpainted_img[pRows, pCols]
            
            toFill = fillRegion[pRows, pCols]
            toFill = np.logical_not(np.logical_not(toFill))
            
            Hq = inpainting.bestexemplar(inpainted_img, Hp, p, np.transpose(toFill), sourceRegion)
            
            [HqRows, HqCols] = inpainting.getPatch(Hq, psz)
            
            
            #update fill region
            fillRegion[pRows, pCols] = 0
            
            
            #Propagate confidence & isophote values
            C[pRows[toFill], pCols[toFill]] = np.copy(C[p[0], p[1]])
            Ix[pRows[toFill], pCols[toFill]] = Ix[HqRows[toFill], HqCols[toFill]]
            Iy[pRows[toFill], pCols[toFill]] = Iy[HqRows[toFill], HqCols[toFill]]
            
            #Copy image data from Hq to Hp  
            ind[(pRows[toFill]), (pCols[toFill])] = ind[HqRows[toFill], HqCols[toFill]]
            inpainted_img[pRows, pCols] = inpainting.ind2img(ind[pRows, pCols], origImg)
            numWhiles +=1
        return inpainted_img




def main(orig_img, psz = 5):
    mask = imagePrep.getMask(orig_img)
    TVA.tic()
    inpaintedImage = inpainting.inpainting(orig_img, mask, psz)
    TVA.toc()
    return inpaintedImage



if __name__ == "__main__":

    orig_img = cv.imread(cv.samples.findFile('barb-rect.png'))
    mask = imagePrep.getMask(orig_img)
    
    
    dmgPxlCnt = (orig_img.shape[0]*orig_img.shape[1]) - np.count_nonzero(orig_img)/3
    
    TVA.tic()
    inpaintedImage = inpainting.inpainting(orig_img, mask, 3)
    TVA.toc()
    
    folderName = "Trials/" + time.strftime("%m.%d--%H.%M.%S")
    os.mkdir(folderName)
    cv.imwrite(folderName + "/output.png", inpaintedImage) #this saves the file
    cv.imwrite(folderName + "/damaged.png", orig_img) #this saves the file
    f = open((folderName + "/notes.txt"), "w+")
    f.write("Endtime: " + time.strftime("%H:%M:%S"))
    f.write("\nRuntime: " +  str((time.time() - startTime_for_tictoc)//60) + " minutes, " + str((time.time() - startTime_for_tictoc)%60) + " seconds." )
    f.write("\nImage ran on the crim3.py function.")
    f.write("\nPsz: " + str(psz))
    f.write("\nIterations: " + str(numWhiles) )
    f.write("\nDamaged Pixels: " + str(dmgPxlCnt))
    f.close()
    print("Images saved to folder.")
    
    print("Displaying inpainted and damaged images:")
    imagePrep.display_image(orig_img)
    imagePrep.display_image(inpaintedImage)
