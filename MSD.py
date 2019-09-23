# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:00:14 2019

@author: 980302306
"""

import numpy as np
from scipy.ndimage import morphology
import cv2

'''
Refer to:
    
https://mlnotebook.github.io/post/surface-distance-function/


'''
def surfd(input1, input2, sampling=1, connectivity=1):
    
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
    

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)

    
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
       
    
    return sds


'''
测试：

用的是Few-shot的图片，从左到右分别是原图，分割后的图，ground truth
由于源代码有padding，所以平分了width，舍弃了padding的部分

然而：
事实上，输入应该是MRI SCAN的三维图像
'''
# =============================================================================
# img=cv2.imread('66.png')
# width=img.shape[1]//3
# img0=img[:,:width,:]
# img1=img[:,width:2*width,:]
# img2=img[:,2*width:3*width,:]  #Ground Truth
# =============================================================================
item1='./data/labeled/SegmentationLabel/anon098-L5S1-disc-vessel.npy'
image = np.load(item1)+70.75# 原始数据三个维度对应人体的：左右、前后、上下
'''
关于使用：（根据文章）
The sampling vector is a typical pixel-size from an MRI scan 
and the 1 indicated I’d like a 6 neighbour (cross-shaped) kernel for finding the edges.
'''
surface_distance = surfd(image, image, [1.25, 1.25, 10],1)

msd = surface_distance.mean()#Mean Surface Distance 
rms = np.sqrt((surface_distance**2).mean())#Residual Mean Square Distance
hd  = surface_distance.max()#Hausdorff Distance
print(msd)
