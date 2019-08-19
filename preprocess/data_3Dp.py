# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:13:47 2018
# 数据预处理工具函数，整体功能是读取指定路径下的npy格式数据文件，预处理后生成自己的训练/验证/测试数据
@author: liuhuaqing
"""
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import product
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def convert_to_onehot(y,C):
    #将label转化为one-hot形式
    y = y.astype(int)
    y_shape = list(y.shape)
    y_shape[-1] = C
    y_onehot = np.reshape(np.eye(C)[y.reshape(-1)], y_shape) 
    return y_onehot.astype(float)

#弹性变形
def elastic_transform_V0(image3D,mask3D,alpha=1,sigma=1):
    # 弹性变形函数版本0
    shape = image3D.shape
    random_state = np.random.RandomState(None) 
    dx = gaussian_filter((random_state.rand(*shape)*2-1),sigma)*alpha
    dy = gaussian_filter((random_state.rand(*shape)*2-1),sigma)*alpha
    dz = gaussian_filter((random_state.rand(*shape)*2-1),sigma)*alpha
    x,y,z,c = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]),np.arange(shape[2]),np.arange(1))
    x,y,z = x[:,:,:,0],y[:,:,:,0],z[:,:,:,0]
    indices = np.reshape(y+dy,(-1,1)),np.reshape(x+dx,(-1,1)),np.reshape(z+dz,(-1,1))
    image3D_elastic = map_coordinates(image3D,indices,order=1,mode='reflect').reshape(shape)
    mask3D_elastic = map_coordinates(mask3D,indices,order=1,mode='reflect').reshape(shape)
    return image3D_elastic, mask3D_elastic

def elastic_transform_V1(image3D,alpha=1,sigma=1):
    # 弹性变形函数版本1，实现的功能和版本0一样
    shape = image3D.shape
    random_state = np.random.RandomState(None) 
    dx = gaussian_filter((random_state.rand(*shape)*2-1),sigma)*alpha
    dy = gaussian_filter((random_state.rand(*shape)*2-1),sigma)*alpha
    dz = gaussian_filter((random_state.rand(*shape)*2-1),sigma)*alpha
    x,y,z,c = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]),np.arange(shape[2]),np.arange(1))
    x,y,z = x[:,:,:,0],y[:,:,:,0],z[:,:,:,0]
    indices = np.reshape(y+dy,(-1,1)),np.reshape(x+dx,(-1,1)),np.reshape(z+dz,(-1,1))
    image3D_elastic = map_coordinates(image3D,indices,order=1,mode='reflect').reshape(shape)
    return image3D_elastic

# 旋转，axis为旋转轴，0,1,2分别代表x,y,z轴
# theta为旋转角度，单位已改为度，非弧度
# center为旋转中心，其为一维np数组[x,y,z]，默认值为图像中心点
def rotation(data, axis, theta, c = np.array([]), patch_shape=(64,64,32)):# c代表旋转点
    #3D矩阵的旋转（实际仅仅又绕z轴旋转的功能，在该项目中，z轴特指CT图像人体身高方向）
    theta = -np.pi * theta / 180
    if c.size == 0:
        c = np.array([np.floor((data.shape[0]-1)/2), np.floor((data.shape[1]-1)/2), np.floor((data.shape[1]-1)/2)])    

    s = patch_shape
    new_data = np.zeros(s) # 补零    

    # 绕x轴旋转
    if axis == 0:
        print('axis==0 not supported')         
    # 绕y轴旋转              
    elif axis == 1:
        print('axis==1 not supported')

    # 绕z轴旋转
    else:
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        i0,j0 = np.int(-s[0]/2),np.int(-s[1]/2)
        i1,j1 = i0+s[0], j0+s[1]
        z0 = c[2]-np.floor(patch_shape[2]/2).astype(int)
        z1 = z0+patch_shape[2]
        for i,j in product( range(i0,i1),range(j0,j1) ):                              
            x = np.floor(i*c_theta-j*s_theta+c[0]).astype(int)
            y = np.floor(i*s_theta+j*c_theta+c[1]).astype(int)
            new_data[i-i0,j-j0,:] = data[x,y,z0:z1]     
    return new_data   

def calc_c_range(data,patch_shape=(64,64,32)):
    # 该函数的作用是：服务于数据扩增中的随机旋转与裁剪，计算允许截取数据块的原始CT范围
    c_range = np.zeros([2,3])
    c_range[0:2,0:2] = np.vstack( 
            (np.ceil( np.array([0,0])+np.linalg.norm(np.array(patch_shape)[0:2])/2),
             np.floor(np.array(data.shape)[0:2]-1-np.linalg.norm(np.array(patch_shape)[0:2])/2)) 
            )
    c_range[0:2,2] = np.array([
            np.ceil(np.array(patch_shape[2])/2).astype(int),
            np.floor(np.array(data.shape[2])-1-np.array(patch_shape[2])/2).astype(int)])
    return c_range


def read_npy_file(item1,item2,get_label=True,do_augment=True,do_patch=True,D=16,H=128,W=128):
    # 读取文件并根据do_patch决定是否做数据扩增
    # 训练的时候做数据扩增，验证和测试的时候不做
    image = np.load(item1)+70.75# 原始数据三个维度对应人体的：左右、前后、上下
    if get_label:
        mask = np.load(item2)
    else:
        mask = np.array([]) 
    #随机裁剪
    if do_patch and not do_augment:
        imageShape = image.shape
        x1 = np.random.randint(low=0, high=imageShape[0]-H, dtype='l')#这个方法产生离散均匀分布的整数，这些整数大于等于low，小于high。
        y1 = np.random.randint(low=0, high=imageShape[1]-W, dtype='l')
        z1 = np.random.randint(low=0, high=imageShape[2]-D, dtype='l')
        image = image[x1:x1+H,y1:y1+W,z1:z1+D]
        if get_label:
            mask = mask[x1:x1+H,y1:y1+W,z1:z1+D]  
    #随机旋转与裁剪        
    if do_patch and do_augment:
        c_range = calc_c_range(image,patch_shape=(H,W,D))
        c = np.array([np.random.randint(low=c_range[0,0],high=c_range[1,0]),
                      np.random.randint(low=c_range[0,1],high=c_range[1,1]),
                      np.random.randint(low=c_range[0,2],high=c_range[1,2])])
        theta = np.random.normal(loc=0.0,scale=10)#旋转角
        image = rotation(image,axis=3,c=c,theta=theta,patch_shape=(H,W,D))
        if get_label:
            mask = rotation(mask,axis=3,c=c,theta=theta,patch_shape=(H,W,D))
    
    if do_augment:
# =============================================================================
#         #弹性变形，实验发现性价比不高：耗时且效果提升不明显
#         alpha=2
#         sigma=1
#         if get_label:
#             image,mask = elastic_transform_V0(image,mask,alpha,sigma)
#         else:
#             image = elastic_transform_V1(image,alpha,sigma)
# =============================================================================
  
        #正太分布随机噪声
        noise = np.random.normal(0, 1, image.shape) 
        image = image+noise
        image = image.astype(np.float32)  
        if get_label:
            mask = mask.astype(np.float32)    
    
        #左右翻转
        if np.random.rand()>0.5:
            image = np.flip(image,0) 
            if get_label:
                mask = np.flip(mask,0)            
    image = np.transpose(image,(2,0,1))#调换维度顺序后，各维度分别是：D,H,W
    image = image[:,:,:,np.newaxis]#在最后增加一个channel维度,得到的维度分别是：D,H,W,channel
    if get_label:
        mask = np.transpose(mask,(2,0,1))#调换维度顺序后，各维度分别是：D,H,W,channel
        mask = mask[:,:,:,np.newaxis]#在最后增加一个channel维度
    if get_label: 
        return (image,mask)
    else:
        return image


def get_files(file_dir,get_label=True,do_shuffle=True):
    # 获取CT和mask成对的数据集文件名，并打乱顺序但不影响CT和mask的对应关系
    # get_label：表示是否获取标签mask
    
    # step1；获取file_dir下所有的图路径名
    image_list = []
    mask_list = []
    for file in os.listdir(file_dir+'/CT'):
        image_list.append(file_dir+'/CT'+'/'+file)
    if get_label:
        for file in os.listdir(file_dir+'/SegmentationLabel'):
            mask_list.append(file_dir+'/SegmentationLabel'+'/'+file)
        
    # step2: 对生成的图片路径和标签做打乱处理
    # 利用shuffle打乱顺序
    if do_shuffle:
        if get_label:
            temp = np.array([image_list,mask_list])
            temp = temp.transpose()#n行2列
            np.random.shuffle(temp)
            #从打乱顺序的temp中取list
            image_list = list(temp[:,0])#打乱顺序的文件路径名字符串
            mask_list = list(temp[:,1])#打乱顺序的文件路径名字符串
        else:
            temp = np.array(image_list)
            temp = temp.transpose()#n行1列
            np.random.shuffle(temp)
            image_list = list(temp)#打乱顺序的文件路径名字符串
    if get_label:
        #读取带标签数据集的时候：
        return image_list, mask_list
    else:
        #读取不带标签数据集的时候：
        return image_list 




def get_dataBatch(data_dir,D,W,H,batch_size,get_label=True,do_patch=True):
    # 该函数作用：获取一个batch的数据
    # get_label：表示是否获取标签
    
    if get_label:
        image_list, mask_list = get_files(data_dir,get_label=True,do_shuffle=True)#从路径中获取成对的文件列表
        assert len(image_list)==len(mask_list)
    else:
        image_list = get_files(data_dir,get_label=False,do_shuffle=True)#从路径中获取成对的文件列表
               
    num = len(image_list)#数据集中npy文件的数量
    if do_patch: #do_patch表示用一个小窗口从CT中截取数据。训练的时候do_patch=True,验证和测试的时候False
        image_batch = np.zeros([batch_size,D,H,W,1])
        if get_label:
            mask_batch = np.zeros([batch_size,D,H,W,1])
        i = 0
        idxs = np.random.randint(0,num,batch_size) # 随机数，用以随机读取CT和label(mask)文件
        for k in idxs:
            if get_label:
                image,mask = read_npy_file(image_list[k],mask_list[k],get_label=True,do_augment=True,do_patch=True,D=D,H=H,W=W)
                image_batch[i,:,:,:,:] = image
                mask_batch[i,:,:,:,:] = mask
            else:
                image = read_npy_file(image_list[k],item2=None,get_label=False,do_augment=True,do_patch=True,D=D,H=H,W=W)  
                image_batch[i,:,:,:,:] = image
            i += 1
        if get_label:
            image_mask_batch = (image_batch,mask_batch)
            return image_mask_batch
        else:
            return image_batch
    else:
        batch_size = 1 #验证和测试的时候只能逐个读取CT和mask对
        idxs = np.random.randint(0,num,batch_size) # 随机数
        if get_label:
            image,mask = read_npy_file(image_list[idxs[0]],mask_list[idxs[0]],get_label=True,do_augment=True,do_patch=True,D=D,H=H,W=W)
        else:
            image = read_npy_file(image_list[idxs[0]],mask_list[idxs[0]],do_augment=False,do_patch=True,D=D,H=H,W=W)
        
        image = image[np.newaxis,:,:,:,:]
        if get_label:
            mask = mask[np.newaxis,:,:,:,:]
            image_mask = (image,mask)
    if get_label:
        return image_mask
    else:
        return image


# 以下代码是为了测试以上函数正确性，读取指定路径的图片，预处理后，在控制台显示出来
if __name__ == "__main__":
    batch_size = 4
    train_dir = './data/Data_3classes/data_20181111/data_npy_20181111/test' 
    tt = time.time()
    for i in range(20):
        try:
            print(i)
            image_mask_batch = get_dataBatch(train_dir,2,100,100,batch_size,get_label=True,do_patch=False)
            imgs = image_mask_batch[0]
            masks = image_mask_batch[1]
            plt.figure()
            plt.subplot(1,2,1),plt.imshow( imgs[0,1,:,:,0] ),plt.axis('off'),plt.title('CT')
            plt.subplot(1,2,2),plt.imshow( masks[0,1,:,:,0]/2.0*255.0 ),plt.axis('off'),plt.title('label')
            plt.show()
        except tf.errors.OutOfRangeError:
            print('start new epoch')
        print(time.time() - tt)
        
