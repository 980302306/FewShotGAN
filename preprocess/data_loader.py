import os
from PIL import Image
from glob import glob
import tensorflow as tf
import numpy as np


def _read_npy_file(item1,item2,frame,width,height):
    image = np.load(item1.decode())+70.75 #-70.75是均值
    mask = np.load(item2.decode())

    imageShape = image.shape
    x1 = np.random.randint(low=0, high=imageShape[0]-width, dtype='l')#这个方法产生离散均匀分布的整数，这些整数大于等于low，小于high。
    y1 = np.random.randint(low=0, high=imageShape[1]-height, dtype='l')
    z1 = np.random.randint(low=0, high=imageShape[2]-frame, dtype='l')
    image = image[x1:x1+width,y1:y1+height,z1:z1+frame]
    mask = mask[x1:x1+width,y1:y1+height,z1:z1+frame]
    
    image = np.transpose(image,(2,0,1))#调换维度顺序后，各维度分别是：frame,width,height
    image = image[:,:,:,np.newaxis]#在最后增加一个channel维度,得到的维度分别是：frame,width,height,channel
    mask = np.transpose(mask,(2,0,1))#调换维度顺序后，各维度分别是：frame,width,height,channel
    mask = mask[:,:,:,np.newaxis]#在最后增加一个channel维度
    image = tf.convert_to_tensor( (image.astype(np.float32))/1000 )
    mask = tf.convert_to_tensor( mask.astype(np.float32) )
    
    return (image,mask)

def get_files(root,shuffle=True):
    # step1；获取file_dir下所有的图路径名
    ext = 'npy'
    CT_root = os.path.join(root,'CT')
    SegmentationLabel_root = os.path.join(root,'SegmentationLabel')  
    image_list = glob("{}/*.{}".format(CT_root,ext))
    mask_list = glob("{}/*.{}".format(SegmentationLabel_root,ext))  
    
    #step2: 对生成的图片路径和标签做打乱处理
    #利用shuffle打乱顺序
    if shuffle:
        temp = np.array([image_list,mask_list])
        temp = temp.transpose()#n行2列
        np.random.shuffle(temp)
        #从打乱顺序的temp中取list
        image_list = list(temp[:,0]) #打乱顺序的文件路径名字符串
        mask_list = list(temp[:,1]) #打乱顺序的文件路径名字符串
    return image_list , mask_list


def get_batch(image_list,mask_list,batch_size,frame,width,height,data_format='NDCHW'):
    image_mask_item = tf.data.Dataset.from_tensor_slices((image_list,mask_list))
    image_mask_item = image_mask_item.shuffle(buffer_size=batch_size)
    image_mask_item = image_mask_item.map(lambda item1,item2: tuple(tf.py_func(_read_npy_file,[item1,item2,frame,width,height],[tf.float32,tf.float32]))) 
    image_mask_batch = image_mask_item.batch(batch_size)
    return image_mask_batch

def get_dataBatch(data_dir,image_F,image_W,image_H,batch_size):
    image_list , mask_list = get_files(data_dir)
    image_mask_batch = get_batch(image_list,mask_list,batch_size,image_F,image_W,image_H)
    iterator_get_dataBatch = tf.data.Iterator.from_structure(image_mask_batch.output_types,image_mask_batch.output_shapes)
    init_op_get_dataBatch = iterator_get_dataBatch.make_initializer(image_mask_batch)
    return iterator_get_dataBatch, init_op_get_dataBatch
