import os
import numpy as np
import tensorflow as tf
import math
from PIL import Image
#import pdb

F = tf.app.flags.FLAGS


"""
Save tensorflow model
Parameters:
* checkpoint_dir - name of the directory where model is to be saved
* sess - current tensorflow session
* saver - tensorflow saver
"""
def save_model(checkpoint_dir, sess, saver):
  model_name = "model.ckpt"
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  saver.save(sess, os.path.join(checkpoint_dir, model_name))


"""
Load tensorflow model
Parameters:
* checkpoint_dir - name of the directory where model is to be loaded from
* sess - current tensorflow session
* saver - tensorflow saver
Returns: True if the model loaded successfully, else False
"""
def load_model(checkpoint_dir, sess, saver):
  print(" [*] Reading checkpoints...")
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    return True
  else:
    return False

"""
To recompose an array of 3D images from patches
"""
def recompose3D_overlap(preds, img_h, img_w, img_d, stride_h, stride_w, stride_d):
  patch_h = preds.shape[1]
  patch_w = preds.shape[2]
  patch_d = preds.shape[3]
  N_patches_h = (img_h-patch_h)//stride_h+1
  N_patches_w = (img_w-patch_w)//stride_w+1
  N_patches_d = (img_d-patch_d)//stride_d+1
  N_patches_img = N_patches_h * N_patches_w * N_patches_d
  print("N_patches_h: " ,N_patches_h)
  print("N_patches_w: " ,N_patches_w)
  print("N_patches_d: " ,N_patches_d)
  print("N_patches_img: ",N_patches_img)
  assert(preds.shape[0]%N_patches_img==0)
  N_full_imgs = preds.shape[0]//N_patches_img
  print("According to the dimension inserted, there are " \
          +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w)+"x" +str(img_d) +" each)")
  # itialize to zero mega array with sum of Probabilities
  raw_pred_martrix = np.zeros((N_full_imgs,img_h,img_w,img_d)) 
  raw_sum = np.zeros((N_full_imgs,img_h,img_w,img_d))
  final_matrix = np.zeros((N_full_imgs,img_h,img_w,img_d),dtype='uint16')

  k = 0 
  # iterator over all the patches
  for i in range(N_full_imgs):
    for h in range((img_h-patch_h)//stride_h+1):
      for w in range((img_w-patch_w)//stride_w+1):
        for d in range((img_d-patch_d)//stride_d+1):
          raw_pred_martrix[i,h*stride_h:(h*stride_h)+patch_h,\
                                w*stride_w:(w*stride_w)+patch_w,\
                                  d*stride_d:(d*stride_d)+patch_d]+=preds[k]
          raw_sum[i,h*stride_h:(h*stride_h)+patch_h,\
                          w*stride_w:(w*stride_w)+patch_w,\
                            d*stride_d:(d*stride_d)+patch_d]+=1.0
          k+=1
  assert(k==preds.shape[0])
  #To check for non zero sum matrix
  assert(np.min(raw_sum)>=1.0)
  final_matrix = np.around(raw_pred_martrix/raw_sum)
  return final_matrix


#functions below are added by liuhuaqing 2019-07-15
def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)


# 语义分割准确率的定义和计算，参考：https://blog.csdn.net/majinlei121/article/details/78965435
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n) #正常情况下全是True
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)#np.bincount 用于统计数组中（从小到大）给取值出现的次数

def Hist(a,b,n):
    hist = fast_hist(a,b,n)
    return hist
    
def pixelAccuracy(trueMask,predMask,n_cls):
    hist = Hist(trueMask,predMask,n_cls)
    PA = np.diag(hist).sum() / hist.sum()
    return PA

def MeanPixelAccuracy(trueMask,predMask,n_cls):
    #epsilon = 1
    hist = Hist(trueMask,predMask,n_cls)
    PAs = np.diag(hist) / hist.sum(1)
    return PAs

def IntersectionoverUnion(trueMask,predMask,n_cls):
    #epsilon = 1
    hist = Hist(trueMask,predMask,n_cls)
    IoUs = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return IoUs

def DiceScore(trueMask,predMask,n_cls):
   # epsilon = 1
    hist = Hist(trueMask,predMask,n_cls)
    correct_pred = np.diag(hist) # 给类别正确预测的像素点数
    pred_classes = np.sum(hist,0) # 预测处的各类别像素点数,
    true_classes = np.sum(hist,1) # 真实的各类别像素点数
    DSs = 2*correct_pred/(pred_classes+true_classes)
    return DSs