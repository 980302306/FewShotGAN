from __future__ import division
import os
import pickle 
import tensorflow as tf
slim = tf.contrib.slim

import sys
sys.path.insert(0, '../preprocess/')
sys.path.insert(0, '../lib/')

from operations import *
from utils import *
from preprocess import *
import numpy as np
from six.moves import xrange
from sklearn.metrics import f1_score

from data_3Dp import *
from itertools import product
import time

F = tf.app.flags.FLAGS


"""
Model class

"""
class model(object):
  def __init__(self, sess, patch_shape, extraction_step):
    self.sess = sess
    self.patch_shape = patch_shape
    self.extraction_step = extraction_step
    self.g_bns = [batch_norm(name='g_bn{}'.format(i,)) for i in range(4)]
    if F.badGAN:
      self.e_bns = [batch_norm(name='e_bn{}'.format(i,)) for i in range(3)]


  def discriminator(self, patch, reuse=False):
    """
    Parameters:
    * patch - input image for the network
    * reuse - boolean variable to reuse weights
    Returns: 
    * logits
    * softmax of logits
    * features extracted from encoding path 
    """
    with tf.variable_scope('D') as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv3d_WN(patch, 32, name='d_h0_conv'))
      h1 = lrelu(conv3d_WN(h0, 32, name='d_h1_conv'))
      p1 = avg_pool3D(h1)

      h2 = lrelu(conv3d_WN(p1, 64, name='d_h2_conv'))
      h3 = lrelu(conv3d_WN(h2, 64, name='d_h3_conv'))
      p3 = avg_pool3D(h3)

      h4 = lrelu(conv3d_WN(p3, 128, name='d_h4_conv'))
      h5 = lrelu(conv3d_WN(h4, 128, name='d_h5_conv'))
      p5 = avg_pool3D(h5)

      h6 = lrelu(conv3d_WN(p5, 256, name='d_h6_conv'))
      h7 = lrelu(conv3d_WN(h6, 256, name='d_h7_conv'))

      up1 = slim.conv3d_transpose(h7,256,2,stride=[2,2,2],activation_fn=None,scope='d_up1_deconv')
      #up1 = deconv3d_WN(h7,256,name='d_up1_deconv')#该方式要求指定固定的batch_size，不利于验证和测试阶段逐张喂CT，故换为slim.conv3d_transpose，实现相同的功能
      up1 = tf.concat([h5,up1],4)
      h8 = lrelu(conv3d_WN(up1, 128, name='d_h8_conv'))
      h9 = lrelu(conv3d_WN(h8, 128, name='d_h9_conv'))
      
      up2 = slim.conv3d_transpose(h9,128,2,stride=[2,2,2],activation_fn=None,scope='d_up2_deconv')
      #up2 = deconv3d_WN(h9,128,name='d_up2_deconv')
      up2 = tf.concat([h3,up2],4)
      h10 = lrelu(conv3d_WN(up2, 64, name='d_h10_conv'))
      h11 = lrelu(conv3d_WN(h10, 64, name='d_h11_conv'))

      up3 = slim.conv3d_transpose(h11,64,2,stride=[2,2,2],activation_fn=None,scope='d_up3_deconv')
      #up3 = deconv3d_WN(h11,64,name='d_up3_deconv')
      up3 = tf.concat([h1,up3],4)
      h12 = lrelu(conv3d_WN(up3, 32, name='d_h12_conv'))
      h13 = lrelu(conv3d_WN(h12, 32, name='d_h13_conv'))

      h14 = conv3d_WN(h13, F.num_classes,name='d_h14_conv')

      return h14,tf.nn.softmax(h14),h6

  def generator(self, z, phase):
    """
    Parameters:
    * z - Noise vector for generating 3D patches
    * phase - boolean variable to represent phase of operation of batchnorm
    Returns: 
    * generated 3D patches
    """
    with tf.variable_scope('G') as scope:
      sh1, sh2, sh3, sh4 = int(self.patch_shape[0]/16), int(self.patch_shape[0]/8),\
                           int(self.patch_shape[0]/4), int(self.patch_shape[0]/2)

      h0 = linear(z, sh1*sh1*sh1*512,'g_h0_lin')
      h0 = tf.reshape(h0, [F.batch_size, sh1, sh1, sh1, 512])
      h0 = relu(self.g_bns[0](h0,phase))

      h1 = relu(self.g_bns[1](deconv3d(h0, [F.batch_size,sh2,sh2,sh2,256], 
                                                          name='g_h1_deconv'),phase))

      h2 = relu(self.g_bns[2](deconv3d(h1, [F.batch_size,sh3,sh3,sh3,128], 
                                                          name='g_h2_deconv'),phase))   

      h3 = relu(self.g_bns[3](deconv3d(h2, [F.batch_size,sh4,sh4,sh4,64], 
                                                          name='g_h3_deconv'),phase))

      h4 = slim.conv3d_transpose(h3,F.num_mod,2,stride=[2,2,2],activation_fn=None,scope='g_h4_deconv')
      #h4 = deconv3d_WN(h3, F.num_mod, name='g_h4_deconv')#该方式要求指定固定的batch_size，不利于验证和测试阶段逐张喂CT，故换为slim.conv3d_transpose，实现相同的功能

      return tf.nn.tanh(h4)

  def encoder(self, patch, phase):
    """
    Parameters:
    * patch - patches generated from the generator
    * phase - boolean variable to represent phase of operation of batchnorm
    Returns: 
    * splitted logits
    """
    with tf.variable_scope('E') as scope:
      h0 = relu(self.e_bns[0](conv3d(patch, 128, 5,5,5, 2,2,2, name='e_h0_conv'),phase))
      h1 = relu(self.e_bns[1](conv3d(h0, 256, 5,5,5, 2,2,2, name='e_h1_conv'),phase))
      h2 = relu(self.e_bns[2](conv3d(h1, 512, 5,5,5, 2,2,2, name='e_h2_conv'),phase))

      h2 = tf.reshape(h2, [h2.shape[0],h2.shape[1]*h2.shape[2]*h2.shape[3]*h2.shape[4]])
      h3 = linear_WN(h2, F.noise_dim*2,'e_h3_lin')
      
      h3 = tf.split(h3,2,1)
      return h3


  """
  Defines the Few shot GAN U-Net model and the corresponding losses

  """
  def build_model(self):
    self.patches_lab = tf.placeholder(tf.float32, [None, self.patch_shape[0], 
                                self.patch_shape[1], self.patch_shape[2], F.num_mod], name='real_images_l')
    self.patches_unlab = tf.placeholder(tf.float32, [None, self.patch_shape[0], 
                                self.patch_shape[1], self.patch_shape[2], F.num_mod], name='real_images_unl')

    self.z_gen = tf.placeholder(tf.float32, [None, F.noise_dim], name='noise')
    self.labels = tf.placeholder(tf.uint8, [None, self.patch_shape[0], self.patch_shape[1],
                                                         self.patch_shape[2]], name='image_labels')
    self.phase = tf.placeholder(tf.bool)

    #To make one hot of labels
    self.labels_1hot = tf.one_hot(self.labels, depth=F.num_classes) #<tf.Tensor 'one_hot:0' shape=(30, 32, 32, 32, 4) dtype=float32>

    # To generate samples from noise
    self.patches_fake = self.generator(self.z_gen, self.phase)

    # Forward pass through network with different kinds of training patches 
    self.D_logits_lab, self.D_probdist, _= self.discriminator(self.patches_lab, reuse=False)
    self.D_logits_unlab, _, self.features_unlab\
                       = self.discriminator(self.patches_unlab, reuse=True)
    self.D_logits_fake, _, self.features_fake\
                       = self.discriminator(self.patches_fake, reuse=True)


    # To obtain Validation Output
    self.Val_output = tf.argmax(self.D_probdist, axis=-1)

    # Supervised loss
    # Weighted cross entropy loss (You can play with these values)
    # Weights of different class are: Background- 0.33, Bone- 1.5, Dice- 0.83, nerve- 1.33
    class_weights = tf.constant([[0.33, 1.5, 0.83, 1.33]])
    weights = tf.reduce_sum(class_weights * self.labels_1hot, axis=-1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.D_logits_lab, labels=self.labels_1hot)
    weighted_losses = unweighted_losses * weights #<tf.Tensor 'mul_1:0' shape=(30, 32, 32, 32) dtype=float32>
    self.d_loss_lab = tf.reduce_mean(weighted_losses)

    # Unsupervised loss
    self.unl_lsexp = tf.reduce_logsumexp(self.D_logits_unlab,-1) #计算log(sum(exp()))
    self.fake_lsexp = tf.reduce_logsumexp(self.D_logits_fake,-1)
    # Unlabeled loss
    self.true_loss = - F.tlw * tf.reduce_mean(self.unl_lsexp) + F.tlw * tf.reduce_mean(tf.nn.softplus(self.unl_lsexp))
    # Fake loss
    self.fake_loss = F.flw * tf.reduce_mean(tf.nn.softplus(self.fake_lsexp))
    self.d_loss_unlab = self.true_loss + self.fake_loss

    #Total discriminator loss
    self.d_loss = self.d_loss_lab + self.d_loss_unlab

    #Feature matching loss
    self.g_loss_fm = tf.reduce_mean(tf.abs(tf.reduce_mean(self.features_unlab,0) \
                                                  - tf.reduce_mean(self.features_fake,0)))

    if F.badGAN:
      # Mean and standard deviation for variational inference loss
      self.mu, self.log_siDicea = self.encoder(self.patches_fake, self.phase)
      # Generator Loss via variational inference
      self.vi_loss = gaussian_nll(self.mu, self.log_siDicea, self.z_gen)
      # Total Generator Loss
      self.g_loss = self.g_loss_fm + F.vi_weight * self.vi_loss
    else:
      # Total Generator Loss
      self.g_loss = self.g_loss_fm 


    t_vars = tf.trainable_variables()
    
    #define the trainable variables
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    if F.badGAN:
      self.e_vars = [var for var in t_vars if 'e_' in var.name]

    self.saver = tf.train.Saver()


  """
  Train function
  Defines learning rates and optimizers.
  Performs Network update and saves the losses
  """
  def train(self):

    # Optimizer operations
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      d_optim = tf.train.AdamOptimizer(F.learning_rate_D, beta1=F.beta1D)\
                  .minimize(self.d_loss,var_list=self.d_vars)
      g_optim = tf.train.AdamOptimizer(F.learning_rate_G, beta1=F.beta1G)\
                  .minimize(self.g_loss,var_list=self.g_vars)
      if F.badGAN:
        e_optim = tf.train.AdamOptimizer(F.learning_rate_E, beta1=F.beta1E)\
                  .minimize(self.g_loss,var_list=self.e_vars)

    tf.global_variables_initializer().run()

    max_par=0.0
    max_loss=100
    for epoch in xrange(int(F.epoch)):
      idx = 0      
      for idx in xrange(int(F.iter_per_epoch)):
          #载入不带人工标识的训练数据(含CT):
          CT_batch_unlabeled = get_dataBatch(
                  F.train_dir_unlabeled,
                  self.patch_shape[2],self.patch_shape[0],self.patch_shape[1],F.batch_size,
                  get_label=False,# 也读取人工标注
                  do_patch=True)# 读取一批训练数据
          #载入带人工标识的训练数据（含成对的CT和mask）:
          CT_mask_batch_labeled = get_dataBatch(
                  F.train_dir_labeled,
                  self.patch_shape[2],self.patch_shape[0],self.patch_shape[1],F.batch_size,
                  get_label=True,# 也读取人工标注
                  do_patch=True)# 读取一批训练数据
          batch_iter_train = (CT_mask_batch_labeled[0],CT_batch_unlabeled,CT_mask_batch_labeled[1])
          
          total_val_loss=0 # total_xx_loss记录的是一个epoch中每一次迭代的loss之和（平均）
          total_train_loss_CE=0
          total_train_loss_UL=0
          total_train_loss_FK=0
          total_gen_FMloss =0

          
          patches_lab, patches_unlab, labels = batch_iter_train
          labels = np.squeeze(labels)
          # Network update
          sample_z_gen = np.random.uniform(-1, 1, [F.batch_size, F.noise_dim]).astype(np.float32)
          
          _ = self.sess.run(d_optim,feed_dict={self.patches_lab:patches_lab,self.patches_unlab:patches_unlab,
                                               self.z_gen:sample_z_gen,self.labels:labels, self.phase: True})
          if F.badGAN:
              _, _ = self.sess.run([e_optim,g_optim],feed_dict={self.patches_unlab:patches_unlab, self.z_gen:sample_z_gen,
                                   self.z_gen:sample_z_gen,self.phase: True})
          else:
              _ = self.sess.run(g_optim,feed_dict={self.patches_unlab:patches_unlab, self.z_gen:sample_z_gen,
                                                   self.z_gen:sample_z_gen,self.phase: True})
        
          feed_dict = {self.patches_lab:patches_lab,self.patches_unlab:patches_unlab,
                       self.z_gen:sample_z_gen,self.labels:labels, self.phase: True} 
          
          # Evaluate losses for plotting/printing purposes   
          d_loss_lab = self.d_loss_lab.eval(feed_dict)
          d_loss_unlab_true = self.true_loss.eval(feed_dict)
          d_loss_unlab_fake = self.fake_loss.eval(feed_dict)
          g_loss_fm = self.g_loss_fm.eval(feed_dict)
          
          total_train_loss_CE=total_train_loss_CE+d_loss_lab
          total_train_loss_UL=total_train_loss_UL+d_loss_unlab_true
          total_train_loss_FK=total_train_loss_FK+d_loss_unlab_fake
          total_gen_FMloss=total_gen_FMloss+g_loss_fm

          idx += 1
          if F.badGAN:
              vi_loss = self.vi_loss.eval(feed_dict)
              print(("Epoch:[%2d] [%4d/%4d] Labeled loss:%.2e Unlabeled loss:%.2e Fake loss:%.2e Generator FM loss:%.8f Generator VI loss:%.8f\n")%
                          (epoch, idx,F.iter_per_epoch,d_loss_lab,d_loss_unlab_true,d_loss_unlab_fake,g_loss_fm,vi_loss))
          else:
              print(("Epoch:[%2d] [%4d/%4d] Labeled loss:%.2e Unlabeled loss:%.2e Fake loss:%.2e Generator loss:%.8f \n")%
                          (epoch, idx,F.iter_per_epoch,d_loss_lab,d_loss_unlab_true,d_loss_unlab_fake,g_loss_fm))

      
        
        
        
      # Save the curret model
      save_model(F.checkpoint_dir, self.sess, self.saver)

      avg_train_loss_CE=total_train_loss_CE/(idx*1.0)
      avg_train_loss_UL=total_train_loss_UL/(idx*1.0)
      avg_train_loss_FK=total_train_loss_FK/(idx*1.0)
      avg_gen_FMloss=total_gen_FMloss/(idx*1.0)

      print('\n\n')
     
      # To compute average CTvise validation loss(cross entropy loss),IOUs and DSs
      avr_val_loss,avr_val_IOUS,avr_val_DSs = self.validate(epoch)            
      print("All validation patches Predicted")


      # For printing the validation results
      # F1_score = f1_score(lab2d, pred2d,[0,1,2,3],average=None)
      print("Validation Dice Coefficient.... ")
      print("Background:",avr_val_DSs[0])
      print("Bone:",avr_val_DSs[1])
      print("Dice:",avr_val_DSs[2])
      print("nerve:",avr_val_DSs[3])

      # To Save the best model
      if(max_par<(avr_val_DSs[2]+avr_val_DSs[3])):
        max_par=(avr_val_DSs[2]+avr_val_DSs[3])
        save_model(F.best_checkpoint_dir, self.sess, self.saver)
        print("Best checkpoint updated from validation results.")

      # To save the losses for plotting 
      print("Average Validation Loss:",avg_val_loss)
      with open('Val_loss_GAN.txt', 'a') as f:
        f.write('%.2e \n' % avg_val_loss)
      with open('Train_loss_CE.txt', 'a') as f:
        f.write('%.2e \n' % avg_train_loss_CE)
      with open('Train_loss_UL.txt', 'a') as f:
        f.write('%.2e \n' % avg_train_loss_UL)
      with open('Train_loss_FK.txt', 'a') as f:
        f.write('%.2e \n' % avg_train_loss_FK)
      with open('Train_loss_FM.txt', 'a') as f:
        f.write('%.2e \n' % avg_gen_FMloss)
    return


  def validate(self,step):
      #在训练一定的次数之后，执行一次验证
      total_val_loss = 0
      total_idx = 0
      IoUs = np.array([])
      DSs = np.array([])#dice score
      CT_fns, label_fns = get_files(F.test_dir,do_shuffle=False)#从测试集中读取成对的文件名列表
      for CT_fn,label_fn in zip( CT_fns, label_fns ):
          image,mask = read_npy_file(CT_fn,label_fn,do_augment=False,do_patch=False)
          start_time = time.time()
          G_mask,sample_val_loss,sample_idx = self.traverse(image[np.newaxis,:,:,:,:],mask[np.newaxis,:,:,:,0])
          stop_time = time.time()
          print('traverse time:',stop_time-start_time)
          trueMask = np.array(mask).flatten().astype(int)
          predMask = np.array(G_mask).flatten().astype(int)
          
          IoUs = np.concatenate((IoUs,100*IntersectionoverUnion(trueMask.astype(int),predMask.astype(int),F.num_classes)))
          DSs = np.concatenate((DSs,100*DiceScore(trueMask.astype(int),predMask.astype(int),F.num_classes)))
          
          print('IoUs=',IoUs)
          print('Dice Scores=',DSs) 
          
          root_path = F.validate_output_save_root
          if not os.path.exists(root_path):
              os.makedirs(root_path)
          asd=np.concatenate((
                  image[np.newaxis,:,:,:,:],
                  mask[np.newaxis,:,:,:,:]/(F.num_classes-1)*255.0,
                  G_mask/(F.num_classes-1)*255.0),axis=0)
          save_image(np.squeeze(asd[:,20,:,:,0]), 
                     '{}/{}_pred_{}.png'.format(root_path, str(step), os.path.split(os.path.splitext(CT_fn)[0])[1]),
                     nrow=F.batch_size, padding=2
                     )
          
          total_val_loss += sample_val_loss
          total_idx += sample_idx
          
      avr_val_IoUs = IoUs.reshape([-1,F.num_classes]).mean(axis=0)
      avr_val_DSs = DSs.reshape([-1,F.num_classes]).mean(axis=0)
      avr_val_loss = total_val_loss/(total_idx*1.0)#在验证集上的平均损失函数
      return avr_val_loss,avr_val_IOUS,avr_val_DSs
    
    
  def traverse(self,image,mask):
      #用patch大小的小窗口遍历一张CT大图,预测每个patch的label,最后拼接成一张大的label(mask)
      imageShape = image.shape
      D,H,W = imageShape[1],imageShape[2],imageShape[3]
      d,h,w = self.patch_shape[0],self.patch_shape[1], self.patch_shape[2]
      
      x0list = np.arange(0,H-h,h//3*2) # 形成height方向的滑窗列表
      y0list = np.arange(0,W-w,w//3*2) # 形成width方向的滑窗列表
      z0list = np.arange(0,D-d,d//2) # 形成frame方向的滑窗列表
      
      if not x0list[-1] == H-h:
          x0list = np.append( x0list,H-h )    # 如果不整除，则最后一个滑窗取不等距到尽头
      if not y0list[-1] == W-w:
          y0list = np.append( y0list,W-w )    # 如果不整除，则最后一个滑窗取不等距到尽头
      if not z0list[-1] == D-d:
          z0list = np.append( z0list,D-d )    # 如果不整除，则最后一个滑窗取不等距到尽头
          
      traverse_mask = np.zeros( np.append(image.shape[0:-1],F.num_classes) )
      weight = np.zeros(imageShape)
       
     
      sample_val_loss = 0
      idx=0
      for x0,y0,z0 in product(x0list,y0list,z0list):
          patches_feed = image[:,z0:z0+d,x0:x0+h,y0:y0+w,:]
          labels_feed = mask[:,z0:z0+d,x0:x0+h,y0:y0+w]
          feed_dict={self.patches_lab:patches_feed, self.labels:labels_feed, self.phase:False}
          
          preds = self.Val_output.eval(feed_dict)
          val_loss = self.d_loss_lab.eval(feed_dict)
          
          traverse_mask[:,z0:z0+d,x0:x0+h,y0:y0+w,:] += convert_to_onehot(preds[:,:,:,:,np.newaxis],F.num_classes)
          weight[:,z0:z0+d,x0:x0+h,y0:y0+w,:] += 1
          sample_val_loss += val_loss
          idx += 1
          
      output_mask = np.argmax(traverse_mask,axis=4)[:,:,:,:,np.newaxis]
      return output_mask,sample_val_loss,idx