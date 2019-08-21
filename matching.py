# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:49:55 2019

@author: IITI204
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:27:30 2019

@author: Vijay

IITI_HRFC

"""




import keras
import os, glob
from keras import backend as k 
from keras import layers
from keras import models
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from scipy import io
from scipy.io import loadmat
from scipy.io import savemat
#from keras.optimizers import SGD
from scipy.spatial import distance
import numpy as np
import h5py
import cv2

import tensorflow as tf 
import keras
print(tf.__version__, keras.__version__)
import hdf5storage

#from keras.layers import LeakyReLU
#from keras.layers import InputLayer
#from keras import layers
#from keras import models
from keras.models import load_model
from keras.utils import CustomObjectScope


##model
def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask

def batch_hard_triplet_loss(y1, y2):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,) --y_ture--y1
        embeddings: tensor of shape (batch_size, embed_dim)---y-pred--y2
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    margin=0.8
    squared=False
    
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(y2, squared=False)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    labels=tf.squeeze(y1,axis=-1) # axis=-1 was there
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
#    triplet_loss = tf.reduce_mean(triplet_loss)
    
#    triplet_loss = tf.ones([batch_size, 1]) * triplet_loss
    
    

    return triplet_loss



 

#def dice(y_true, y_pred):
#   return tf.contrib.losses.metric_learning.triplet_semihard_loss( labels=y_true, embeddings=y_pred, margin=1.0)
 

with CustomObjectScope({'batch_hard_triplet_loss': batch_hard_triplet_loss,'k':k}):
     model = load_model('new_resnet_with_3_conv_level_FV_128_lr_e-4_BS_256_batch_hard_triplet_loss_margin_0.8_epoch_100_train_815982_val_8760_custom_val.h5')
     
     
# load the image names of DBI
#pore_name=loadmat('name_pore.mat');   # this is dict

# read the dict corresponding to the name key

#pore_name=pore_name['name2'];  # to extract location pore_name[0][0][0], last zero remain fixed


# load pore coordinates
#pore_map1=hdf5storage.loadmat('pore_map_complete_6400.mat')['complete_poreMap'];
#pore_map1=hdf5storage.loadmat('complete_poremap_IITI_HRFP_only_poremap_th_0.2_w_5.mat')['complete_poreMap'];
pore_map=loadmat('complete_pore_IITI_HRFC_800x8.mat')['IITI_HRFC_Pore']  # old pore map
#pore_map = loadmat('IITI_HRFC_FT_th_0.1_w_5.mat')['Pore_IIT_HRFC_FT']  # new pore map with fine tunning
#PORE_MAP=np.zeros((800 ,8),dtype=object);
#
#i_range=range(0,800)
#j_range=range(0,8)
#count=0;
#for i in i_range:
#    for j in j_range:
#        PORE_MAP[i, j]=pore_map1[0][count];
#        count+=1;

Score_DBI=[]
for ii in range(1,801):
    for jj in range(1,9):

     path='C:\\college_pc\\fingerprint_data\\Final_IITI\\IITI_6400_800x800\\%d_%d.jpg' %(ii,jj)
     print(path)
     img=cv2.imread(path,0);
     ii_map=ii-1
     jj_map=jj-1

     xx=pore_map[ii_map][jj_map];
     # find the pore indices where it is one

#     inds = np.where(Pore_cord_1==1);
#     xx=inds[0];
#     yy=inds[1];

     wind=20;
     porepatch=[];
     for x in range(0,len(xx)): 
       pore_x=xx[x,0]-1;
       pore_y=xx[x,1]-1;
       if ((pore_x-wind)>=0 and (pore_x+wind)<=799 and (pore_y-wind)>=0 and (pore_y+wind)<=799):
         porepatch1=img[pore_x-wind : pore_x+wind+1, pore_y-wind :pore_y+wind+1];
         porepatch1 = porepatch1[:, :, np.newaxis]
         porepatch.append(porepatch1)
     porepatch=np.asarray(porepatch) 
     score= model.predict(porepatch, batch_size=64,verbose=0); 
#     score1=score[0];
     Score_DBI.append(score)
         
        


# code for genuine and impostor score


def pairwise_distances(x1, x2):
  # memory efficient implementation based on Yaroslav Bulatov's answer in
  # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
  sqr1 = np.sum(x1 * x1, axis=1, keepdims=True)
  sqr2 = np.sum(x2 * x2, axis=1)
  D = sqr1 - 2 * np.matmul(x1, x2.T) + sqr2

  return D

def find_correspondences(descs1,
                         descs2,
                         pts1=None,
                         pts2=None,
                         euclidean_weight=0,
                         transf=None,
                         thr=None):
  '''
  Finds bidirectional correspondences between descs1 descriptors and
  descs2 descriptors. If thr is provided, discards correspondences
  that fail a distance ratio check with threshold thr. If pts1, pts2,
  and transf are give, the metric considered when finding correspondences
  is
    d(i, j) = ||descs1(j) - descs2(j)||^2 + euclidean_weight *
      * ||transf(pts1(i)) - pts2(j)||^2

  Args:
    descs1: [N, M] np.array of N descriptors of dimension M each.
    descs2: [N, M] np.array of N descriptors of dimension M each.
    pts1: [N, 2] np.array of N coordinates for each descriptor in descs1.
    pts2: [N, 2] np.array of N coordinates for each descriptor in descs2.
    euclidean_weight: weight given to spatial constraint in comparison
      metric.
    transf: alignment transformation that aligns pts1 to pts2.
    thr: distance ratio check threshold.

  Returns:
    list of correspondence tuples (j, i, d) in which index j of
      descs2 corresponds with i of descs1 with distance d.
  '''
  # compute descriptors' pairwise distances
  D = pairwise_distances(descs1, descs2)

  # add points' euclidean distance
  if euclidean_weight != 0:
    assert transf is not None
    assert pts1 is not None
    assert pts2 is not None

    # assure pts are np array
    pts1 = transf(np.array(pts1))
    pts2 = np.array(pts2)

    # compute points' pairwise distances
    euclidean_D = pairwise_distances(pts1, pts2)

    # add to overral keypoints distance
    D += euclidean_weight * euclidean_D

  # find bidirectional corresponding points
  pairs = []
  if thr is None or len(descs1) == 1 or len(descs2) == 1:
    # find the best correspondence of each element
    # in 'descs2' to an element in 'descs1'
    corrs2 = np.argmin(D, axis=0)

    # find the best correspondence of each element
    # in 'descs1' to an element in 'descs2'
    corrs1 = np.argmin(D, axis=1)

    # keep only bidirectional correspondences
    for i, j in enumerate(corrs2):
      if corrs1[j] == i:
        pairs.append((j, i, D[j, i]))
  else:
    # find the 2 best correspondences of each
    # element in 'descs2' to an element in 'descs1'
    corrs2 = np.argpartition(D.T, [0, 1])[:, :2]

    # find the 2 best correspondences of each
    # element in 'descs1' to an element in 'descs2'
    corrs1 = np.argpartition(D, [0, 1])[:, :2]

    # find bidirectional corresponding points
    # with second best correspondence 'thr'
    # worse than best one
    for i, (j, _) in enumerate(corrs2):
      if corrs1[j, 0] == i:
        # discard close best second correspondences
        if D[j, i] < D[corrs2[i, 1], i] * thr:
          if D[j, i] < D[j, corrs1[j, 1]] * thr:
            pairs.append((j, i, D[j, i]))

  return pairs
     

 # genuine and impostor score for  DBI
#bf = cv2.BFMatcher();
#
#def match_pair_in(des1, des2):
#     matches = bf.knnMatch(des1,des2, k=2);
#     good = []
#     count1=0
#     for m,n in matches:
#        if m.distance < 0.9*n.distance:
#           good.append([m])
#           count1+=count1
#     good_len=len(good)
#     
#     return(good_len)

 
temp1=0;
gen_score=[];
count=1;
for i in range(0,800):
    print(i)
    for kk in range (4,8):
      for j in range (0,4):
        a2=Score_DBI[temp1+kk];
        a1=Score_DBI[temp1+j];
        Match_found=find_correspondences(a2,a1,thr=0.9);
        match=len(Match_found); 
        gen_score.append(match)
        count=count+1
    temp1=temp1+8;


## Imposter score generation
Imp_score=[];
temp_imp=4;
temp_imp1=9;
count_imp=1;
for ik in range(0, 800):
    a_imp=Score_DBI[temp_imp];
#    print('subject_count is', ik)
#%     a_sub=name{1,temp_imp};
#%     a_sub=strsplit(a_sub,'.');
#%     a_sub=strsplit(a_sub{1,1},'_');
#%     a_sub=a_sub{1,1};   % subject  index of imposter
    for jk in range(0,6400,8):
        if ((temp_imp-jk)!=4):
          print('subject_count is', ik, 'inside_count is', jk)
          b_imp=Score_DBI[jk];
          Match_imp=find_correspondences(a_imp,b_imp,thr=0.9);
          match_imp=len(Match_imp);
          Imp_score.append(match_imp);
          
#%           Imp_score_euc{count_imp,1}=D_imp;
#%           Imp_score{count_imp,2}=N;
#%           Imp_score{count_imp,3}=0;
#%           Imp_score{count_imp,3}=[N1;N2;I2];
        count_imp=count_imp+1
        temp_imp1=temp_imp1+10;        
    temp_imp=temp_imp+8;
    
    

# define ROC
def roc(pos, neg):
  '''
  Computes Receiver Operating Characteristic curve for given comparison scores.

  Args:
    pos: scores of genuine comparisons.
    neg: scores of impostor comparisons.

  Returns:
    fars: False Acceptance Rates (FARs) over all possible thresholds.
    frrs: False Rejection Rates (FRRs) over all possible thresholds.
  '''
  # sort comparisons arrays for efficiency
  pos = sorted(pos, reverse=True)
  neg = sorted(neg, reverse=True)

  # get all scores
  scores = list(pos) + list(neg)
  scores = np.unique(scores)

  # iterate to compute statistsics
  fars = [0.0]
  frrs = [1.0]
  pos_cursor = 0
  neg_cursor = 0
  for score in reversed(scores):
    # find correspondent positive score
    while pos_cursor < len(pos) and pos[pos_cursor] > score:
      pos_cursor += 1

    # find correspondent negative score
    while neg_cursor < len(neg) and neg[neg_cursor] > score:
      neg_cursor += 1

    # compute metrics for score
    far = neg_cursor / len(neg)
    frr = 1 - pos_cursor / len(pos)

    # add to overall statisics
    fars.append(far)
    frrs.append(frr)

  # add last step
  fars.append(1.0)
  frrs.append(0.0)

  return fars, frrs

## define EER
def eer(pos, neg):
  '''
  Computes the Equal Error Rate of given comparison scores.
  If FAR and FRR crossing is not exact, lineary interpolate the ROC and
  compute its intersection with the identity line f(x) = x.

  Args:
    pos: scores of genuine comparisons.
    neg: scores of impostor comparisons.

  Returns:
    EER of comparisons.
  '''
  print(len(pos))
  print(len(neg))
  
  # compute roc curve
  fars, frrs = roc(pos, neg)

  # iterate to find equal error rate
  old_far = None
  old_frr = None
  for far, frr in zip(fars, frrs):
    # if crossing happened, eer is found
    if far >= frr:
      break
    else:
      old_far = far
      old_frr = frr

  # if crossing is precisely found, return it
  # otherwise, approximate it though ROC linear
  # interpolation and intersection with f(x) = x
  if far == frr:
    return far
  else:
    return (far * old_frr - old_far * frr) / (far - old_far - (frr - old_frr))    

  

EER_IITI_HRFP = eer(gen_score, Imp_score)*100;
print(EER_IITI_HRFP)  

## this is when data is in sequence 601-602-603---800 IITI
#test_img_data=[]
#test_label=[]
#filenames=[]
#temp=0
#nbrofimages =800
#for x in range(1, 800+1):
#    
#   
#        for jj in range (1,9): 
#    
#         path = 'C:\\college_pc\\fingerprint_data\\Final_IITI\\Partial_train_test\\complete\\%d_%d.jpg' %( x,jj)
#         str_split=path.split('\\')
#         filenames.append(str_split[-1])
#         str_end=str_split[-1].split('.')
#         str_final=str_end[0].split('_')
#         test_label.append(int(str_final[0]))
#         temp=temp+1
#     
#         n=cv2.imread(path,0)  # where 1 is for RGB the channel
#         clahe = cv2.createCLAHE()
#         cl1 = clahe.apply(n)
#         n2=cv2.resize(cl1,(224,224))
##        n=n/256
#         n2 = n2[:, :, np.newaxis]
#         test_img_data.append(n2)
#    
#test_data=np.asarray(test_img_data) 


### for any random data DBII
#imgnames = sorted(glob.glob("D:\\veri_out\\DBI_content\\Test\\*.jpg"))
#
#
#
#test_img_data=[]
#test_label=[]
#filenames=[]
#temp=0
#
#for imgname in imgnames:
#    
#    #make sure path is correct.
#      
#     str_split=imgnames[temp].split('\\')
#     str_end=str_split[-1].split('.')
#     str_final=str_end[0].split('_')
#     test_label.append(int(str_final[0]))
#     temp=temp+1
#
#     n=cv2.imread(imgname,0)  # where 1 is for RGB the channel
##     n=np.transpose(n, axes=None);
#     clahe = cv2.createCLAHE()
#     cl1 = clahe.apply(n)
#     n2=cv2.resize(cl1,(224,224))
#     n2 = n2[:, :, np.newaxis]
#     test_img_data.append(n2)
#    
#test_data=np.asarray(test_img_data) 
#     
#Train_final=train_data[0:4000]
#Train_final_label=train_label[0:4000]
#
#val_final=train_data[4000:]
#val_final_label=train_label[4000:]



#filepath2 = 'C:/Users/Admin/Desktop/New folder/Test4d_label_2lakh.mat'
#arrays_test = {}
#f2 = h5py.File(filepath2)
#for k, v in f2.items():
#    arrays_test[k] = np.array(v)
#
#test_label=arrays_test['Test4d_label']


#test_final=np.reshape(test_data,[1600,224,224,1]);
#
#test_final_label=np.reshape(test_label,[42066,80,80,1]);
#

#score1=np.reshape(score,[224,224,1,1600]);
#savemat('D:\\python_practice_code\\pore_resnet_code', dict(score1=score1))

## code for accuracy score genuine and imposter by considering 3 template and 5 test image

#score_dist=score[0];  # this holds the ouput as feature vector
