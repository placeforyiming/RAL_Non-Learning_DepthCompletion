import tensorflow as tf
import numpy as np
import csv
import os
import cv2
from net import *
from data_read import *
from data_read import *
from evaluation import *


import argparse







parser = argparse.ArgumentParser()

parser.add_argument('--model_name', action="store", dest= "model_name",default="ResNet18_large",help='ResNet18 or ResNet18_large or SparsityCNN or NormalizeConv')

parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.0001,help='learning_rate')



parser.add_argument('--epoch_num', action="store", dest="epoch_num", type=int, default=8,help='how many epochs to eval')


parser.add_argument('--if_removal', action="store", dest="if_removal", type=bool, default=True,help='.if_removal')


parser.add_argument('--model_type', action="store", dest="model_type", type=str, default="DT",help='baseline or DT')



input_parameters = parser.parse_args()



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


def nearest_point(refined_lidar):   
    value_mask=np.asarray(1.0-np.squeeze(refined_lidar)>0.1).astype(np.uint8)
    dt,lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)
    return dt,lbl



def outlier_removal(lidar):
  # output 2 dimension image
    DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

    sparse_lidar=np.squeeze(lidar)
    valid_pixels = (sparse_lidar > 0.1).astype(np.float)
    
    
    lidar_sum=cv2.filter2D(sparse_lidar,-1,DIAMOND_KERNEL_7)
    
    lidar_count=cv2.filter2D(valid_pixels,-1,DIAMOND_KERNEL_7)
    
    lidar_aveg=lidar_sum/(lidar_count+0.00001)
    
    potential_outliers=((sparse_lidar-lidar_aveg)>1.0).astype(np.float)
    
    
    return  (sparse_lidar*(1-potential_outliers)).astype(np.float32)

def outlier_removal_batch(lidar_batch,if_removal=True):
    batch_size=np.shape(lidar_batch)[0]
    new_batch=[]
    new_batch_width_offset=[]
    new_batch_height_offset=[]
    for i in range(batch_size):
        if if_removal:
            lidar_single=outlier_removal(lidar_batch[i,:,:])
        else:
            lidar_single=lidar_batch[i,:,:,0]
                

        dt,lbl=nearest_point(lidar_single)
        with_value=np.squeeze(lidar_single)>0.1
        
        height_index=[j for j in range(352)]
        height_index=np.reshape(height_index,[352,1])
        height_index=np.tile(height_index,[1,1216])
        
        width_index=[j for j in range(1216)]
        width_index=np.reshape(width_index,(1,1216))
        width_index=np.tile(width_index,[352,1])
        

        depth_list=np.squeeze(lidar_single)[with_value]
        height_list=height_index[with_value]
        width_list=width_index[with_value]
        
        label_list=np.reshape(lbl,[1,1216*352])
        depth_list_all=depth_list[label_list-1]
        height_list_all= height_list[label_list-1]
        width_list_all= width_list[label_list-1]
        
        height_list_all=np.reshape(height_list_all,(352,1216))
        width_list_all=np.reshape(width_list_all,(352,1216))
        depth_map=np.reshape(depth_list_all,(352,1216))
        

        height_list_offset=height_list_all-height_index
        width_list_offset=width_list_all-width_index
        

        new_batch.append(depth_map)
        new_batch_width_offset.append(width_list_offset)
        new_batch_height_offset.append(height_list_offset)

    new_batch=np.asarray(new_batch)
    new_batch=np.expand_dims(new_batch,axis=-1)
    new_batch_width_offset=np.asarray(new_batch_width_offset)
    new_batch_width_offset=np.expand_dims(new_batch_width_offset,axis=-1)
    new_batch_height_offset=np.asarray(new_batch_height_offset)
    new_batch_height_offset=np.expand_dims(new_batch_height_offset,axis=-1)


    refined_lidar=new_batch.astype(np.float32)
    width_offset=new_batch_width_offset.astype(np.float32)
    height_offset=new_batch_height_offset.astype(np.float32)

    final_input=np.concatenate((refined_lidar,width_offset),axis=-1)
    final_input=np.concatenate((final_input,height_offset),axis=-1)
    return final_input


save_path='./checkpoints/'+input_parameters.model_name+'/'+input_parameters.model_type+'/'






if not(os.path.exists('./checkpoints')):
    os.mkdir('./checkpoints')
if not(os.path.exists('./checkpoints/'+input_parameters.model_name)):
    os.mkdir('./checkpoints/'+input_parameters.model_name)
if not(os.path.exists('./checkpoints/'+input_parameters.model_name+'/'+input_parameters.model_type)):
    os.mkdir('./checkpoints/'+input_parameters.model_name+'/'+input_parameters.model_type)


if not(input_parameters.model_type=="baseline"):
	save_path=save_path+str(input_parameters.if_removal)+'/'
	if not(os.path.exists(save_path)):
		os.mkdir(save_path)

lr=input_parameters.learning_rate



if input_parameters.model_name=='ResNet18':
   
    depth_network=ResNet18()
    
if input_parameters.model_name=='ResNet18_large':
   
    depth_network=ResNet18_large()

    





#load weights
depth_network.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_num)+"_full")



evaluate=Result()
rmse_total=0
mae_total=0
irmse_total=0
imae_total=0

for i in range(1000):

    print (i)
    img,lidar,gt,intrinsic= read_one_val(i)

    lidar_new=outlier_removal(lidar)

    if not(input_parameters.model_type=="baseline"):
        if input_parameters.if_removal:
            lidar=outlier_removal_batch(lidar,if_removal=True) 
        else:
            lidar=outlier_removal_batch(lidar,if_removal=False)

    lidar = lidar[:,96:,:,:]
    gt = gt[:,96:,:]

    with_gt=(gt>0.1)
    total_value=np.sum(with_gt)
    with_gt=tf.dtypes.cast(with_gt,tf.float32)

    depth_predicted=depth_network.call(lidar,training=False)

    
    if i==1:
        total_parameters=0
        all_variables=depth_network.trainable_variables
        for mm in all_variables:
            if len(np.shape(mm))>1:
                total_parameters+=np.shape(mm)[0]*np.shape(mm)[1]*np.shape(mm)[2]*np.shape(mm)[3]
            else:
                total_parameters+=np.shape(mm)[0]
        print (total_parameters)

    depth_predicted=depth_predicted+lidar[:,:,:,:1]

    depth_predicted=np.squeeze(depth_predicted).astype(np.float32)
    

    depth_predicted=np.squeeze(lidar_new)[96:,:]+depth_predicted*(1.0-np.squeeze(lidar_new)[96:,:]>0.01)
    

    depth_predicted[depth_predicted<0.9]=0.9
    depth_predicted[depth_predicted>80.0]=80.0
    evaluate.evaluate(np.squeeze(depth_predicted),np.squeeze(gt))

    irmse, imae, mse, rmse, mae=evaluate.irmse, evaluate.imae, evaluate.mse, evaluate.rmse, evaluate.mae
    rmse_total+=rmse
    mae_total+=mae
    irmse_total+=irmse
    imae_total+=imae
    print ("rmse:")
    print (rmse_total/(i+1))
    print ("mae:")
    print (mae_total/(1+i))
    print ("irmse:")
    print (irmse_total/(1+i))
    print ("imae:")
    print (imae_total/(1+i))

print ("rmse:")
print (rmse_total/1000.0)
print ("mae:")
print (mae_total/1000.0)      
print ("irmse:")
print (irmse_total/1000.0)      
print ("imae:")
print (imae_total/1000.0)       

print ('method time')
print (np.sum(time_interval)/len(time_interval))
