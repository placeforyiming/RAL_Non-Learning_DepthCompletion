from data_read import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tools import *
from evaluation import *
import time

evaluate=Result()
rmse_total=0
mae_total=0
irmse_total=0
imae_total=0
#validation
A=calculate_normal()




kernelx=self_gaussian(kernel_size=7,g_range=2.5)
time_interval=[]
threshold=0.1
for i in range(1000):

    print (i)
    img,lidar,gt,intrinsic= read_one_val(i)

    # prepare camera parameters
    px,py,fx,fy=construct_px_py_fx_fy(lidar,intrinsic)
    # remove outliers
    if i>20:
        time_a=time.time()


    lidar_new=outlier_removal(lidar)

    # get all points
    total_points,x_indices,y_indices,width_image,height_image=get_all_points(lidar_new,intrinsic)

    # map points on range image
    range_image,proj_xyz,proj_idx,proj_mask=do_range_projection(total_points)
    # fill in range image
    range_image_new=fill_spherical(range_image)
    # calculate normal in spherical
    normal_spherical=A.calculate_normal(range_image_new)
    # put normal from spherical on original image
    normal_img=put_normal_on_image(normal_spherical,x_indices,y_indices,proj_idx,proj_mask)
    a=normal_img[:,:,0]
    b=normal_img[:,:,1]
    c=normal_img[:,:,2]
    

    depth_map,a_list_all,b_list_all,c_list_all,height_list_offset,width_list_offset=Distance_Transform(lidar_new,a,b,c,width_image,height_image)
    
    upper=(width_list_offset/fx*a_list_all+height_list_offset/fy*b_list_all)
    lower=(width_image-px)/fx*a_list_all+(height_image-py)/fy*b_list_all+c_list_all

    lower[np.logical_and(np.abs(lower)<threshold,lower>0)]=threshold
    lower[np.logical_and(np.abs(lower)<threshold,lower<0)]=-threshold
    residual=upper/lower

   # residual[residual>0.2]=0.2
   # residual[residual<-0.2]=-0.2
    
    #print (np.max(residual))
    #print (np.min(residual))
    #print (np.mean(residual))
    depth_predicted=depth_map+residual*depth_map
    
    #depth_predicted= cv2.GaussianBlur(depth_predicted,(3,3),0)
    
    depth_predicted = cv2.filter2D(depth_predicted, -1, kernelx)

    depth_predicted=np.squeeze(lidar_new)+depth_predicted*(1.0-np.squeeze(lidar_new)>0.01)
    depth_predicted[depth_predicted<0.9]=0.9
    depth_predicted[depth_predicted>80.0]=80.0
    

    #depth_predicted = cv2.filter2D(depth_predicted, -1, kernelx)
    if i>20:
      time_b=time.time()
      time_interval.append(time_b-time_a)
    
    
    #depth_predicted= cv2.GaussianBlur(depth_predicted,(3,3),0)
    
  
    
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
