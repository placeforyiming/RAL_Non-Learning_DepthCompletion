from data_read import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tools import *
from evaluation import *
import time
import os

evaluate=Result()
rmse_total=0
mae_total=0
irmse_total=0
imae_total=0
#validation
A=calculate_normal()

if not os.path.exists("./output"):
    os.mkdir("./output")

kernelx=self_gaussian(kernel_size=7,g_range=2.5)
time_interval=[]
threshold=0.1
for i in range(1000):

    print (i)
    img,lidar,index,intrinsic= read_one_test(i)

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


    depth_predicted=depth_map+residual*depth_map
    
    
    depth_predicted = cv2.filter2D(depth_predicted, -1, kernelx)

    depth_predicted=np.squeeze(lidar_new)+depth_predicted*(1.0-np.squeeze(lidar_new)>0.01)
    depth_predicted[depth_predicted<0.9]=0.9
    depth_predicted[depth_predicted>80.0]=80.0
    
    
    depth_predicted=depth_predicted*256.0
    depth_predicted=depth_predicted.astype(np.uint16)
    im=Image.fromarray(depth_predicted)
    im.save("./output/"+index.split('/')[-1])