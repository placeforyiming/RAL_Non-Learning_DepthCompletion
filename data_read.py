import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import tensorflow as tf
def get_paths_and_transform():
    root_d = os.path.join('./depth_selection/KITTI/Sparse_Lidar')
    root_rgb = os.path.join('./depth_selection/KITTI/RGB')
    glob_sparse_lidar = "train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png"

    glob_sparse_lidar = os.path.join(root_d,glob_sparse_lidar)
    all_lidar_path_with_new=glob.glob(glob_sparse_lidar)
    all_lidar_path_without_new=[i for i in all_lidar_path_with_new if not (('left' in i) or('right' in i)) ]
    paths_sparse_lidar = sorted(all_lidar_path_without_new)
    def get_rgb_paths(p):
        ps = p.split('/')
        pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
        return pnew
    
    glob_rgb = [get_rgb_paths(i) for i in paths_sparse_lidar]
    return paths_sparse_lidar,glob_rgb


def img_path_to_lidar(img_path):
    #img_path:'./Dataset/KITTI/RGB/train/2011_09_26_drive_0051_sync/image_02/data/0000000432.png'
    path_list=img_path.split('/')
    return path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+'Sparse_Lidar/'+path_list[4]+'/'+path_list[5]+'/proj_depth/velodyne_raw/'+path_list[6]+'/'+path_list[8]

def img_path_to_ground_truth(img_path):
    #img_path:'./Dataset/KITTI/RGB/train/2011_09_26_drive_0051_sync/image_02/data/0000000432.png'
    path_list=img_path.split('/')
    return path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+'ground_truth/'+path_list[4]+'/'+path_list[5]+'/proj_depth/groundtruth/'+path_list[6]+'/'+path_list[8]



    
    
def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    img_file.close()
    return rgb_png

def depth_new_read(filename):
    depth=io.imread(filename)
    depth=depth/255.0*100
    return depth


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.

    depth  = np.array(Image.fromarray(depth).resize((1216,352), Image.NEAREST))

    depth = np.expand_dims(depth,-1)
    return depth



lidar_path,img_path=get_paths_and_transform()
total_img=len(img_path)
total_lidar=len(lidar_path)

class Data_load():
    def __init__(self):
        lidar_path,img_path=get_paths_and_transform()
        self.lidar_path=lidar_path
        self.img_path=img_path
        self.num_sample=[i for i in range(len(self.img_path))]
        np.random.shuffle(self.num_sample)
        self.index=0
        self.total_sample=len(self.img_path)
       
    def read_batch(self,batch_size=4):
        i=0
        img_batch=[]
        lidar_batch=[]
        gt_batch=[]

        while (i<(batch_size)):
            i=i+1
            
            #img=rgb_read(img_path[index])

            depth=depth_read(self.lidar_path[ self.num_sample[self.index]])
            gt_path=img_path_to_ground_truth(self.img_path[self.num_sample[self.index]])
            ground_truth=depth_read(gt_path)

            lidar_batch.append(depth)
            #img_batch.append(img)
            gt_batch.append(ground_truth)
            self.index=self.index+1
        if self.index+batch_size>self.total_sample:
            return [0],[1]
        else:
            return  np.asarray(lidar_batch),np.asarray(gt_batch)#np.asarray(img_batch),

        


    

def read_batch(batch_size=4):
    i=0
    img_batch=[]
    lidar_batch=[]
    gt_batch=[]
    
    while (i<(batch_size)):
        i=i+1
        index = random.randint(0,(total_img-1))
        #img=rgb_read(img_path[index])
        depth=depth_read(lidar_path[index])
        gt_path=img_path_to_ground_truth(img_path[index])
        ground_truth=depth_read(gt_path)
        
        lidar_batch.append(depth)
        #img_batch.append(img)
        gt_batch.append(ground_truth)
    return  np.asarray(lidar_batch),np.asarray(gt_batch)

def read_batch_v2(batch_size=4, index=0):
    i=0
    img_batch=[]
    lidar_batch=[]
    gt_batch=[]
    
    while (i<(batch_size)):
        i=i+1
        depth=depth_read(lidar_path[index])
        gt_path=img_path_to_ground_truth(img_path[index])
        ground_truth=depth_read(gt_path)
        
        lidar_batch.append(depth)
        gt_batch.append(ground_truth)
    return  np.asarray(lidar_batch),np.asarray(gt_batch)

    

def read_one_val(index,line_number=64):
    ground_truth_path='./depth_selection/val_selection_cropped/groundtruth_depth'
    if line_number==64:
        velodyne_raw_path='./depth_selection/val_selection_cropped/velodyne_raw'
    if line_number==32:
        velodyne_raw_path='./depth_selection/val_selection_cropped/velodyne_raw_32'
    if line_number==16:
        velodyne_raw_path='./depth_selection/val_selection_cropped/velodyne_raw_16'
    intrinsics_path='./depth_selection/val_selection_cropped/intrinsics'
    image_path='./depth_selection/val_selection_cropped/image'
    instance_path='./depth_selection/val_selection_cropped/instance'


    ground_truth=os.listdir('./depth_selection/val_selection_cropped/groundtruth_depth')
    image=os.listdir('./depth_selection/val_selection_cropped/image')
    velodyne_raw=os.listdir('./depth_selection/val_selection_cropped/velodyne_raw')
    intrinsics=os.listdir('./depth_selection/val_selection_cropped/intrinsics')
    
    i=image[index]
    img_one=[]
    lidar_one=[]
    intrinsics_matrix=[]
    ground_thuth_one=[]
    

    
    img_file = Image.open(image_path+'/'+i)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    img_file.close()
    img=rgb_png
    
    img_file = Image.open(velodyne_raw_path+  '/'+i[:27]+'velodyne_raw'+i[32:])
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    depth = depth_png.astype(np.float32) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth,-1)

    temp_i=i
    ii=temp_i.replace("image","predicted")
    img_file = Image.open(instance_path+  '/'+ii)
    instance_png = np.array(img_file, dtype=int)
    img_file.close()
    instance_png = np.expand_dims(instance_png,-1)
    
    
    img_file = Image.open(ground_truth_path+  '/'+i[:27]+'groundtruth_depth'+i[32:])
    ground_truth = np.array(img_file, dtype=int)
    img_file.close()
    ground_truth = ground_truth.astype(np.float) / 256.

    F = open(intrinsics_path+'/'+i[:len(i)-4]+'.txt','r')
    intrinsics_matrix_per=F.readline().split(' ')

    intrinsics_matrix_per=[float(n) for n in intrinsics_matrix_per if not(n=='\n')]
    F.close()
    
    img_one.append(img)
    lidar_one.append(depth[:,:,0])
    ground_thuth_one.append(ground_truth)
    intrinsics_matrix.append(intrinsics_matrix_per)
        
    return  np.asarray(img_one),np.asarray(lidar_one),np.asarray(ground_thuth_one), np.asarray(intrinsics_matrix)#, np.asarray(instance_png)

def read_one_test(index):


    ground_truth_path='./depth_selection/test_depth_completion_anonymous/groundtruth_depth'
    velodyne_raw_path='./depth_selection/test_depth_completion_anonymous/velodyne_raw'
    intrinsics_path='./depth_selection/test_depth_completion_anonymous/intrinsics'
    image_path='./depth_selection/test_depth_completion_anonymous/image'
    image=os.listdir('./depth_selection/test_depth_completion_anonymous/image')
    velodyne_raw=os.listdir('./depth_selection/test_depth_completion_anonymous/velodyne_raw')
    intrinsics=os.listdir('./depth_selection/test_depth_completion_anonymous/intrinsics')
    
    i=image[index]
    img_one=[]
    lidar_one=[]
    intrinsics_matrix=[]
    ground_thuth_one=[]
    

    
    img_file = Image.open(image_path+'/'+i)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    img_file.close()
    img=rgb_png
    
    img_file = Image.open(velodyne_raw_path+  '/'+i[:27])
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth,-1)
    
    F = open(intrinsics_path+'/'+i[:len(i)-4]+'.txt','r')
    intrinsics_matrix_per=F.readline().split(' ')

    intrinsics_matrix_per=[float(n) for n in intrinsics_matrix_per if not(n=='\n')]
    F.close()
    
    intrinsics_matrix.append(intrinsics_matrix_per)
    img_one.append(img)
    lidar_one.append(depth[:,:,0])

    return  np.asarray(img_one),np.asarray(lidar_one),image[index], np.asarray(intrinsics_matrix)


class KITTI_demo_loader():
    
    def __init__(self):
        self.lidar_sequence=[]
        self.rgb_sequence=[]
        self.demo_KITTI()
        self.index=0

    
    def demo_KITTI(self):
        RGB_image_path=glob.glob('./depth_selection/KITTI/RGB/train/2011_09_26_drive_0039_sync/image_02/data/*.png')
        Lidar_image_path=glob.glob('./depth_selection/KITTI/Sparse_Lidar/train/2011_09_26_drive_0039_sync/proj_depth/*/image_02/*.png')
        for i in Lidar_image_path:
            temp=i.split('/')[-1]
            for j in RGB_image_path:
                if j.split('/')[-1]==temp:
                    self.lidar_sequence.append(i)
                    self.rgb_sequence.append(j)
        self.rgb_sequence.sort()
        self.lidar_sequence.sort()
                    
    def rgb_read(self,filename):
        assert os.path.exists(filename), "file not found: {}".format(filename)
        img_file = Image.open(filename)
        # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
        rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
        rgb_png =np.array(Image.fromarray(rgb_png).resize((1216,352), Image.NEAREST))
        img_file.close()
        return rgb_png
    
    def depth_read(self,filename):
        # loads depth map D from png file
        # and returns it as a numpy array,
        # for details see readme.txt
        assert os.path.exists(filename), "file not found: {}".format(filename)
        img_file = Image.open(filename)
        depth_png = np.array(img_file, dtype=int)
        img_file.close()
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255, \
            "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

        depth = depth_png.astype(np.float) / 256.
        # depth[depth_png == 0] = -1.

        depth  = np.array(Image.fromarray(depth).resize((1216,352), Image.NEAREST))

        depth = np.expand_dims(depth,-1)
        return depth
    
    def read_one_image(self):
        lidar_ind_path=self.lidar_sequence[self.index]
        rgb_ind_path=self.rgb_sequence[self.index]
        name_ind_path=lidar_ind_path.split('/')
        self.index=self.index+1
        gt_path=img_path_to_ground_truth(rgb_ind_path)
        return self.rgb_read(rgb_ind_path),self.depth_read(lidar_ind_path),self.depth_read(gt_path)