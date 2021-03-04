
import numpy as np
import cv2

def nearest_point(refined_lidar):   
    value_mask=np.asarray(1.0-np.squeeze(refined_lidar)>0.1).astype(np.uint8)
    dt,lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)
    return dt,lbl


def Distance_Transform_Normal(lidar,a,b,c,width_image,height_image):
    # a,b,c are sparser than lidar
    lidar=np.squeeze(lidar)
    height,width=np.shape(lidar)
    mask_normal=np.logical_and(np.logical_and(a==0,b==0),c==0)

    mask_lidar=lidar>0.001
    lidar[np.logical_and(mask_normal,mask_lidar)]=0.0
    mask_lidar=lidar>0.001
    mask_normal=np.logical_and(np.logical_and(a==0,b==0),c==0)
    
    dt,lbl=nearest_point(lidar)
    with_value=mask_lidar
    
    depth_list=np.squeeze(lidar[with_value])
    height_list=height_image[with_value]
    width_list=width_image[with_value]


    a_list=a[with_value]
    b_list=b[with_value]
    c_list=c[with_value]
    
    
    label_list=np.reshape(lbl,[1,height*width])
    depth_list_all=depth_list[label_list-1]
    height_list_all= height_list[label_list-1]
    width_list_all= width_list[label_list-1]
    a_list_all= a_list[label_list-1]
    b_list_all= b_list[label_list-1]
    c_list_all= c_list[label_list-1]
        
    height_list_all=np.reshape(height_list_all,(height,width))
    width_list_all=np.reshape(width_list_all,(height,width))
    a_list_all=np.reshape(a_list_all,(height,width))
    b_list_all=np.reshape(b_list_all,(height,width))
    c_list_all=np.reshape(c_list_all,(height,width))
    depth_map=np.reshape(depth_list_all,(height,width))
        
    height_list_offset=height_list_all-height_image
    width_list_offset=width_list_all-width_image
    return depth_map,a_list_all,b_list_all,c_list_all,height_list_offset,width_list_offset


def Distance_Transform_simple(lidar):
    # a,b,c are sparser than lidar
    lidar=np.squeeze(lidar)
    height,width=np.shape(lidar)
    
    mask_lidar=lidar>0.9
    
    value_mask=np.asarray(1.0-np.squeeze(mask_lidar)).astype(np.uint8)
    dt,lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)

    with_value=mask_lidar
    
    depth_list=np.squeeze(lidar[with_value])
    label_list=np.reshape(lbl,[1,height*width])
    depth_list_all=depth_list[label_list-1]
    depth_map=np.reshape(depth_list_all,(height,width))
        
    return depth_map


def self_gaussian(kernel_size=7,g_range=3):

    DIAMOND_KERNEL_9 = np.asarray(
    [
        [0,0, 0, 0, 1, 0, 0, 0,0],
        [0,0, 0, 1, 1, 1, 0, 0,0],
        [0,0, 1, 1, 1, 1, 1, 0,0],
        [0,1, 1, 1, 1, 1, 1, 1,0],
        [1,1, 1, 1, 1, 1, 1, 1,1],
        [0,1, 1, 1, 1, 1, 1, 1,0],
        [0,0, 1, 1, 1, 1, 1, 0,0],
        [0,0, 0, 1, 1, 1, 0, 0,0],
        [0,0, 0, 0, 1, 0, 0, 0,0],
    ], dtype=np.uint8)

    

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
    
    DIAMOND_KERNEL_5 = np.asarray(
    [
        [ 0, 0, 1, 0, 0],
        [ 0, 1, 1, 1, 0],
        [ 1, 1, 1, 1, 1],
        [ 0, 1, 1, 1, 0],
        [ 0, 0, 1, 0, 0],
    ], dtype=np.uint8)
        
    if kernel_size==7:
        kernel=DIAMOND_KERNEL_7 
    if kernel_size==9:
        kernel=DIAMOND_KERNEL_9
    
    if kernel_size==5:
        kernel=DIAMOND_KERNEL_5 
    x, y = np.meshgrid(np.linspace(-g_range,g_range,kernel_size), np.linspace(-g_range,g_range,kernel_size))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    result=g*kernel
    return result/np.sum(result)
def Distance_Transform(lidar,a,b,c,width_image,height_image):
    # a,b,c are sparser than lidar
    lidar=np.squeeze(lidar)
    height,width=np.shape(lidar)
    mask_normal=np.logical_and(np.logical_and(a==0,b==0),c==0)

    mask_lidar=lidar>0.001
    c[np.logical_and(mask_normal,mask_lidar)]=1.0
    mask_normal=np.logical_and(np.logical_and(a==0,b==0),c==0)
    
    dt,lbl=nearest_point(lidar)
    with_value=mask_lidar
    
    depth_list=np.squeeze(lidar[with_value])
    height_list=height_image[with_value]
    width_list=width_image[with_value]


    a_list=a[with_value]
    b_list=b[with_value]
    c_list=c[with_value]
    
    
    label_list=np.reshape(lbl,[1,height*width])
    depth_list_all=depth_list[label_list-1]
    height_list_all= height_list[label_list-1]
    width_list_all= width_list[label_list-1]
    a_list_all= a_list[label_list-1]
    b_list_all= b_list[label_list-1]
    c_list_all= c_list[label_list-1]
        
    height_list_all=np.reshape(height_list_all,(height,width))
    width_list_all=np.reshape(width_list_all,(height,width))
    a_list_all=np.reshape(a_list_all,(height,width))
    b_list_all=np.reshape(b_list_all,(height,width))
    c_list_all=np.reshape(c_list_all,(height,width))
    depth_map=np.reshape(depth_list_all,(height,width))
        
    height_list_offset=height_list_all-height_image
    width_list_offset=width_list_all-width_image
    return depth_map,a_list_all,b_list_all,c_list_all,height_list_offset,width_list_offset
        
    
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

    DIAMOND_KERNEL_9 = np.asarray(
    [
        [0,0, 0, 0, 1, 0, 0, 0,0],
        [0,0, 0, 1, 1, 1, 0, 0,0],
        [0,0, 1, 1, 1, 1, 1, 0,0],
        [0,1, 1, 1, 1, 1, 1, 1,0],
        [1,1, 1, 1, 1, 1, 1, 1,1],
        [0,1, 1, 1, 1, 1, 1, 1,0],
        [0,0, 1, 1, 1, 1, 1, 0,0],
        [0,0, 0, 1, 1, 1, 0, 0,0],
        [0,0, 0, 0, 1, 0, 0, 0,0],
    ], dtype=np.uint8)

    
    sparse_lidar=np.squeeze(lidar)
    valid_pixels = (sparse_lidar > 0.1).astype(np.float)
    
    
    lidar_sum=cv2.filter2D(sparse_lidar,-1,DIAMOND_KERNEL_7)
    
    lidar_count=cv2.filter2D(valid_pixels,-1,DIAMOND_KERNEL_7)
    
    lidar_aveg=lidar_sum/(lidar_count+0.00001)
    
    potential_outliers=((sparse_lidar-lidar_aveg)>1.0).astype(np.float)
    
    
    return  (sparse_lidar*(1-potential_outliers)).astype(np.float32)

def construct_px_py_fx_fy(lidar,intrinsic):
    px=intrinsic[0,2]
    py=intrinsic[0,5]
    fx=intrinsic[0,0]
    fy=intrinsic[0,4]
    lidar_32=np.squeeze(lidar).astype(np.float32)
    height,width=np.shape(lidar_32)
    z_image=np.ones((height,width))
    px_matrix=z_image*px
    py_matrix=z_image*py
    fx_matrix=z_image*fx
    fy_matrix=z_image*fy
    return px_matrix,py_matrix,fx_matrix,fy_matrix

def get_all_points(lidar,intrinsic):

    
    lidar_32=np.squeeze(lidar).astype(np.float32)
    height,width=np.shape(lidar_32)
    x_axis=[i for i in range(width)]
    x_axis=np.reshape(x_axis,[width,1])
    x_image=np.tile(x_axis, height)
    x_image=np.transpose(x_image)
    y_axis=[i for i in range(height)]
    y_axis=np.reshape(y_axis,[height,1])
    y_image=np.tile(y_axis, width)
    z_image=np.ones((height,width))
    image_coor_tensor=[x_image,y_image,z_image]
    image_coor_tensor=np.asarray(image_coor_tensor).astype(np.float32)
    image_coor_tensor=np.transpose(image_coor_tensor,[1,0,2])

    intrinsic=np.reshape(intrinsic,[3,3]).astype(np.float32)
    intrinsic_inverse=np.linalg.inv(intrinsic)
    points_homo=np.matmul(intrinsic_inverse,image_coor_tensor)

    lidar_32=np.reshape(lidar_32,[height,1,width])
    points_homo=points_homo*lidar_32
    extra_image=np.ones((height,width)).astype(np.float32)
    extra_image=np.reshape(extra_image,[height,1,width])
    points_homo=np.concatenate([points_homo,extra_image],axis=1)

    extrinsic_v_2_c=[[0.007,-1,0,0],[0.0148,0,-1,-0.076],[1,0,0.0148,-0.271],[0,0,0,1]]
    extrinsic_v_2_c=np.reshape(extrinsic_v_2_c,[4,4]).astype(np.float32)
    extrinsic_c_2_v=np.linalg.inv(extrinsic_v_2_c)
    points_lidar=np.matmul(extrinsic_c_2_v,points_homo)


    mask=np.squeeze(lidar)>0.1
    total_points=[points_lidar[:,0,:][mask],points_lidar[:,1,:][mask],points_lidar[:,2,:][mask]]
    total_points=np.asarray(total_points)
    total_points=np.transpose(total_points)
    
    return total_points,x_image[mask],y_image[mask],x_image,y_image



def do_range_projection(points,proj_H=96,proj_W=2048,fov_up=3.0,fov_down=-18.0):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    #Now the settings are specilized for KITTI
    # projected range image - [H,W] range (-1 is no data)
    proj_range = np.full((proj_H, proj_W), -1,dtype=np.float32)

    # unprojected range (list of depths for each point)
    unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    proj_xyz = np.full((proj_H, proj_W, 3), -1,dtype=np.float32)

   

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
    proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    proj_mask = np.zeros((proj_H, proj_W),dtype=np.int32) 
    
    
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)


    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W                              # in [0.0, W]
    proj_y *= proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    #self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    #self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    #self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = points[order]
    #remission = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = points
    
    proj_idx[proj_y, proj_x] = indices
    proj_mask = (proj_idx > 0).astype(np.float32)
    return proj_range[:,790:1260], proj_xyz[:,790:1260],proj_idx[:,790:1260],proj_mask[:,790:1260]


def fill_spherical(range_image):
    # fill in spherical image for calculating normal vector
    height,width=np.shape(range_image)[:2]
    value_mask=np.asarray(1.0-np.squeeze(range_image)>0.1).astype(np.uint8)
    dt,lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)

    with_value=np.squeeze(range_image)>0.1
        
    depth_list=np.squeeze(range_image)[with_value]
        
    label_list=np.reshape(lbl,[1,height*width])
    depth_list_all=depth_list[label_list-1]

    depth_map=np.reshape(depth_list_all,(height,width))
    
    depth_map = cv2.GaussianBlur(depth_map,(7,7),0)
    #depth_map=range_image*with_value+depth_map*(1-with_value)
    return depth_map



class calculate_normal():
    def __init__(self,proj_H=96,proj_W=2048,fov_up=3.0,fov_down=-18.0,W_start=790,W_end=1260):
        self.proj_H=proj_H
        self.proj_W=proj_W
        self.fov_up=fov_up
        self.fov_down=fov_down
        self.W_start=W_start
        self.W_end=W_end
        
        fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = fov_down  / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up) 
        
        zero_matrix=np.zeros((self.proj_H,self.W_end-self.W_start))
        one_matrix=np.ones((self.proj_H,self.W_end-self.W_start))
        self.theta_channel=np.zeros((self.proj_H,self.W_end-self.W_start))
        self.phi_channel=np.zeros((self.proj_H,self.W_end-self.W_start))
        for i in range(self.proj_H):
            for j in range(self.W_end-self.W_start):
                
                self.theta_channel[i,j]=np.pi*(float(j+0.5+self.W_start)/self.proj_W*2-1)
                self.phi_channel[i,j]=(1-float(i+0.5)/self.proj_H)*fov -abs(fov_down)
        self.R_theta=[np.cos(self.theta_channel),-np.sin(self.theta_channel),zero_matrix,np.sin(self.theta_channel),np.cos(self.theta_channel),zero_matrix,zero_matrix,zero_matrix,one_matrix]
        self.R_theta=np.asarray(self.R_theta)
        self.R_theta=np.transpose(self.R_theta, (1, 2, 0))
        self.R_theta=np.reshape(self.R_theta,[self.proj_H,(self.W_end-self.W_start),3,3])
        self.R_phi=[np.cos(self.phi_channel),zero_matrix,-np.sin(self.phi_channel),zero_matrix,one_matrix,zero_matrix,np.sin(self.phi_channel),zero_matrix,np.cos(self.phi_channel)]
        self.R_phi=np.asarray(self.R_phi)
        self.R_phi=np.transpose(self.R_phi, (1, 2, 0))
        self.R_phi=np.reshape(self.R_phi,[self.proj_H,(self.W_end-self.W_start),3,3])
        self.R_theta_phi=np.matmul(self.R_theta,self.R_phi)
        
    def calculate_normal(self,range_image):
        
        one_matrix=np.ones((self.proj_H,self.W_end-self.W_start))
        img_gaussian =cv2.GaussianBlur(range_image,(3,3),0)
        #prewitt
        kernelx = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        self.partial_r_theta=img_prewitty /(np.pi*2.0/self.proj_W)/6
        self.partial_r_phi=img_prewittx/(((self.fov_up-self.fov_down)/180.0*np.pi)/self.proj_H)/6



        partial_vector=[1.0*one_matrix,self.partial_r_theta/(range_image*np.cos(self.phi_channel)),self.partial_r_phi/range_image]
        partial_vector=np.asarray(partial_vector)
        partial_vector=np.transpose(partial_vector, (1, 2, 0))
        partial_vector=np.reshape(partial_vector,[self.proj_H,(self.W_end-self.W_start),3,1])
        normal_vector=np.matmul(self.R_theta_phi,partial_vector)
        normal_vector=np.squeeze(normal_vector)
        normal_vector=normal_vector/np.reshape(np.max(np.abs(normal_vector),axis=2),(self.proj_H,self.W_end-self.W_start,1))
        normal_vector_camera=np.zeros((self.proj_H,self.W_end-self.W_start,3))
        normal_vector_camera[:,:,0]=normal_vector[:,:,1]
        normal_vector_camera[:,:,1]=-normal_vector[:,:,2]
        normal_vector_camera[:,:,2]=normal_vector[:,:,0]
        return normal_vector_camera
    

def put_normal_on_image(normal_spherical,x_indices,y_indices,proj_idx,proj_mask):
    point_id=proj_idx[proj_mask.astype(np.bool)]
    normal_vector=normal_spherical[proj_mask.astype(np.bool)]
    x_axis=x_indices[point_id]
    y_axis=y_indices[point_id]
    img=np.zeros((352,1216,3))
    img[y_axis,x_axis]=normal_vector
    return img