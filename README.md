# A Surface Geometry Model for LiDAR Depth Completion (IEEE RA-L) 

**Abstract:**

LiDAR depth completion is a task that predicts depth values for every pixel on the corresponding camera frame, although only sparse LiDAR points are available. Most of the existing state-of-the-art solutions are based on deep neural networks, which need a large amount of data and heavy computations for training the models. In this letter, a novel non-learning depth completion method is proposed by exploiting the local surface geometry that is enhanced by an outlier removal algorithm. The proposed surface geometry model is inspired by the observation that most pixels with unknown depth have a nearby LiDAR point. Therefore, it is assumed those pixels share the same surface with the nearest LiDAR point, and their respective depth can be estimated as the nearest LiDAR depth value plus a residual error. The residual error is calculated by using a derived equation with several physical parameters as input, including the known camera intrinsic parameters, estimated normal vector, and offset distance on the image plane. The proposed method is further enhanced by an outlier removal algorithm that is designed to remove incorrectly mapped LiDAR points from occluded regions. On KITTI dataset, the proposed solution achieves the best error performance among all existing non-learning methods and is comparable to the best self-supervised learning method and some supervised learning methods. Moreover, since  outlier points from occluded regions is a commonly existing problem, the proposed outlier removal algorithm is a general preprocessing
step that is applicable to many robotic systems with both camera and LiDAR sensors.

**Method Pipeline:**
<p align="center">
<img src="/demo/pipeline.png" alt="pipeline" width="800" height="300">
</p>

**Outlier Removal Examples:**
<p align="center">
<img src="/demo/outlier_removal.png" alt="outlier_removal" width="700" height="300">
</p>


**Environment Settings**
Note: we have two different outlier removal methods. The function *outlier_removal()* in *tools.py* is a simple opencv filter-based algorithm. It can work on 64 line LiDAR but need mannually adjust two hyperparameters, thus can not directly extend on sparser 16 line or 32 line LiDAR. To design a more general outlier removal algoritem, we think from first principal rule and have an alternative one in *non_learning_completion_multi_res.py*. The function *outlier_removal_mask()* is parameter-free, that we believe more general. However, we only have one month to prepare the reply of our RA-L submission, so the current implementation utilize the extract_patch function in tensorflow. This is why the tensorflow 2.0 is temporary needed. we will implement a c++ version just like the convd operator in OpenCV in the short future.

Pull the docker image:

*docker pull tensorflow/tensorflow:2.1.0-gpu-py3-jupyter*

Activate the docker container with mounting the code in this repository.

Install packages with the certain version in the docker container:

*bash install_dependency.sh*

