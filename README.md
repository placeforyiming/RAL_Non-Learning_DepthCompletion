# A Surface Geometry Model for LiDAR Depth Completion (IEEE RA-L) 

## Abstract:

We develop a non-learning depth completion pipeline with the assumption that the empty pixel shares the same surface with the nearest value. The pipeline has two major technical modules: an outlier removal and a surface model to explicitly calculate the depth with a mathematical equation. You can check our paper for more details:

The arxiv version: https://arxiv.org/pdf/2104.08466.pdf

The IEEE RAL version: https://ieeexplore.ieee.org/abstract/document/9387169?casa_token=m6hbd7riBioAAAAA:bsyIXoghUPIyvK_oASnZAATLqq1h1yt2gAEQbMPLSvdOl0zRsFadmPPuYTPnIEDwlrdW8bM


## Method Pipeline:
<p align="center">
<img src="/demo/pipeline.png" alt="pipeline" width="800" height="300">
</p>

## Outlier Removal Examples:
<p align="center">
<img src="/demo/outlier_removal.png" alt="outlier_removal" width="700" height="300">
</p>



## Data Structure


Download the validation and test dataset from KITTI depth completion benchmark website, then put them under the fodler depth_selection with the stucture:

----- depth_selection\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-------val_selection_cropped\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-------test_depth_completion_anonymous



## Environment Settings


Note: we have two different outlier removal methods. The function ***outlier_removal()*** in ***tools.py*** is a simple opencv filter-based algorithm. It can work on 64 line LiDAR but need mannually adjust two hyperparameters, thus can not directly extend on sparser 16 line or 32 line LiDAR. To design a more general outlier removal algoritem, we think from first principal rule and have an alternative one in ***non_learning_completion_multi_res.py***. The function ***outlier_removal_mask()*** is parameter-free, that we believe more general. However, we only have one month to prepare the reply of our RA-L submission, so the current implementation utilize the extract_patch function in tensorflow. This is why the tensorflow 2.0 is temporary needed. we will implement a c++ version just like the convd operator in OpenCV in the short future.

Pull the docker image:

***docker pull tensorflow/tensorflow:2.1.0-gpu-py3-jupyter***


Activate the docker container with mounting the code in this repository.

Install packages with the certain version in the docker container:

***bash install_dependency.sh***


To subsample the 32 line and 16 line LiDAR, please run:

***python subsample_Lidar_val.py***


After installing all packages, you can open the jupyter note book by **bash jupyter.sh**, and use **depth_completion.ipynb** to explore each step of our method.



## Run the Code 


Run the completion with 64 line LiDAR on KITTI validation (simple outlier removal):

***python non_learning_completion_CPU_64line.py***

Run the completion with 64 line LiDAR on KITTI test (simple outlier removal):

***python non_learning_completion_CPU_64line_test.py***



Run the completion with 64 line LiDAR on KITTI validation (general outlier removal):

***python non_learning_completion_multi_res.py***

Run the completion with 64 line LiDAR on KITTI test (general outlier removal):

***python non_learning_completion_multi_res_test.py***


@article{zhao2021surface,

  title={A Surface Geometry Model for LiDAR Depth Completion},
  
  author={Zhao, Yiming and Bai, Lin and Zhang, Ziming and Huang, Xinming},
  
  journal={IEEE Robotics and Automation Letters},
  
  year={2021},
  
  publisher={IEEE}
  
}
 
