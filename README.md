# CVTHead
Official Implementation of WACV 2024 paper, "CVTHead: One-shot Controllable Head Avatar with Vertex-feature Transformer"


[![arXiv](https://img.shields.io/badge/arXiv-2311.svg)](https://arxiv.org/abs/2311.06443)


**Efficient** and **controllable** head avatar generation from a single image with **point-based neural rendering**. 


<img src="https://github.com/HowieMa/CVTHead/blob/main/examples/shape.gif" alt="novel shape" width="150" height="150">
<img src="https://github.com/HowieMa/CVTHead/blob/main/examples/pose.gif" alt="novel expression" width="150" height="150">
<img src="https://github.com/HowieMa/CVTHead/blob/main/examples/exp.gif" alt="novel expression" width="150" height="150">
<img src="https://github.com/HowieMa/CVTHead/blob/main/examples/jaw.gif" alt="novel expression" width="150" height="150">


![framework](https://github.com/HowieMa/CVTHead/blob/main/assets/framework.png)

## Introduction
Reconstructing personalized animatable head avatars has significant implications in the fields of AR/VR. Existing methods for achieving explicit face control of 3D Morphable Models (3DMM) typically rely on multi-view images or videos of a single subject, making the reconstruction process complex. Additionally, the traditional rendering pipeline is time-consuming, limiting real-time animation possibilities. In this paper, we introduce CVTHead, a novel approach that generates controllable neural head avatars from a single reference image using point-based neural rendering. CVTHead considers the sparse vertices of mesh as the point set and employs the proposed Vertex-feature Transformer to learn local feature descriptors for each vertex. This enables the modeling of long-range dependencies among all the vertices. Experimental results on the VoxCeleb dataset demonstrate that CVTHead achieves comparable performance to state-of-the-art graphics-based methods. Moreover, it enables efficient rendering of novel human heads with various expressions, head poses, and camera views. These attributes can be explicitly controlled using the coefficients of 3DMMs, facilitating versatile and realistic animation in real-time scenarios. 



## Install
#### Setup environment
~~~
conda create -n cvthead python=3.9
conda activate cvthead

pip install -r requirements.txt
~~~

#### Download pre-trained weights
~~~
cd data/
bash fetch_data.sh
cd ..
~~~
Please go to `data/README.md` for more details. 


## Inference

Download our pre-trained model `cvthead.pt` from [Google Drive](https://drive.google.com/drive/folders/12wDExqDiU2LDTrM-2Mg9HFEvjeFJQlG5?usp=sharing)
and put it under `data/` folder

Here is a demo to use CVTHead for cross-identity face reenactment
~~~
python inference.py --src_pth examples/1.png --drv_pth examples/2.png --out_pth examples/output.png --ckpt_pth data/cvthead.pt
~~~

Here is a demo for face generation under the control of FLAME coefficients
~~~
python inference.py --src_pth examples/1.png --out_pth examples --ckpt_pth data/cvthead.pt --flame
~~~

## Training

#### Dataset Preparation (VoxCelebV1)

- Download videos
Please refer [video-preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing) to download the video. 
We follow the original given bounding box from VoxCeleb1, rather than the union of a third-party detector bbox and the given bbox in [this](https://github.com/AliaksandrSiarohin/video-preprocessing/blob/master/crop_vox.py#L19) preprocessing step.  


Note: some videos may no longer be available on Youtube. Due to the copyright and privacy issue, I cannot share these face videos to others.  

- obtain per frame landmarks with `face_alignment`


The data organization should looks like
~~~
--VoxCeleb1

---- vox_video
------ train
-------- xxxx.mp4
-------- ......
------ test
-------- id10280#NXjT3732Ekg#001093#001192.mp4
-------- xxxx.mp4
-------- ......

---- vox_lmks_meta
------ train
-------- xxxx.pkl
-------- ......
------ test
-------- id10280#NXjT3732Ekg#001093#001192.pkl
-------- xxxx.pkl
-------- ......
~~~

#### training scripts
- We find that spliting the training into two separate stages can obtain more stable training curves. 
In the first stage, we train the model without the adversarial loss. 
In the second stage, we continue training the model with all losses claimed in the paper for a few epoches. 

~~~
torchrun --standalone --nnodes 1 --nproc_per_node 2 main_stage1.py --config configs/vox1.yaml
torchrun --standalone --nnodes 1 --nproc_per_node 2 main_stage2.py --config configs/vox1.yaml
~~~

## Acknowledgement
[ROME](https://github.com/SamsungLabs/rome)   
[DECA](https://github.com/yfeng95/DECA)   
[Spiralnet++](https://github.com/sw-gong/spiralnet_plus)   
[face-parsing.PyTorch](https://github.com/VisionSystemsInc/face-parsing.PyTorch)  
[face-alignment](https://github.com/1adrianb/face-alignment)   


## Citation
If you found this code helpful, please consider citing:
~~~
@article{ma2023cvthead,
  title={CVTHead: One-shot Controllable Head Avatar with Vertex-feature Transformer},
  author={Ma, Haoyu and Zhang, Tong and Sun, Shanlin and Yan, Xiangyi and Han, Kun and Xie, Xiaohui},
  journal={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
~~~
