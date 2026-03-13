# GTINet: Global Topology-aware Interactions for Unsupervised Point Cloud Registration

by Yinuo Jiang, Beitong Zhou, Xiaoyu Liu, Qingyi Li, Cheng Cheng, details are in [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10440120).

### Introduction

This repository contains the source code and pre-trained models for **GTINet** (Global Topology-aware Interactions Network), an unsupervised point cloud registration framework published in **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**.

GTINet addresses the feature matching ambiguity problem in unsupervised point cloud registration through two novel modules:
- **Global Structural Relations (GSR) module**: Transforms local features into global features via global graph convolutions
- **Contextual Topological Interactions (CTI) module**: Learns geometric feature similarities and relative positional knowledge through topology-aware attention layers

### Usage

#### 1. Requirements

**Hardware:**
- NVIDIA GeForce RTX 3090 (or equivalent GPU)

**Software:**
- Python 3.8+
- PyTorch >= 1.9.0
- CUDA >= 11.1
- cuDNN >= 8.0

**Python Dependencies:**
```
numpy>=1.19.5
scipy>=1.6.0
h5py>=3.2.0
tqdm>=4.62.0
pyyaml>=5.4.0
easydict>=1.9
scikit-learn>=0.24.0
tensorboard>=2.7.0
open3d>=0.12.0
pointnet2-py (included in repository)
```

#### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GTINet.git
cd GTINet

# Install pointnet2 operators
cd pointnet2 && python setup.py install && cd ..

# Install other dependencies
pip install -r requirements.txt
```

#### 3. Datasets

##### ModelNet40
Download from [official source](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip):
```bash
wget --no-check-certificate https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip -d data/
```

##### 7Scenes
Download from [7Scenes project page](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/):
```
7scene
├── chess
│   ├── cloud_bin_0.info.txt
│   ├── cloud_bin_0.ply
│   ├── ...
├── fire
├── heads
├── office
├── pumpkin
├── redkitchen
├── stairs
```

##### KITTI Odometry
Download from [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php):
```
sequences
├── 00
│   ├── velodyne/
│   ├── calib.txt
├── 01
├── 02
├── ...
├── 10
```

#### 5. Training

Train on different datasets:

```bash
# ModelNet40
CUDA_VISIBLE_DEVICES=0 python main.py ./config/train_modelnet40.yaml

# 7Scenes
CUDA_VISIBLE_DEVICES=0 python main.py ./config/train_7scene.yaml

# KITTI Odometry
CUDA_VISIBLE_DEVICES=0 python main.py ./config/train_kitti.yaml
```
### Acknowledgement

Our code is built upon several excellent open-source projects:
- [RIENet](https://github.com/supersyq/RIENet) - Reliable Inlier Evaluation Network
- [DCP](https://github.com/WangYueFt/dcp) - Deep Closest Point
- [RPMNet](https://github.com/yewzijian/RPMNet) - Robust Point Matching Network
- [PointNet++](https://github.com/charlesq34/pointnet2) - PointNet++ implementation
- [DGCNN](https://github.com/WangYueFt/dgcnn) - Dynamic Graph CNN

We thank the authors for their valuable contributions to the community.

