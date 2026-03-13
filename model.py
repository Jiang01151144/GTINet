#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from pointnet2 import pointnet2_utils
import torch.nn.functional as F
from torch.autograd import Variable
from util import transform_point_cloud
from chamfer_loss import *
from utils import pairwise_distance_batch, get_graph_feature, Pointer, get_knn_index, Discriminator, feature_extractor, compute_rigid_transformation, get_keypoints
from Pos_transformer import GeometricTransformer
from gconv import Siamese_Gconv
import math
from affinity_layer import Affinity
from sinkhorn import sinkhorn_rpm
from data_modelnet40 import farthest_subsample_points

class norm(nn.Module):
    def __init__(self, axis=2):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        mean = torch.mean(x, self.axis,keepdim=True) 
        std = torch.std(x, self.axis,keepdim=True)   
        x = (x-mean)/(std+1e-6) 
        return x

class Modified_softmax(nn.Module):
    def __init__(self, axis=1):
        super(Modified_softmax, self).__init__()
        self.axis = axis
        self.norm = norm(axis = axis)
    def forward(self, x):
        x = self.norm(x)
        x = Gradient.apply(x)
        x = F.softmax(x, dim=self.axis)
        return x

class Gradient(torch.autograd.Function):                                                                                                       
    @staticmethod
    def forward(ctx, input):
        return input*8
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        
class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.num_keypoints = args.n_keypoints
        self.weight_function = Discriminator(args)
        self.nn_margin = args.nn_margin
        self.DeSmooth = nn.Sequential(
            nn.Conv1d(in_channels=968, out_channels=968+128, kernel_size=1, stride=1,  bias=False),
            nn.LeakyReLU(), 
            #norm(axis=2),
            nn.Conv1d(in_channels=968+128, out_channels=968, kernel_size=1, stride=1,bias=False),
            #Modified_softmax(axis=2)
            ) 
        self.tah = nn.Tanh()
        
    def forward(self, *input):
        """
            Args:
                src: Source point clouds. Size (B, 3, N)
                tgt: target point clouds. Size (B, 3, M)
                src_embedding: Features of source point clouds. Size (B, C, N)
                tgt_embedding: Features of target point clouds. Size (B, C, M)
                src_idx: Nearest neighbor indices. Size [B * N * k]
                k: Number of nearest neighbors.
                src_knn: Coordinates of nearest neighbors. Size [B, N, K, 3]
                i: i-th iteration.
                tgt_knn: Coordinates of nearest neighbors. Size [B, M, K, 3]
                src_idx1: Nearest neighbor indices. Size [B * N * k]
                idx2:  Nearest neighbor indices. Size [B, M, k]
                k1: Number of nearest neighbors.
            Returns:
                R/t: rigid transformation.
                src_keypoints, tgt_keypoints: Selected keypoints of source and target point clouds. Size (B, 3, num_keypoint)
                src_keypoints_knn, tgt_keypoints_knn: KNN of keypoints. Size [b, 3, num_kepoints, k]
                loss_scl: Spatial Consistency loss.
        """
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        src_idx = input[4]
        k = input[5]
        src_knn = input[6] # [b, n, k, 3]
        i = input[7]
        tgt_knn = input[8] # [b, n, k, 3]
        src_idx1 = input[9] # [b * n * k1]
        idx2 = input[10] #[b, m, k1]
        k1 = input[11]

        batch_size, num_dims_src, _ = src.size()
        batch_size, _, num_points_tgt = tgt.size()
        batch_size, _, num_points = src_embedding.size()

        ########################## Matching Map Refinement Module ##########################
        distance_map = pairwise_distance_batch(src_embedding, tgt_embedding) #[b, n, m]
        # point-wise matching map
        perm_matrix = torch.exp(-distance_map)
        refined_matching_map = torch.softmax(-distance_map, dim=2) #[b, n, m]  Eq. (1)

        # spatial consistency loss 
        idx_tgt_corr = torch.argmax(perm_matrix, dim=2).unsqueeze(0).view(batch_size,num_points,1)
        src_corr = torch.matmul(tgt, refined_matching_map.transpose(2, 1).contiguous())# [b,3,n] Eq. (4)

        ############################## Inlier Evaluation Module ##############################
        # neighborhoods of pseudo target point clouds
        src_knn_corr = src_corr.transpose(2,1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_knn_corr = src_knn_corr.view(batch_size, num_points, k, num_dims_src)#[b, n, k, 3]

        # edge features of the pseudo target neighborhoods and the source neighborhoods 
        knn_distance = src_corr.transpose(2,1).contiguous().unsqueeze(2) - src_knn_corr #[b, n, k, 3]
        src_knn_distance = src.transpose(2,1).contiguous().unsqueeze(2) - src_knn #[b, n, k, 3]
        
        # inlier confidence
        weight = torch.gather(perm_matrix, dim = 2, index = idx_tgt_corr)
        weight = (weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-14)).transpose(1,2)#normalize，B，1，N
        
        # compute rigid transformation 
        R, t = compute_rigid_transformation(src, src_corr, weight) # weighted SVD
        ########################### Preparation for the Loss Function #########################
        # choose k keypoints with highest weights
        src_topk_idx, src_keypoints, tgt_keypoints = get_keypoints(src, src_corr, weight, self.num_keypoints)

        # spatial consistency loss 
        idx_tgt_corr = torch.argmax(refined_matching_map, dim=-1).int() # [b, n]
        identity = torch.eye(num_points_tgt).cuda().unsqueeze(0).repeat(batch_size, 1, 1) # [b, m, m]
        one_hot_number = pointnet2_utils.gather_operation(identity, idx_tgt_corr) # [b, m, n]
        src_keypoints_idx = src_topk_idx.repeat(1, num_points_tgt, 1) # [b, m, num_keypoints]
        keypoints_one_hot = torch.gather(one_hot_number, dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * self.num_keypoints, num_points_tgt)
        predicted_keypoints_scores = torch.gather(refined_matching_map.transpose(2, 1), dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * self.num_keypoints, num_points_tgt)
        loss_scl = (-torch.log(predicted_keypoints_scores + 1e-15) * keypoints_one_hot).sum(1).mean()

        # neighorhood information
        src_keypoints_idx2 = src_topk_idx.unsqueeze(-1).repeat(1, 3, 1, k) #[b, 3, num_keypoints, k]
        tgt_keypoints_knn = torch.gather(knn_distance.permute(0,3,1,2), dim = 2, index = src_keypoints_idx2) #[b, 3, num_kepoints, k]

        src_transformed = transform_point_cloud(src, R, t.view(batch_size, 3))
        src_transformed_knn_corr = src_transformed.transpose(2,1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_transformed_knn_corr = src_transformed_knn_corr.view(batch_size, num_points, k, num_dims_src) #[b, n, k, 3]

        knn_distance2 = src_transformed.transpose(2,1).contiguous().unsqueeze(2) - src_transformed_knn_corr #[b, n, k, 3]
        src_keypoints_knn = torch.gather(knn_distance2.permute(0,3,1,2), dim = 2, index = src_keypoints_idx2) #[b, 3, num_kepoints, k]
        return R, t.view(batch_size, 3), src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, loss_scl

def logcosh(pred, true):
    loss = torch.log(torch.cosh(pred - true))
    return torch.sum(loss)

def logcosh_kitti(pred, true):
    loss = torch.log(torch.cosh(pred - true))
    return loss
    
class LossFunction(nn.Module):
    def __init__(self, args):
        super(LossFunction, self).__init__()


        self.criterion = nn.MSELoss(reduction='sum')
        self.criterion4 = nn.SmoothL1Loss(reduction='sum')
        self.GAL = GlobalAlignLoss()
        self.margin = args.loss_margin

    def forward(self, *input):
        """
            Compute global alignment loss and neighorhood consensus loss
            Args:
                src_keypoints: Keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
            Returns:
                neighborhood_consensus_loss
                global_alignment_loss
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        batch_size = src_keypoints.size()[0]
        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin) 
        
        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = logcosh(transformed_srckps_forward, tgt_keypoints)
        knn_consensus_loss = logcosh(src_keypoints_knn, tgt_keypoints_knn)
        neighborhood_consensus_loss = knn_consensus_loss/k + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss

class LossFunction_kitti(nn.Module):
    def __init__(self, args):
        super(LossFunction_kitti, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='none')
        self.GAL = GlobalAlignLoss()
        self.margin = args.loss_margin

    def forward(self, *input):
        """
            Compute global alignment loss and neighorhood consensus loss
            Args:
                src_keypoints: Selected keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Selected keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
            Returns:
                neighborhood_consensus_loss
                global_alignment_loss
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin) 

        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = logcosh_kitti(transformed_srckps_forward, tgt_keypoints).sum(1).sum(1)
        knn_consensus_loss = logcosh_kitti(src_keypoints_knn, tgt_keypoints_knn).sum(1).sum(1).mean()
        neighborhood_consensus_loss = knn_consensus_loss + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss

class GTINet_LCE(nn.Module):
    def __init__(self, args):
        super(GTINet_LCE, self).__init__()
        self.emb_dims = args.emb_dims
        

        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(self.emb_dims)
        self.relu = nn.ReLU()

    def forward(self, x, k):

        idx, relative_coords, knn_points, idx2 = get_graph_feature(x, k)
        
        x = self.relu(self.bn1(self.conv1(relative_coords)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        features = x.max(dim=-1)[0]  # [B, emb_dims, N]
        
        return features, idx, knn_points

class GTINet_GSR(nn.Module):

    def __init__(self, args):
        super(GTINet_GSR, self).__init__()
        self.emb_dims = args.emb_dims
        self.gnn_layer = Siamese_Gconv(self.emb_dims, self.emb_dims)
        self.cross_graph = nn.Linear(self.emb_dims * 2, self.emb_dims)
        self.inst_norm = nn.InstanceNorm2d(1, affine=True)
        
    def forward(self, src_feat, tgt_feat):
        src_feat = src_feat.transpose(1, 2)
        tgt_feat = tgt_feat.transpose(1, 2)
        
        d_k = src_feat.size(-1)
        
        scores_src = torch.matmul(src_feat, src_feat.transpose(1, 2)) / math.sqrt(d_k)
        scores_tgt = torch.matmul(tgt_feat, tgt_feat.transpose(1, 2)) / math.sqrt(d_k)
        
        E_src = torch.softmax(scores_src, dim=-1)
        E_tgt = torch.softmax(scores_tgt, dim=-1)
        
        src_intra, tgt_intra = self.gnn_layer(
            [E_src, src_feat], 
            [E_tgt, tgt_feat]
        )
        
        distance = pairwise_distance_batch(
            src_intra.transpose(1, 2), 
            tgt_intra.transpose(1, 2)
        )  # [B, N, M]
        
        # Instance normalization
        distance = self.inst_norm(distance[:, None, :, :]).squeeze(1)
        
        tgt_proj = torch.softmax(-distance, dim=1)  # [B, N, M]
        src_proj = torch.softmax(-distance, dim=2)  # [B, N, M]
        
        src_cross = torch.bmm(src_intra, tgt_proj)  # [B, N, C]
        tgt_cross = torch.bmm(tgt_intra, src_proj.transpose(1, 2))  # [B, M, C]
        
        src_global = self.cross_graph(torch.cat([src_intra, src_cross], dim=-1))
        tgt_global = self.cross_graph(torch.cat([tgt_intra, tgt_cross], dim=-1))
        
        return src_global.transpose(1, 2), tgt_global.transpose(1, 2)

class GTINet_CTI(nn.Module):
    def forward(self, src_feat, tgt_feat, src_xyz, tgt_xyz):
        # src_feat: [B, C, N], tgt_feat: [B, C, M]
        
        s = self.affinity(src_feat.transpose(1, 2), tgt_feat.transpose(1, 2))
        log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
        geo_sim = torch.exp(log_s)  # [B, N, M]
        
        src_weight = torch.softmax(geo_sim, dim=1)  # [B, N, M]
        tgt_weight = torch.softmax(geo_sim, dim=2)  # [B, N, M]
        
        src_proj = torch.bmm(src_weight, tgt_xyz.transpose(1, 2))  # [B, N, 3]
        tgt_proj = torch.bmm(tgt_weight.transpose(1, 2), src_xyz.transpose(1, 2))  # [B, M, 3]
        
        src_pos = self.pos_proj(src_proj.transpose(1, 2)).transpose(1, 2)  # [B, N, C]
        tgt_pos = self.pos_proj(tgt_proj.transpose(1, 2)).transpose(1, 2)  # [B, M, C]
        
        src_pos_aware = src_feat.transpose(1, 2) + src_pos  # [B, N, C]
        tgt_pos_aware = tgt_feat.transpose(1, 2) + tgt_pos  # [B, M, C]
        
        src_enhanced, tgt_enhanced = self.attention(
            src_weight, tgt_weight,
            src_pos_aware.transpose(1, 2), 
            tgt_pos_aware.transpose(1, 2),
            src_xyz, tgt_xyz
        )
        
        src_out = self.relu(src_feat + src_enhanced)
        tgt_out = self.relu(tgt_feat + tgt_enhanced)
        
        return src_out, tgt_out

class GTINet(nn.Module):

    def __init__(self, args):
        super(GTINet, self).__init__()
        
        self.lce = GTINet_LCE(args)      # Local Connection Embedding
        self.gsr = GTINet_GSR(args)      # Global Structural Relations
        self.cti = GTINet_CTI(args)      # Contextual Topological Interactions
        
        self.rfsc = SVDHead(args)         # Rigid Fitting with Soft Correspondences
        
        self.iter = args.n_iters
        self.list_k1 = args.list_k1
        self.list_k2 = args.list_k2
        
        if args.dataset == 'kitti':
            self.loss = LossFunction_kitti(args)
        else:
            self.loss = LossFunction(args)
        
        self.num_keypoints = args.n_keypoints

    def forward(self, *input):

        src = input[0]
        tgt = input[1]
        batch_size = src.size(0)
        
        rotation_ab_pred = torch.eye(3, device=src.device).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device).view(1, 3).repeat(batch_size, 1)
        
        global_alignment_loss, consensus_loss, spatial_consistency_loss = 0.0, 0.0, 0.0

        for i in range(self.iter):
            src_feat, src_idx, src_knn = self.lce(src, self.list_k1[i])
            tgt_feat, _, tgt_knn = self.lce(tgt, self.list_k1[i])

            src_global, tgt_global = self.gsr(src_feat, tgt_feat)

            src_context, tgt_context = self.cti(src_global, tgt_global, src, tgt)

            src_idx1, _ = get_knn_index(src, self.list_k2[i])
            _, tgt_idx = get_knn_index(tgt, self.list_k2[i])
            _, _, src_knn_full, src_idx_full = get_graph_feature(src, self.list_k1[i])
            _, _, tgt_knn_full, _ = get_graph_feature(tgt, self.list_k1[i])

            rotation_i, translation_i, src_keypoints, tgt_keypoints, \
            src_keypoints_knn, tgt_keypoints_knn, spatial_loss_i = self.rfsc(
                src, tgt, src_context, tgt_context, src_idx_full, 
                self.list_k1[i], src_knn_full, i, tgt_knn_full,
                src_idx1, tgt_idx, self.list_k2[i]
            )

            rotation_ab_pred = torch.matmul(rotation_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_i, translation_ab_pred.unsqueeze(2)).squeeze(2) + translation_i

            src = transform_point_cloud(src, rotation_i, translation_i)

            neighborhood_loss_i, global_loss_i = self.loss(
                src_keypoints, tgt_keypoints,
                rotation_i, translation_i,
                src_keypoints_knn, tgt_keypoints_knn,
                self.list_k2[i], src, tgt
            )

            global_alignment_loss += global_loss_i      # L_g
            consensus_loss += neighborhood_loss_i       # L_l
            spatial_consistency_loss += spatial_loss_i  # L_s

        return rotation_ab_pred, translation_ab_pred, \
               global_alignment_loss, consensus_loss, spatial_consistency_loss