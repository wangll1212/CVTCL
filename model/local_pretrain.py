# MoCo-related code is modified from https://github.com/facebookresearch/moco
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../')
from backbone.select_backbone import select_backbone


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class InfoNCE(nn.Module):
    '''
    Basically, it's a MoCo for video input: https://arxiv.org/abs/1911.05722
    '''
    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(InfoNCE, self).__init__()

        self.dim = dim 
        self.K = K
        self.m = m
        self.T = T

        # create the encoders (including non-linear projection head: 2 FC layers)
        backbone, self.param = select_backbone(network)
        feature_size = self.param['feature_size']
        self.encoder_q = nn.Sequential(
                            backbone, 
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))

        backbone, _ = select_backbone(network)
        self.encoder_k = nn.Sequential(
                            backbone, 
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Notes: for handling sibling videos, e.g. for UCF101 dataset


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''Momentum update of the key encoder'''
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        '''
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        '''
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self,block, clip1, clip2, clip3, clip4):
        '''Output: logits, targets'''
        (B, N, *_) = block.shape # [B,N,C,T,H,W]  [16,2,3,32,128,128]  # t为帧数
        # print(block.shape)
        assert N == 2    # 同一视频的两个片段的不同数据增强
        x1 = block[:,0,:].contiguous()  # [B,1,C,T,H,W]  [16,1,3,32,128,128]
        (B, N, *_) = clip1.shape   # (B,2,3,8,128,128)
        assert N == 2
        (B, N, *_) = clip2.shape
        (B, N, *_) = clip3.shape
        (B, N, *_) = clip4.shape
        clip11 = clip1[:, 0, :].contiguous()
        clip12 = clip2[:, 0, :].contiguous()
        clip13 = clip3[:, 0, :].contiguous()
        clip14 = clip4[:, 0, :].contiguous()

        x2 = block[:,1,:].contiguous()  # [B,1,C,T,H,W]


        clip21 = clip1[:, 1, :].contiguous()
        clip22 = clip2[:, 1, :].contiguous()
        clip23 = clip3[:, 1, :].contiguous()
        clip24 = clip4[:, 1, :].contiguous()

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        q_clip11 = self.encoder_q(clip11)
        q_clip11 = nn.functional.normalize(q_clip11, dim=1)
        q_clip11 = q_clip11.view(B, self.dim)


        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            x2, idx_unshuffle = self._batch_shuffle_ddp(x2)
            clip21, idx1_unshuffle = self._batch_shuffle_ddp(clip21)
            clip22, idx2_unshuffle = self._batch_shuffle_ddp(clip22)
            clip23, idx3_unshuffle = self._batch_shuffle_ddp(clip23)
            clip24, idx4_unshuffle = self._batch_shuffle_ddp(clip24)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)
            k_clip21 = self.encoder_k(clip21)  # keys: B,C,1,1,1
            k_clip21 = nn.functional.normalize(k_clip21, dim=1)
            k_clip22 = self.encoder_k(clip22)  # keys: B,C,1,1,1
            k_clip22 = nn.functional.normalize(k_clip22, dim=1)
            k_clip23 = self.encoder_k(clip23)  # keys: B,C,1,1,1
            k_clip23 = nn.functional.normalize(k_clip23, dim=1)
            k_clip24 = self.encoder_k(clip24)  # keys: B,C,1,1,1
            k_clip24 = nn.functional.normalize(k_clip24, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_clip21 = self._batch_unshuffle_ddp(k_clip21, idx1_unshuffle)
            k_clip22 = self._batch_unshuffle_ddp(k_clip22, idx2_unshuffle)
            k_clip23 = self._batch_unshuffle_ddp(k_clip23, idx3_unshuffle)
            k_clip24 = self._batch_unshuffle_ddp(k_clip24, idx4_unshuffle)

        k = k.view(B, self.dim)
        k_clip21 = k_clip21.view(B, self.dim)
        k_clip22 = k_clip22.view(B, self.dim)
        k_clip23 = k_clip23.view(B, self.dim)
        k_clip24 = k_clip24.view(B, self.dim)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: B,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T

        # L_inter:(q_clip11, {k_clip21,k_clip22,k_clip23,k_clip24}, N)
        l_pos_local1 = torch.einsum('nc,nc->n', [q_clip11, k_clip21]).unsqueeze(-1)
        l_pos_local2 = torch.einsum('nc,nc->n', [q_clip11, k_clip22]).unsqueeze(-1)
        l_pos_local3 = torch.einsum('nc,nc->n', [q_clip11, k_clip23]).unsqueeze(-1)
        l_pos_local4 = torch.einsum('nc,nc->n', [q_clip11, k_clip24]).unsqueeze(-1)
        l_neg_local = torch.einsum('nc,ck->nk', [q_clip11, self.queue.clone().detach()])
        logits_local = torch.cat((torch.cat((l_pos_local1, l_pos_local2, l_pos_local3, l_pos_local4), dim=0), l_neg_local.repeat(4, 1)), dim=1)
        logits_local /= self.T

        # L_intra
        l_pos_intra = torch.einsum('nc,nc->n', [q_clip11, k_clip21]).unsqueeze(-1)
        l_neg_intra1 = torch.einsum('nc,nc->n', [q_clip11, k_clip22]).unsqueeze(-1)
        l_neg_intra2 = torch.einsum('nc,nc->n', [q_clip11, k_clip23]).unsqueeze(-1)
        l_neg_intra3 = torch.einsum('nc,nc->n', [q_clip11, k_clip24]).unsqueeze(-1)
        logits_intra = torch.cat((l_pos_intra.repeat(3, 1), torch.cat((l_neg_intra1, l_neg_intra2, l_neg_intra3), dim=0)), dim=-1)
        logits_intra /= self.T


        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        labels_local = torch.zeros(logits_local.shape[0], dtype=torch.long).cuda()
        labels_intra = torch.zeros(logits_intra.shape[0], dtype=torch.long).cuda()
        
        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k)

        return logits, logits_local, logits_intra, labels, labels_local,  labels_intra


class UberNCE(InfoNCE):
    '''
    UberNCE is a supervised version of InfoNCE,
    it uses labels to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''
    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(UberNCE, self).__init__(network, dim, K, m, T)
        # extra queue to store label
        self.register_buffer("queue_label", torch.ones(K, dtype=torch.long) * -1)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, block, k_label):
        '''Output: logits, binary mask for positive pairs
        '''
        (B, N, *_) = block.shape # [B,N,C,T,H,W]
        assert N == 2  # 同一视频的两个片段的不同数据增强
        x1 = block[:,0,:].contiguous()
        x2 = block[:,1,:].contiguous()

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.view(B, self.dim)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: B,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # mask: binary mask for positive keys
        mask = k_label.unsqueeze(1) == self.queue_label.unsqueeze(0) # B,K
        mask = torch.cat([torch.ones((mask.shape[0],1), dtype=torch.long, device=mask.device).bool(),
                          mask], dim=1) # B,(1+K)
                
        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, k_label)

        return logits, mask


class CoCLR(InfoNCE):
    '''
    CoCLR: using another view of the data to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''
    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07, topk=5, reverse=False):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(CoCLR, self).__init__(network, dim, K, m, T)

        self.topk = topk

        # create another encoder, for the second view of the data 
        backbone, _ = select_backbone(network)
        feature_size = self.param['feature_size']
        self.sampler = nn.Sequential(
                            backbone,
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))
        for param_s in self.sampler.parameters():
            param_s.requires_grad = False  # not update by gradient

        # create another queue, for the second view of the data
        self.register_buffer("queue_second", torch.randn(dim, K))
        self.queue_second = nn.functional.normalize(self.queue_second, dim=0)
        
        # for handling sibling videos, e.g. for UCF101 dataset
        self.register_buffer("queue_vname", torch.ones(K, dtype=torch.long) * -1) 
        # for monitoring purpose only
        self.register_buffer("queue_label", torch.ones(K, dtype=torch.long) * -1)
        
        self.queue_is_full = False
        self.reverse = reverse 

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_second, vnames):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        keys_second = concat_all_gather(keys_second)
        vnames = concat_all_gather(vnames)
        # labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_second[:, ptr:ptr + batch_size] = keys_second.T
        self.queue_vname[ptr:ptr + batch_size] = vnames
        self.queue_label[ptr:ptr + batch_size] = torch.ones_like(vnames)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, block1, block2, clip11, clip21,clip12,clip22, clip13,clip23,clip14, clip24, k_vsource):
        '''Output: logits, targets'''
        # block1, block2两个视频片段
        (B, N, *_) = block1.shape # B,N,C,T,H,W
        assert N == 2   # 两种模态数据
        x1 = block1[:,0,:].contiguous()  # RGB
        f1 = block1[:,1,:].contiguous()  # flow
        (B, N, *_) = clip11.shape   # (B,2,3,8,128,128)
        assert N == 2
        # (B, N, *_) = clip12.shape
        # (B, N, *_) = clip13.shape
        # (B, N, *_) = clip14.shape
        clip11_rgb = clip11[:, 0, :].contiguous()
        # clip12_rgb = clip12[:, 0, :].contiguous()
        # clip13_rgb = clip13[:, 0, :].contiguous()
        # clip14_rgb = clip14[:, 0, :].contiguous()
        clip11_flow = clip11[:, 1, :].contiguous()
        # clip12_flow = clip12[:, 1, :].contiguous()
        # clip13_flow = clip13[:, 1, :].contiguous()
        # clip14_flow = clip14[:, 1, :].contiguous()


        x2 = block2[:,0,:].contiguous()
        f2 = block2[:,1,:].contiguous()
        (B, N, *_) = clip21.shape   # (B,2,3,8,128,128)
        assert N == 2
        (B, N, *_) = clip22.shape
        (B, N, *_) = clip23.shape
        (B, N, *_) = clip24.shape
        clip21_rgb = clip21[:, 0, :].contiguous()
        clip22_rgb = clip22[:, 0, :].contiguous()
        clip23_rgb = clip23[:, 0, :].contiguous()
        clip24_rgb = clip24[:, 0, :].contiguous()
        clip21_flow = clip21[:, 1, :].contiguous()
        clip22_flow = clip22[:, 1, :].contiguous()
        clip23_flow = clip23[:, 1, :].contiguous()
        clip24_flow = clip24[:, 1, :].contiguous()

        if self.reverse:
            x1, f1 = f1, x1
            x2, f2 = f2, x2
            clip11_rgb, clip11_flow  = clip11_flow, clip11_rgb
            # clip12_rgb, clip12_flow  = clip12_flow, clip12_rgb
            # clip13_rgb, clip13_flow  = clip13_flow, clip13_rgb
            # clip14_rgb, clip14_flow  = clip14_flow, clip14_rgb
            clip21_rgb, clip21_flow  = clip21_flow, clip21_rgb
            clip22_rgb, clip22_flow  = clip22_flow, clip22_rgb
            clip23_rgb, clip23_flow  = clip23_flow, clip23_rgb
            clip24_rgb, clip24_flow  = clip24_flow, clip24_rgb

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        q_clip11 = self.encoder_q(clip11_rgb)
        q_clip11 = nn.functional.normalize(q_clip11, dim=1)
        q_clip11 = q_clip11.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            x2, idx_unshuffle = self._batch_shuffle_ddp(x2)           
            clip21, idx1_unshuffle = self._batch_shuffle_ddp(clip21_rgb)
            clip22, idx2_unshuffle = self._batch_shuffle_ddp(clip22_rgb)
            clip23, idx3_unshuffle = self._batch_shuffle_ddp(clip23_rgb)
            clip24, idx4_unshuffle = self._batch_shuffle_ddp(clip24_rgb)


            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)
            k_clip21 = self.encoder_k(clip21_rgb)  # keys: B,C,1,1,1
            k_clip21 = nn.functional.normalize(k_clip21, dim=1)
            k_clip22 = self.encoder_k(clip22_rgb)  # keys: B,C,1,1,1
            k_clip22 = nn.functional.normalize(k_clip22, dim=1)
            k_clip23 = self.encoder_k(clip23_rgb)  # keys: B,C,1,1,1
            k_clip23 = nn.functional.normalize(k_clip23, dim=1)
            k_clip24 = self.encoder_k(clip24_rgb)  # keys: B,C,1,1,1
            k_clip24 = nn.functional.normalize(k_clip24, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k = k.view(B, self.dim)
            k_clip21 = self._batch_unshuffle_ddp(k_clip21, idx1_unshuffle)
            k_clip22 = self._batch_unshuffle_ddp(k_clip22, idx2_unshuffle)
            k_clip23 = self._batch_unshuffle_ddp(k_clip23, idx3_unshuffle)
            k_clip24 = self._batch_unshuffle_ddp(k_clip24, idx4_unshuffle)
            k_clip21 = k_clip21.view(B, self.dim)
            k_clip22 = k_clip22.view(B, self.dim)
            k_clip23 = k_clip23.view(B, self.dim)
            k_clip24 = k_clip24.view(B, self.dim)

            # compute key feature for second view
            kf = self.sampler(f2) # keys: B,C,1,1,1
            kf = nn.functional.normalize(kf, dim=1)
            kf = kf.view(B, self.dim)

        # if queue_second is full: compute mask & train CoCLR, else: train InfoNCE

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: N,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # L_inter:(q_clip11, {k_clip21,k_clip22,k_clip23,k_clip24}, N)
        l_pos_local1 = torch.einsum('nc,nc->n', [q_clip11, k_clip21]).unsqueeze(-1)
        l_pos_local2 = torch.einsum('nc,nc->n', [q_clip11, k_clip22]).unsqueeze(-1)
        l_pos_local3 = torch.einsum('nc,nc->n', [q_clip11, k_clip23]).unsqueeze(-1)
        l_pos_local4 = torch.einsum('nc,nc->n', [q_clip11, k_clip24]).unsqueeze(-1)
        l_neg_local = torch.einsum('nc,ck->nk', [q_clip11, self.queue.clone().detach()])
        logits_local = torch.cat((torch.cat((l_pos_local1, l_pos_local2, l_pos_local3, l_pos_local4), dim=0), l_neg_local.repeat(4, 1)), dim=1)
        logits_local /= self.T

        # L_intra
        l_pos_intra = torch.einsum('nc,nc->n', [q_clip11, k_clip21]).unsqueeze(-1)
        l_neg_intra1 = torch.einsum('nc,nc->n', [q_clip11, k_clip22]).unsqueeze(-1)
        l_neg_intra2 = torch.einsum('nc,nc->n', [q_clip11, k_clip23]).unsqueeze(-1)
        l_neg_intra3 = torch.einsum('nc,nc->n', [q_clip11, k_clip24]).unsqueeze(-1)
        logits_intra = torch.cat((l_pos_intra.repeat(3, 1), torch.cat((l_neg_intra1, l_neg_intra2, l_neg_intra3), dim=0)), dim=-1)
        logits_intra /= self.T


        # labels: positive key indicators
        labels_local = torch.zeros(logits_local.shape[0], dtype=torch.long).cuda()
        labels_intra = torch.zeros(logits_intra.shape[0], dtype=torch.long).cuda()

        # mask: binary mask for positive keys
        # handle sibling videos, e.g. for UCF101. It has no effect on K400
        mask_source = k_vsource.unsqueeze(1) == self.queue_vname.unsqueeze(0) # B,K
        mask = mask_source.clone()

        if not self.queue_is_full:
            self.queue_is_full = torch.all(self.queue_label != -1)
            if self.queue_is_full: print('\n===== queue is full now =====')

        if self.queue_is_full and (self.topk != 0):
            mask_sim = kf.matmul(self.queue_second.clone().detach())
            mask_sim[mask_source] = - np.inf # mask out self (and sibling videos)
            _, topkidx = torch.topk(mask_sim, self.topk, dim=1)
            topk_onehot = torch.zeros_like(mask_sim)
            topk_onehot.scatter_(1, topkidx, 1)
            mask[topk_onehot.bool()] = True

        mask = torch.cat([torch.ones((mask.shape[0],1), dtype=torch.long, device=mask.device).bool(),
                          mask], dim=1)

        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, kf, k_vsource)

        return logits, logits_local, logits_intra, mask.detach(), labels_local,  labels_intra
