# CVTCL: Cross-view Temporal Contrastive Learning for Self-supervised Video Representation

This repository contains the implementation of:

* local_InfoNCE (MoCo on videos)
* CVTCL

### Pretrain Instruction

* local_InfoNCE pretrain on UCF101-RGB
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 local_nce.py --net s3d --model infonce --moco-k 2048 \
--dataset ucf101-2clip --seq_len 32 --ds 1 --batch_size 8 \
--epochs 300 --schedule 250 280 -j 16
```

* local_InfoNCE pretrain on UCF101-Flow
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 local_nce.py --net s3d --model infonce --moco-k 2048 \
--dataset ucf101-f-2clip --seq_len 32 --ds 1 --batch_size 8 \
--epochs 300 --schedule 250 280 -j 16
```

* CVTCL pretrain on UCF101 for one cycle
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 local_coclr.py --net s3d --topk 5 --moco-k 2048 \
--dataset ucf101-2stream-2clip --seq_len 32 --ds 1 --batch_size 32 \
--epochs 100 --schedule 80 --name_prefix Cycle1-FlowMining_ -j 8 \
--pretrain {rgb_infoNCE_checkpoint.pth.tar} {flow_infoNCE_checkpoint.pth.tar}
```
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 local_coclr.py --net s3d --topk 5 --moco-k 2048 --reverse \
--dataset ucf101-2stream-2clip --seq_len 32 --ds 1 --batch_size 32 \
--epochs 100 --schedule 80 --name_prefix Cycle1-RGBMining_ -j 8 \
--pretrain {flow_infoNCE_checkpoint.pth.tar} {rgb_cycle1_checkpoint.pth.tar} 
```

### Finetune Instruction
`cd eval/`
e.g. finetune UCF101-rgb:
```
CUDA_VISIBLE_DEVICES=0,1 python main_classifier.py --net s3d --dataset ucf101 \
--seq_len 32 --ds 1 --batch_size 32 --train_what ft --epochs 500 --schedule 400 450 \
--pretrain {selected_rgb_pretrained_checkpoint.pth.tar}
```
then run the test with 10-crop (test-time augmentation is helpful, 10-crop gives better result than center-crop):
```
CUDA_VISIBLE_DEVICES=0,1 python main_classifier.py --net s3d --dataset ucf101 \
--seq_len 32 --ds 1 --batch_size 32 --train_what ft --epochs 500 --schedule 400 450 \
--test {selected_rgb_finetuned_checkpoint.pth.tar} --ten_crop
```

### Nearest-neighbour Retrieval Instruction
`cd eval/`
e.g. nn-retrieval for UCF101-rgb
```
CUDA_VISIBLE_DEVICES=0 python main_classifier.py --net s3d --dataset ucf101 \
--seq_len 32 --ds 1 --test {selected_rgb_pretrained_checkpoint.pth.tar} --retrieval
```

### Dataset
* RGB for UCF101: [[download-from-server]](http://thor.robots.ox.ac.uk/~vgg/data/CoCLR/ucf101_rgb_lmdb.tar) [[download-from-gdrive]](https://drive.google.com/file/d/1jVqBWl6iHYzcnb0IZ5ezpH_uK5jdHtoF/view?usp=sharing) (tar file, 29GB, packed with lmdb)
* TVL1 optical flow for UCF101: [[download-from-server]](http://thor.robots.ox.ac.uk/~vgg/data/CoCLR/ucf101_flow_lmdb.tar) [[download-from-gdrive]](https://drive.google.com/file/d/1NRElvRyVKX8siVu5HFKOETn4uqnzM4GH/view?usp=sharing) (tar file, 20.5GB, packed with lmdb)



