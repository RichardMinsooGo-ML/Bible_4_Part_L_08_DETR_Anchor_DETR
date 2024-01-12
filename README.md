
# Engilish
*  **DETR Theory** : [https://wikidocs.net/216404](https://wikidocs.net/216404) <br>
*  **Anchor DETR Theory** : [https://wikidocs.net/227361](https://wikidocs.net/227361) <br>
*  **Implementation** : [https://wikidocs.net/227362](https://wikidocs.net/227362)

# 한글
*  **DETR Theory** : [https://wikidocs.net/215294](https://wikidocs.net/215294) <br>
*  **Anchor DETR Theory** : [https://wikidocs.net/226027](https://wikidocs.net/226027) <br>
*  **Implementation** : [https://wikidocs.net/225906](https://wikidocs.net/225906)

This repository is folked from [https://github.com/yjh0410/DeTR-LAB](https://github.com/yjh0410/DeTR-LAB).
At this repository, simplification and explanation and will be tested at Colab Environment.


## DETR

I evaluate DETR by loading official pretrained weight.

| Model        |  backbone  |  FPS<sup><br>3090  |  FLOPs   |  Params |    AP    |  Weight  |
|--------------|------------|--------------------|----------|---------|----------|----------|
| DETR-R50     |    R-50    |  37                |  95.2 B  |  36.7 M |   41.7   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r50-e632da11.pth) |
| DETR-R50-DC5 |    R-50    |  20                |  162.1 B |  36.7 M |   43.0   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r50-dc5-f0fb7ef5.pth) |
| DETR-R101    |    R-101   |  25                |  174.7 B |  55.7 M |   43.1   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r101-2c7b67e5.pth) |
| DETR-R101-DC5|    R-101   |  14                |  241.6 B |  55.7 M |   44.3   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r101-dc5-a2e86def.pth) |

## Anchor DETR
I evaluate AnchorDETR by loading official pretrained weight.

| Model               |  backbone  |  FPS<sup><br>3090  |  FLOPs   |  Params |    AP    |  Weight  |
|---------------------|------------|--------------------|----------|---------|----------|----------|
| Anchor-DETR-R50     |    R-50    |       37           |  97.0 B  |  30.7 M |   42.1   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r50_c5.pth) |
| Anchor-DETR-R50-DC5 |    R-50    |       20           |  154.0 B |  30.7 M |   44.2   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r50_dc5.pth) |
| Anchor-DETR-R101    |    R-101   |       21           |  176.5 B |  49.7 M |   43.5   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r101_c5.pth) |
| Anchor-DETR-R101-DC5|    R-101   |       16           |  233.5 B |  49.7 M |   45.2   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r101_dc5.pth) |


## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_L_08_DETR_Anchor_DETR.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

```Shell
! pip install thop
```

## Step x. Download pretrained weight

```Shell
! wget https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r50-e632da11.pth
# ! wget https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r50-dc5-f0fb7ef5.pth
# ! wget https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r101-2c7b67e5.pth
# ! wget https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r101-dc5-a2e86def.pth

! wget https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r50_c5.pth
# ! wget https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r50_dc5.pth
# ! wget https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r101.pth
# ! wget https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r101_dc5.pth
```

## Demo - DETR
### Detect with Image
```Shell
! python demo.py --mode image \
                 --path_to_img /content/dataset/demo/images/ \
                 -v detr_r50 \
                 --cuda \
                 --weight /content/detr-r50-e632da11.pth \
                 --path_to_save /content/save_folder
                 # --show

# Check Result : /content/save_folder/image
```

### Detect with Video
```Shell
! python demo.py --mode video \
                 --path_to_vid /content/dataset/demo/videos/street.mp4 \
                 -v detr_r50 \
                 --cuda \
                 --weight /content/detr-r50-e632da11.pth \
                 --path_to_save /content/save_folder
                 # --show

# Check Result : /content/save_folder/image
```

### Detect with Camera
```Shell
# ! python demo.py --mode camera \
#                  -v detr_r50 \
#                  --cuda \
#                  --weight /content/detr-r50-e632da11.pth
                 # --show
```

## Demo - Anchor DETR
### Detect with Image
```Shell
! python demo.py --mode image \
                 --path_to_img /content/dataset/demo/images/ \
                 -v anchor_detr_r50 \
                 --cuda \
                 --weight /content/AnchorDETR_r50_c5.pth \
                 --path_to_save /content/save_folder
                 # --show
```

### Detect with Video
```Shell
! python demo.py --mode video \
                 --path_to_vid /content/dataset/demo/videos/street.mp4 \
                 -v anchor_detr_r50 \
                 --cuda \
                 --weight /content/AnchorDETR_r50_c5.pth \
                 --path_to_save /content/save_folder
                 # --show
```

### Detect with Camera
```Shell
# ! python demo.py --mode camera \
#                  -v anchor_detr_r50 \
#                  --cuda \
#                  --weight /content/AnchorDETR_r50_c5.pth
                 # --show
```

## Download COCO Dataset

```Shell
# COCO dataset download and extract

# ! wget http://images.cocodataset.org/zips/train2017.zip
! wget http://images.cocodataset.org/zips/val2017.zip
! wget http://images.cocodataset.org/zips/test2017.zip
# ! wget http://images.cocodataset.org/zips/unlabeled2017.zip

# ! unzip train2017.zip  -d dataset/COCO
! unzip val2017.zip  -d dataset/COCO
! unzip test2017.zip  -d dataset/COCO

# ! unzip unlabeled2017.zip -d dataset/COCO

# ! rm train2017.zip
# ! rm val2017.zip
# ! rm test2017.zip
# ! rm unlabeled2017.zip

! wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

! unzip annotations_trainval2017.zip -d dataset/COCO
# ! unzip stuff_annotations_trainval2017.zip
# ! unzip image_info_test2017.zip
# ! unzip image_info_unlabeled2017.zip

# ! rm annotations_trainval2017.zip
# ! rm stuff_annotations_trainval2017.zip
# ! rm image_info_test2017.zip
# ! rm image_info_unlabeled2017.zip

clear_output()
```

## Evaluate DETR
```Shell
! python eval.py -d coco-val \
                 --cuda \
                 -v detr_r50 \
                 --weight /content/detr-r50-e632da11.pth \
                 --root /content/dataset/
```

## Test DETR
```Shell
! python test.py -d coco \
                 --cuda \
                 -v detr_r50 \
                 --weight /content/detr-r50-e632da11.pth \
                 --root /content/dataset/ \
                 --save_folder /content/save_folder
                 # --show
```

## Evaluate Anchor-DETR
```Shell
! python eval.py -d coco-val \
                 --cuda \
                 -v anchor_detr_r50 \
                 --weight /content/AnchorDETR_r50_c5.pth \
                 --root /content/dataset/
```

## Test Anchor-DETR
```Shell
! python test.py -d coco \
                 --cuda \
                 -v anchor_detr_r50 \
                 --weight /content/AnchorDETR_r50_c5.pth \
                 --root /content/dataset/ \
                 --save_folder /content/save_folder
                 # --show
```

## Training at Single GPU : DETR

```Shell
# T4 GPU memory 10.4GB
! python train.py --cuda \
                  -d coco \
                  --root /content/dataset \
                  -v detr_r18 \
                  --eval_epoch 5 \
                  --no_warmup \
                  --aux_loss
```

```Shell
# Train
! python train.py --cuda \
                  -d coco \
                  --root /content/dataset \
                  -v detr_r50 \
                  --eval_epoch 5 \
                  --no_warmup \
                  --aux_loss \
                  --pretrained /content/detr-r50-e632da11.pth
                  # --resume weights/coco/detr_r50/detr-r50-e632da11.pth
```

```Shell
# Train
! python train.py --cuda \
                  -d coco \
                  --root /content/dataset \
                  -v detr_r50-DC5 \
                  --eval_epoch 5 \
                  --no_warmup \
                  --aux_loss
```

```Shell
# Train
! python train.py --cuda \
                  -d coco \
                  --root /content/dataset \
                  -v detr_r101 \
                  --eval_epoch 5 \
                  --no_warmup \
                  --aux_loss
```

```Shell
# Train
! python train.py --cuda \
                  -d coco \
                  --root /content/dataset \
                  -v detr_r101-DC5 \
                  --eval_epoch 5 \
                  --no_warmup \
                  --aux_loss
```

## Training at Single GPU : Anchor-DETR

```Shell
# T4 GPU memory 4.8 GB
! python train.py --cuda \
                  -d coco \
                  --root /content/dataset \
                  -v anchor_detr_r18 \
                  --eval_epoch 5 \
                  --no_warmup \
                  --aux_loss
```

```Shell
# T4-GPU memory 5.9GB
! python train.py --cuda \
                  -d coco \
                  --root /content/dataset \
                  -v anchor_detr_r50 \
                  --eval_epoch 5 \
                  --no_warmup \
                  --aux_loss \
                  --pretrained /content/AnchorDETR_r50_c5.pth
                  # --resume weights/coco/detr_r50/detr-r50-e632da11.pth
```

```Shell
# Train
! python train.py --cuda \
                  -d coco \
                  --root /content/dataset \
                  -v anchor_detr_r50-DC5 \
                  --eval_epoch 5 \
                  --no_warmup \
                  --aux_loss
```

```Shell
# Train
! python train.py --cuda \
                  -d coco \
                  --root /content/dataset \
                  -v anchor_detr_r101 \
                  --eval_epoch 5 \
                  --no_warmup \
                  --aux_loss
```

```Shell
# Train
! python train.py --cuda \
                  -d coco \
                  --root /content/dataset \
                  -v anchor_detr_r101-DC5 \
                  --eval_epoch 5 \
                  --no_warmup \
                  --aux_loss
```

## Training at Multi GPU 
```Shell
# 8 GPUs
# python -m torch.distributed.run --nproc_per_node=8 train.py \
#                                                     --cuda \
#                                                     -dist \
#                                                     -d coco \
#                                                     --root /mnt/share/ssd2/dataset/ \
#                                                     -v detr_r18 \
#                                                     --num_workers 4 \
#                                                     --eval_epoch 10 \
#                                                     --no_warmup \
#                                                     --aux_loss \
```


