# Only-Dinov2 Branch Task-1 
The code is built on top of [3D-RetinaNet for ROAD](https://github.com/gurkirt/road-dataset).

The first task requires developing models for scenarios where only little annotated data is available at training time. 
More precisely, only 3 out of 15 videos (from the training partition train_1 of the ROAD-R dataset) are used for training the models in this task.
The videos' ids are: 2014-07-14-14-49-50_stereo_centre_01, 2015-02-03-19-43-11_stereo_centre_04, and 2015-02-24-12-32-19_stereo_centre_04.

**Note: The results obtained from this branch (without data augmentation, TTA, or any complex post-processing) are approximately around 0.23. However, the scores for agent and location are significantly higher. Therefore, it is used to integrate with the TBSD model in the parent directory to achieve higher results.**

## Table of Contents
- <a href='#dep'>Dependencies and data preparation</a>
- <a href='#dep'>Pretrained Model</a>
- <a href='#training'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#Acknowledgments'>Acknowledgments</a>

## Dependencies and Data Preparation
The environment configuration is the same as the TBSD in the parent directory.

The `road` directory should look like this:

```
   /.../road/
        - road_trainval_v1.0.json
        - videos/
            - 2014-06-25-16-45-34_stereo_centre_02
            - 2014-06-26-09-53-12_stereo_centre_02
            - ........
        - rgb-images
            - 2014-06-25-16-45-34_stereo_centre_02/
                - 00001.jpg
                - 00002.jpg
                - .........*.jpg
            - 2014-06-26-09-53-12_stereo_centre_02
                - 00001.jpg
                - 00002.jpg
                - .........*.jpg
            - ......../
                - ........*.jpg
```

And you need to place the directory for configuring the dataset in the parent level of this file directory, i.e., the parent level of the directory where the README.md file is located. Please refer to [road-dataset](https://github.com/gurkirt/road-dataset) for the specific format.

## Pretrained Model

Please place the pre-trained models in the `/pretrainmodel` folder. You can obtain the pre-trained models from the link provided below.

| Model                                                        | Link                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| swin_base_patch244_window1677_sthv2.pth (optional)           | [swin-base-ssv2](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth) |
| swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth | [swin-large-k700](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth) |
| yolox_l.pth                                                  | [yolox-l](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
| vit-giant-p14_dinov2-pre_3rdparty_20230426-2934a630.pth      | [dinov2-giant](https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-giant-p14_dinov2-pre_3rdparty_20230426-2934a630.pth) |
| vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth      | [dinov2-large](https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth) |
| pretrained weight for head                                   | [pretrained weight for head](https://drive.google.com/drive/folders/1Kw6aMJ9D7PktVQkWfBf_KvAUAamTaEU-) |

Note: You may need to run `get_kinetics_weights.sh` (included in the [ROAD-R Challenge](https://sites.google.com/view/road-r/) ) to obtain the file named resnet50RCGRU.pth. Otherwise, you may encounter an error.

## Training

To train the model, provide the following positional arguments:
 - `DATA_ROOT`: path to a directory in which `road` can be found, containing `road_test_v1.0.json`, `road_trainval_v1.0.json`, and directories `rgb-images` and `videos`.
 - `SAVE_ROOT`: path to a directory in which the experiments (e.g. checkpoints, training logs) will be saved.
 - `MODEL_PATH`: path to the directory containing the weights for the chosen backbone (e.g. `resnet50RCGRU.pth`).

The remaining experimental details and logs can be found in `actual_task1_logs_TBSD` and `actual_task1_logs_only_dinov2`. The folder `all_history_logs` in the main directory contains all the experimental information for tasks one and two.

Example train command (to be run from the root of this repository):

```
python main.py --TASK=1 --DATA_ROOT="yourpath/road-dataset-master/" --pretrained_model_path="yourpath/road-dataset-master/ROAD-R-2023-Challenge-main_me/pretrainmodel/swin_base_patch244_window1677_sthv2.pth" --pretrained_model_path2="yourpath/road-dataset-master/ROAD-R-2023-Challenge-main_me/pretrainmodel/pretrained_weights_task1.pth" --MODEL_PATH="yourpath/road-dataset-master/ROAD-R-2023-Challenge-main_me/kinetics-pt/" --SAVE_ROOT="yourpath/road-dataset-master/SAVE/" --MODE="train" --LOGIC="Lukasiewicz" --VAL_STEP=1 --LR=6e-5 --MAX_EPOCHS=25
```

## Testing 
Below is an example command to test a model.

```
python main.py --RESUME=20 --TASK=1 --LOGIC="Lukasiewicz" --EXPDIR="/root/autodl-tmp/road-dataset-master/experiments/" --DATA_ROOT="/root/autodl-tmp/road-dataset-master/" --pretrained_model_path="/root/autodl-tmp/road-dataset-master/ROAD-R-2023-Challenge-main_me/pretrainmodel/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth" --pretrained_model_path2="/root/autodl-tmp/road-dataset-master/ROAD-R-2023-Challenge-main_me/pretrainmodel/pretrained_weights_task1.pth" --MODEL_PATH="/root/autodl-tmp/road-dataset-master/ROAD-R-2023-Challenge-main_me/kinetics-pt/" --SAVE_ROOT="/root/autodl-tmp/road-dataset-master/SAVE/" --MODE="gen_dets" --TEST_SUBSETS=test --EVAL_EPOCHS=20 --EXP_NAME="/root/autodl-tmp/road-dataset-master/SAVE/road/logic-ssl_cache_Lukasiewicz_8.0/resnet50RCGRU512-Pkinetics-b8s16x1x1-roadt1-h3x3x3-10-20-18-17-18x/"
```

## Acknowledgments

[1] [road-dataset](https://github.com/gurkirt/road-dataset)

[2] [ROAD-R-2023-Challenge](https://github.com/mihaela-stoian/ROAD-R-2023-Challenge)

[3] [3D-RetinaNet for ROAD](https://github.com/gurkirt/road-dataset)

[4] [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)

[5] [dinov2](https://github.com/facebookresearch/dinov2)

[6] [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

[7] [mmpretrain](https://github.com/open-mmlab/mmpretrain)

[8] [mmaction2](https://github.com/open-mmlab/mmaction2)

