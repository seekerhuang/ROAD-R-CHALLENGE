# ROAD-R Challenge Task-2
The code is built on top of [3D-RetinaNet for ROAD](https://github.com/gurkirt/road-dataset).

Based on the information provided, for Task 2, participants are allowed to train their models using the annotations from all 18 videos in the training and validation sets：

> **Task 2:** in this task the participants will be able to train their models using the annotations for *all* videos in the training and validation set. [See Here](https://eval.ai/web/challenges/challenge-page/2081/evaluation)

Therefore, we trained our model using the *all* videos in the training and validation set **without employing any data augmentation, complex post-processing, TTA, or model ensemble tricks**. After applying maxhs-based post-processing (with a threshold set at 0.4), we achieved an impressive **F1-score of 0.60 in Task 2**. This further demonstrates the robust generalization of our model construction.

The second task requires that the models' predictions are compliant with the 243 requirements provided in `constraints/requirements.txt`.

## Table of Contents
- <a href='#dep'>Dependencies and data preparation</a>
- <a href='#training'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#postprocessing'>Post-processing</a>
- <a href='#Acknowledgments'>Acknowledgments</a>

## Dependencies and data preparation
Please refer to the "environment" folder in the directory, where you can choose the .yml file for building.

```
conda env create -f environment.yml
conda activate base
```

or:

```
pip install -r requirements.txt
```

The `road` directory should look like this:

```
   road/
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

## Pretrained Model：

Please place the pre-trained models in the `/pretrainmodel` folder. You can obtain the pre-trained models from the link provided below.

| Model                                                        | Link                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| swin_base_patch244_window1677_sthv2.pth (optional)           | [swin-base-ssv2](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth) |
| swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth | [swin-large-k700](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth) |
| yolox_l.pth                                                  | [yolox-l](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
| vit-giant-p14_dinov2-pre_3rdparty_20230426-2934a630.pth      | [dinov2-giant](https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-giant-p14_dinov2-pre_3rdparty_20230426-2934a630.pth) |
| vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth      | [dinov2-large](https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth) |
| pretrained weight for head                                   | [pretrained weight for head](https://drive.google.com/drive/folders/1Kw6aMJ9D7PktVQkWfBf_KvAUAamTaEU-) |

## Training

To train the model, provide the following positional arguments:
 - `DATA_ROOT`: path to a directory in which `road` can be found, containing `road_test_v1.0.json`, `road_trainval_v1.0.json`, and directories `rgb-images` and `videos`.
 - `SAVE_ROOT`: path to a directory in which the experiments (e.g. checkpoints, training logs) will be saved.
 - `MODEL_PATH`: path to the directory containing the weights for the chosen backbone (e.g. `resnet50RCGRU.pth`).

Example train command (to be run from the root of this repository):

```
python main.py --RESUME=1 --TASK=2 --EXP_NAME="yourpath/road-dataset-master/SAVE/road/logic-ssl_cache_Lukasiewicz_8.0/resnet50RCGRU512-Pkinetics-b15s16x1x1-roadt1t2t3-h3x3x3-10-23-22-54-45x/"  --EXPDIR="yourpath/road-dataset-master/experiments/" --DATA_ROOT="yourpath/road-dataset-master/" --pretrained_model_path="yourpath/road-dataset-master/ROAD-R-2023-Challenge-main_me/pretrainmodel/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth" --pretrained_model_path2="yourpath/road-dataset-master/ROAD-R-2023-Challenge-main_me/pretrainmodel/pretrained_weights_task2.pth" --MODEL_PATH="yourpath/road-dataset-master/ROAD-R-2023-Challenge-main_me/kinetics-pt/" --SAVE_ROOT="yourpath/road-dataset-master/SAVE/" --MODE="train" --LOGIC="Lukasiewicz" --VAL_STEP=1 --LR=9e-5 --MAX_EPOCHS=25
```

Note: By default, Task 2 is trained using mixed precision. If you prefer not to use mixed precision training to achieve higher accuracy, you can simply comment out the relevant code in the train file.

## Testing 

Below is an example command to test a model.

```
python main.py --RESUME=22 --TASK=2 --MODEL_PATH="yourpath/road-dataset-master/ROAD-R-2023-Challenge-main_me/kinetics-pt/" --EXPDIR="yourpath/road-dataset-master/experiments/" --DATA_ROOT="yourpath/road-dataset-master/" --TEST_SUBSETS=test --EVAL_EPOCHS=22 --pretrained_model_path="yourpath/road-dataset-master/ROAD-R-2023-Challenge-main_me/pretrainmodel/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth" --pretrained_model_path2="yourpath/road-dataset-master/ROAD-R-2023-Challenge-main_me/pretrainmodel/pretrained_weights_task2.pth"  --SAVE_ROOT="yourpath/road-dataset-master/SAVE2/" --EXP_NAME="yourpath/road-dataset-master/SAVE/road/logic-ssl_cache_Lukasiewicz_8.0/resnet50RCGRU512-Pkinetics-b15s16x1x1-roadt1t2t3-h3x3x3-10-23-22-54-45x/" --MODE="gen_dets" --LOGIC="Lukasiewicz"
```

## Postprocessing

To postprocess the predictions, and thus guarantee that the requirements are satisfied, use the output `.pkl` file (from `EXP_NAME`) as input to the postprocessing module, based on the [MaxHS solver](https://github.com/fbacchus/MaxHS/tree/master), from `postprocessing/`.

Below is an example command to postprocess a `.pkl` file.

```
python post_processing_raw.py --file_path="yourpath_of_pkl" --requirements_path="yourpath/road-dataset-master/ROAD-R-2023-Challenge-main_me/constraints/WDIMACS_requirements.txt" 
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
