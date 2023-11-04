import pickle
import numpy as np
from collections import defaultdict
from itertools import product
from tqdm import tqdm




def compute_iou(box, box2):
    xmin = max(box[0], box2[0])
    ymin = max(box[1], box2[1]) 
    xmax = min(box[2], box2[2])
    ymax = min(box[3], box2[3])

    iw = np.maximum(xmax - xmin, 0.) 
    ih = np.maximum(ymax - ymin, 0.)
    
    if iw > 0 and ih > 0:
        intsc = iw*ih 
    else:
        intsc = 0.0
        
    union = (box2[2] - box2[0]) * (box2[3] - box2[1]) + \
           (box[2] - box[0]) * (box[3] - box[1]) - intsc 
           
    iou = intsc/union
    return iou


def nms(bboxes, iou_threshold):
    if len(bboxes) == 0:
        return []
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    keep = []
    while len(bboxes) > 0:
        keep.append(bboxes[0])
        if len(bboxes) == 1:
            break
        ious = [compute_iou(keep[-1][:4], bboxes[i][:4]) for i in range(1, len(bboxes))]
        idx = np.where(np.array(ious) <= iou_threshold)[0]
        bboxes = [bboxes[i + 1] for i in idx]
    return keep




'''
This code is used to integrate the best results from different models and different epochs. 
The index following 'labels' are adjustable and should be chosen based on the best results of your trained model on the validation set. 
The parameters provided here are an example, integrating the "only_dinov2" agent and localization results into the original "tbsd" model's results.
'''
def ensem_labels(base_bbox, new_bbox):
  base_bbox['labels'][:10] = new_bbox['labels'][:10]
  base_bbox['labels'][-12:] = new_bbox['labels'][-12:]
  return base_bbox

def merge_pkls(base_pkl, new_pkl):
  merged_pkl = base_pkl.copy()

  for video_name in tqdm(new_pkl):
    for frame_name in new_pkl[video_name]:
      # if frame_name not in merged_pkl[video_name]:
      #   merged_pkl[video_name][frame_name] = new_pkl[video_name][frame_name]
      #   continue
      
      for new_bbox in new_pkl[video_name][frame_name]:
        max_iou = -1
        max_idx = -1
        for i, base_bbox in enumerate(merged_pkl[video_name][frame_name]):
          iou = compute_iou(new_bbox['bbox'], base_bbox['bbox'])
          if iou > max_iou:
            max_iou = iou
            max_idx = i
        
        if max_iou > 0.88:
          merged_pkl[video_name][frame_name][max_idx] = ensem_labels(
            merged_pkl[video_name][frame_name][max_idx], new_bbox)

  return merged_pkl
              
# example  
with open('./pred_detections-20-140-45_test_50_dinoswin.pkl', 'rb') as f:
  base_pkl = pickle.load(f)

with open('./pred_detections-23-120-50_test_50_onlydino.pkl', 'rb') as f:
  new_pkl = pickle.load(f)
  
merged_pkl = merge_pkls(base_pkl, new_pkl)

with open('./final_pred_detections-20-140-45_test_50_dinoswin.pkl', 'wb') as f:
  pickle.dump(merged_pkl, f)