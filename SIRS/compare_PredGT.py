##############################################
### DATE: 20230913
### AUTHOR: zzc
### TODO: compare IR preds with GTs
### REQ: .yaml .json

import yaml
import json

pred_path = '/root/autodl-tmp/zzc_backup/code/private/SIRS/output/SIRS_i2t_dict.yaml'
gt_path = '/root/autodl-tmp/zzc_backup/data/RSITMD/dataset_RSITMD.json'

with open(pred_path, 'r') as handle:
    pred = yaml.load(handle, Loader=yaml.FullLoader)
    
with open(gt_path, 'r') as handle:
    gt = json.load(handle)

match_dict = {}
for pred_name in pred:
    k = 0
    for pred_t in pred[pred_name]:
        for gt_item in gt['images']:
            if gt_item['filename'] == pred_name:
                gt_t = []
                for sen in gt_item['sentences']:
                    gt_t.append(sen['raw'])
                break
        if pred_t in gt_t:
            k += 1
    if k >= 3:
        match_dict[pred_name] = k

print(match_dict)