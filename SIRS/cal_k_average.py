##############################################
### DATE: 20230803
### AUTHOR: zzc
### TODO: calculate the average of k-folds results

import os
import sys
import numpy as np
'''
sys.argv[1]: 
rsitmd_irmim_mr0.4
rsitmd_irmim_mr0.6
rsitmd_irmim_mr0.2
rsitmd_irmim_mr0.1
rsitmd_irmim_mr0.15
rsitmd_irmim_mr0.05

rsitmd_irseg_v3
rsitmd_irseg_v4
rsitmd_irseg_v5
rsitmd_irseg_v6
rsitmd_irseg_v5_full
rsitmd_irseg_v5_all
rsitmd_irseg_v8
rsitmd_irseg_v8_IR80SS20
rsitmd_irseg_v8_IR99SS1
rsitmd_irseg_v8_IR20SS80
rsitmd_irseg_v8_IR1SS99
rsitmd_irseg_v9_woSSA_IR20SS80
rsitmd_irseg_v10_SSA_maxPooling
rsitmd_irseg_v10_SSA_mPBN
rsitmd_irseg_v10_SSA_mPGN
rsitmd_irseg_v10_SSA_aPGN
rsicd_irseg_v1_noSeg
rsitmd_irseg_v10_SSA_aPBN
rsitmd_irseg_v10_SSA_aP
'''

# root_dir = os.path.join('/root/autodl-tmp/zzc_backup/archive/IRMIM_series/', sys.argv[1])
# root_dir = os.path.join('/root/zzc/archive/IRSeg_series/', sys.argv[1])
root_dir = os.path.join('/root/autodl-tmp/zzc_backup/archive/IRSeg_series/', sys.argv[1])
ss_eval = float(sys.argv[2])

def get_IR_metrics(content):
    # r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t
    if content[-3][1:8] == 'average':
        # print(content[-5].split(' ')[1].split(':')[-1])
        r1i = float(content[-5].split(' ')[1].split(':')[-1])
        r5i = float(content[-5].split(' ')[2].split(':')[-1])
        r10i = float(content[-5].split(' ')[3].split(':')[-1])
        r1t = float(content[-4].split(' ')[1].split(':')[-1])
        r5t = float(content[-4].split(' ')[2].split(':')[-1])
        r10t = float(content[-4].split(' ')[3].split(':')[-1])
        mr = float(content[-3].split(':')[-1])
        return (r1i, r5i, r10i, r1t, r5t, r10t, mr)
    else:
        return (0,0,0,0,0,0,0)

def get_SS_metrics(content):
    # Acc, Acc_class, mIoU, FWIoU
    if content[-3][1:8] == 'average':
        acc = []
        acc_class = []
        miou = []
        fwiou = []
        # print(content[-5].split(' ')[1].split(':')[-1])
        for line in content:
            if line[0] == '{':
                tmp = eval(line)
                acc.append(tmp['Acc'])
                acc_class.append(tmp['Acc_class'])
                miou.append(tmp['mIoU'])
                fwiou.append(tmp['FWIoU'])
        acc = sum(acc) / len(acc)
        acc_class = sum(acc_class) / len(acc_class)
        miou = sum(miou) / len(miou)
        fwiou = sum(fwiou) / len(fwiou)
        return (acc, acc_class, miou, fwiou)
    else:
        return (0,0,0,0)

r1is = []
r5is = []
r10is = []
r1ts = []
r5ts = []
r10ts = []
mrs = []

accs = []
accs_class = []
mious = []
fwious = []

k_num = 0
for k_th in os.listdir(root_dir):
    k_dir = os.path.join(root_dir, k_th)
    if os.path.isdir(k_dir):
        # print('k_dir: ', k_dir)
        for file in os.listdir(k_dir):
            if file.split('.')[-1] == 'txt':
                # print('file: ', file)
                file_path = os.path.join(k_dir, file)
                # print('file_path: ', file_path)
                with open(file_path, 'r') as f:
                    content = f.readlines()
                    # results.append(float(content[-3].split(':')[-1])) 
                    #################################################################
                    r1i, r5i, r10i, r1t, r5t, r10t, mr = get_IR_metrics(content)
                    if mr == 0 :
                        print(k_th)
                        continue
                    r1is.append(r1i)
                    r5is.append(r5i)
                    r10is.append(r10i)
                    r1ts.append(r1t)
                    r5ts.append(r5t)
                    r10ts.append(r10t)
                    mrs.append(mr)
                    #################################################################
                    if ss_eval:
                        acc, acc_class, miou, fwiou = get_SS_metrics(content)
                        accs.append(acc)
                        accs_class.append(acc_class)
                        mious.append(miou)
                        fwious.append(fwiou)
                    #################################################################
                    k_num += 1
                    
assert k_num == len(mrs) == len(r1is)  
print('Exp {} average {}-folds'.format(root_dir.split('/')[-1], k_num))
print('    mr is {:.2f}, std is {:.2f}, up is {:.2f}, down is {:.2f}'.format(sum(mrs)/len(mrs), np.std(mrs), sum(mrs)/len(mrs)+np.std(mrs), sum(mrs)/len(mrs)-np.std(mrs)))
print('    r1i is {:.2f}, std is {:.2f}, up is {:.2f}, down is {:.2f}'.format(sum(r1is)/len(r1is), np.std(r1is), sum(r1is)/len(r1is)+np.std(r1is), sum(r1is)/len(r1is)-np.std(r1is)))
print('    r5i is {:.2f}, std is {:.2f}, up is {:.2f}, down is {:.2f}'.format(sum(r5is)/len(r5is), np.std(r5is), sum(r5is)/len(r5is)+np.std(r5is), sum(r5is)/len(r5is)-np.std(r5is)))
print('    r10i is {:.2f}, std is {:.2f}, up is {:.2f}, down is {:.2f}'.format(sum(r10is)/len(r10is), np.std(r10is), sum(r10is)/len(r10is)+np.std(r10is), sum(r10is)/len(r10is)-np.std(r10is)))
print('    r1t is {:.2f}, std is {:.2f}, up is {:.2f}, down is {:.2f}'.format(sum(r1ts)/len(r1ts), np.std(r1ts), sum(r1ts)/len(r1ts)+np.std(r1ts), sum(r1ts)/len(r1ts)-np.std(r1ts)))
print('    r5t is {:.2f}, std is {:.2f}, up is {:.2f}, down is {:.2f}'.format(sum(r5ts)/len(r5ts), np.std(r5ts), sum(r5ts)/len(r5ts)+np.std(r5ts), sum(r5ts)/len(r5ts)-np.std(r5ts)))
print('    r10t is {:.2f}, std is {:.2f}, up is {:.2f}, down is {:.2f}'.format(sum(r10ts)/len(r10ts), np.std(r10ts), sum(r10ts)/len(r10ts)+np.std(r10ts), sum(r10ts)/len(r10ts)-np.std(r10ts)))

if ss_eval:
    assert k_num == len(mrs) == len(r1is) == len(accs)     
    print('=========================================================================')

    print('    acc is {:.4f}, std is {:.4f}'.format(sum(accs)/len(accs), np.std(accs)))
    print('    acc_class is {:.4f}, std is {:.4f}'.format(sum(accs_class)/len(accs_class), np.std(accs_class)))
    print('    miou is {:.4f}, std is {:.4f}'.format(sum(mious)/len(mious), np.std(mious)))
    print('    fwiou is {:.4f}, std is {:.4f}'.format(sum(fwious)/len(fwious), np.std(fwious)))