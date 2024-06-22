##############################################
### DATE: 20230817
### AUTHOR: zzc
### TODO: merge image and mask into one for visualization

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

### ref: https://tool.oschina.net/commons?type=3
mask_color_dict = {
    ### black
    'Background':    [0, 0, 0], 
    ### orange
    'Plane':         [255, 165, 0],
    ### cyan
    'Boat':          [0, 255, 255],
    ### yellow
    'StorageTank':   [255, 255, 0],
    ### MediumBlue
    'Pond':          [0, 0, 205],
    ### DodgerBlue
    'River':         [30, 144, 255],
    ### Gold
    'Beach':         [255, 215, 0],
    ### OliveDrab
    'Playground':    [107, 142, 35],
    ### RoyalBlue
    'SwimmingPool':  [65, 105, 225],
    ### GreenYellow
    'Court':         [173, 255, 47],
    ### Gold4
    'BaseballField': [139, 117, 0],
    ### Tomato
    'Center':        [255, 99, 71],
    ### white
    'Church':        [255, 255, 255],
    ### HotPink
    'Stadium':       [255, 105, 180],
    ### grey51
    'Bridge':        [130, 130, 130],
}

def mask2color(mask, mapping_dict, transparency=127):
    new_mask = np.zeros([mask.shape[0], mask.shape[1], 4]).astype(np.uint8)
    for i, key in enumerate(mapping_dict):
        if len(np.where(mask==i*10)[0]) > 0:
            new_mask[mask==i*10] = mapping_dict[key] + [transparency]
    return new_mask

def show_merged(image_path, mask_path, mapping_dict, show=True, save=False):
    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_rgb, contours, -1, (0, 0, 0), 2)
    mask = mask2color(mask, mapping_dict)
    # plt.figure(figsize=(10,10))
    fig, ax = plt.subplots()
    plt.imshow(image_rgb, aspect="equal")
    plt.imshow(mask, aspect="equal")
    plt.axis('off')
    plt.xticks([])
    # if show:
    #     plt.show()
    if save:
        plt.margins(0, 0)
        plt.savefig(save, bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.close()

### for all 
# tmp = ''
# idx = 0
# image_list = os.listdir(image_dir)
# mask_list = os.listdir(mask_dir)
# image_list.sort()
# mask_list.sort()
# for iname, mname in zip(image_list, mask_list):
#     if iname.split('_')[0] != tmp:
#         tmp = iname.split('_')[0]
#         idx = 0
#     else:
#         idx += 1
#     if idx >= 3:
#         continue
#     print(iname)
#     image_path = os.path.join(image_dir, iname)
#     mask_path = os.path.join(mask_dir, mname)
#     save_path = os.path.join(save_dir, iname.replace('tif', 'png'))
#     show_merged(image_path, mask_path, mask_color_dict, save=save_path)

# print('DONE!')

### for single

# image_path = '../img_demo/airport_4691.tif'
# mask_path = '../seg_demo/mask_airport_4691.png'
# save_path = './merged_airport_4691.png'
image_path = '../img_demo/river_3583.tif'
mask_path = '../seg_demo/mask_river_3583.png'
save_path = './merged_river_3583.png'

show_merged(image_path, mask_path, mask_color_dict, save=save_path)
print('DONE!')