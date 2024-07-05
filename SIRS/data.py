##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: processing data

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
from vocab import deserialize_vocab
from PIL import Image
import cv2

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, vocab, opt, weights=False):
        self.vocab = vocab
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']

        # Captions
        self.captions = []
        self.maxlength = 0

        if data_split == 'check':
            self.images = []
            for image_name in os.listdir(self.seg_path):
                self.images.append('  '+image_name.replace('png', 'tif')+' ')
            with open(self.loc+'train_caps_verify.txt', 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())
            self.captions = self.captions[:len(self.images)]        
        elif data_split != 'test':
            with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
            
        if 'seg_path' in opt['dataset']:
            self.seg_path = opt['dataset']['seg_path']
            print('Initializing Seg-IR {} dataloader...'.format(data_split))
            #################################################
            ### TODO: initialize weights
            # self.files = self.recursive_glob(rootdir=self.seg_path, suffix='.png')
            self.files = ['mask_'+str(name)[2:-1].replace('tif', 'png') for name in self.images]
            self.class_names = [
                'Background',  
                'Plane',
                'Boat',
                'StorageTank',
                'Pond',
                'River',
                'Beach',
                'Playground',
                'SwimmingPool',
                'Court',
                'BaseballField',
                'Center',
                'Church',
                'Stadium',
                # 'Viaduct',
                'Bridge',
                # 'Road',
                # 'NaturalLandform',
                # 'Residential',
                # 'Industrial',         
            ]
            if weights == True:
                pixel_num = np.zeros([len(self.class_names)])
                for mask_name in self.files:
                    mask_path = self.seg_path + mask_name
                    # print(mask_path)
                    mask = cv2.imread(mask_path)
                    mask = mask.flatten() / 10
                    inds, times=np.unique(mask, return_counts=True)
                    # print('inds: ', inds)
                    # print('times: ', times)
                    for ind, time in zip(inds, times):
                        pixel_num[int(ind)] += time
                self.weights = torch.from_numpy(np.max(pixel_num) / pixel_num).float()
                print('weights: ', self.weights)
            ###################################################
            print('Initialization complete!')
        else:
            print('Initializing Non-Seg dataloader...')
            self.seg_path = False
        self.data_split = data_split
        if data_split == "train" or data_split == "check":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                # transforms.RandomRotation(0, 90),
                transforms.RandomRotation(90),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        caption = self.captions[index]

        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8'))
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]


        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)

        image = Image.open(self.img_path + str(self.images[img_id])[2:-1]).convert('RGB')
        # image = np.transpose(image, (2, 0, 1))
        # image = torch.tensor(np.array(Image.open(self.img_path + str(self.images[img_id])[2:-1]).convert('RGB')))
        if not self.seg_path:
            image = self.transform(image)  # torch.Size([3, 256, 256])
            return image, caption, tokens_UNK, index, img_id
        else:
            seg = Image.open(self.seg_path + self.files[img_id]).convert('L')
            # sample = {'image': image, 'label': seg}
            # train_set = self.transform(sample)
            from torchvision.transforms import functional as F
            ############################################################################
            ### TODO: change to origin version, check results in v5_origin
            # if self.data_split != 'test':
            #     ## resize
            #     image = F.resize(image, 278, Image.BILINEAR)
            #     seg = F.resize(seg, 278, Image.BILINEAR)
            
            #     ## random rotation
            #     angle = float(torch.empty(1).uniform_(float(0), float(90)).item())
            #     image = F.rotate(image, angle, False, False, None, None)
            #     seg = F.rotate(seg, angle, False, False, None, None)
            
            #     ## random crop
            #     w, h = F._get_image_size(image)
            #     th, tw = 256, 256
            #     i = torch.randint(0, h - th + 1, size=(1, )).item()
            #     j = torch.randint(0, w - tw + 1, size=(1, )).item()
            #     image = F.crop(image, i, j, th, tw)
            #     seg = F.crop(seg, i, j, th, tw)
            # else:
            #     ## resize
            #     image = F.resize(image, 256, Image.BILINEAR)
            #     seg = F.resize(seg, 256, Image.BILINEAR)

            ## resize 
            image = F.resize(image, 278, Image.BILINEAR)
            seg = F.resize(seg, 278, Image.BILINEAR)
            
            ## random rotation
            angle = float(torch.empty(1).uniform_(float(0), float(90)).item())
            image = F.rotate(image, angle, False, False, None, None)
            seg = F.rotate(seg, angle, False, False, None, None)
            
            ## random crop
            w, h = F._get_image_size(image)
            # w, h = F.get_image_size(image)
            th, tw = 256, 256
            i = torch.randint(0, h - th + 1, size=(1, )).item()
            j = torch.randint(0, w - tw + 1, size=(1, )).item()
            image = F.crop(image, i, j, th, tw)
            seg = F.crop(seg, i, j, th, tw)
            
            # ## resize
            # image = F.resize(image, 256, Image.BILINEAR)
            # seg = F.resize(seg, 256, Image.BILINEAR)
            ############################################################################
            
            ## to tensor
            image = F.to_tensor(image)
            seg = F.to_tensor(seg)
            
            ## normalize
            image = F.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), False)
            seg = seg * 255 / 10
            seg = seg.int()
        
            # return image, seg, caption, tokens_UNK, index, img_id
            return image, seg, caption, tokens_UNK, index, img_id


    def __len__(self):
        return self.length


def collate_fn(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, captions, tokens, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l !=0 else 1 for l in lengths]

    return images, targets, lengths, ids

def collate_Seg_fn(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, segs, captions, tokens, ids, img_ids = zip(*data)
    # images, segs = train_set['image'], train_set['label']

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    segs = torch.stack(segs, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l !=0 else 1 for l in lengths]

    return (images, segs), targets, lengths, ids


def get_precomp_loader(data_split, vocab, batch_size=100,
                       shuffle=True, num_workers=0, opt={}, weights=False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_split, vocab, opt, weights=weights)

    if 'seg_path' in opt['dataset']:
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                pin_memory=False,
                                                collate_fn=collate_Seg_fn,
                                                num_workers=num_workers)
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                pin_memory=False,
                                                collate_fn=collate_fn,
                                                num_workers=num_workers)
    if weights == True:
        return data_loader, dset.weights
    else:
        return data_loader, None

def get_loaders(vocab, opt):
    if opt['model']['name'] == 'IRSeg' and 'seg_path' in opt['dataset'].keys():
        weights = True
    else:
        weights = False
    train_loader, weights = get_precomp_loader( 'train', vocab,
                                opt['dataset']['batch_size'], True, opt['dataset']['train_workers'], opt=opt, weights=weights)
    val_loader = get_precomp_loader( 'val', vocab,
                                    opt['dataset']['batch_size_val'], False, opt['dataset']['test_workers'], opt=opt)
    return train_loader, val_loader, weights


def get_test_loader(vocab, opt):
    test_loader, _ = get_precomp_loader( 'test', vocab,
                                    opt['dataset']['batch_size_val'], False, opt['dataset']['test_workers'], opt=opt)
    return test_loader
