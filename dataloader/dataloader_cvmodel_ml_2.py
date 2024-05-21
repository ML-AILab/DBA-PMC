import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 2.define dataset
class CVModel_ML_2_Dataset(Dataset):
    def __init__(self, label_list, transforms=None, train=False, val=False):
        self.train = train
        self.val = val
        self.transforms = transforms

        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row["filename"], row["label"], row['label_2']))
        self.imgs = imgs

    def __getitem__(self, index):
        filename, label, label_2 = self.imgs[index]
        img = Image.open(filename).convert('RGB')
        img = self.transforms(img)
        return img, label, label_2

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    imgs = []
    label = []
    label_2 = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        label_2.append(sample[2])

    return torch.stack(imgs, 0), label, label_2


def get_files_fromtxt(root, mode, num_classes):
    if mode == 'train' or mode == 'val':
        r_txt = open(root, 'r', encoding='utf-8')
        all_data_path, labels, labels_2 = [], [], []
        count = 0
        for inf in r_txt:
            img_path = inf[:-1].split('\t')[0]
            if len(inf[:-1].split('\t')) > 1:
                label_list = inf[:-1].split('\t')[1].split(',')
            else:
                label_list = []
            
            if len(inf[:-1].split('\t')) > 1:
                count += 1
                label_list_2 = [0]
            else:
                label_list_2 = []

            label_list = [int(x) for x in label_list if x != '']
            label_npy = np.zeros(num_classes)
            label_npy[label_list] = 1.0
            label_npy_2 = np.zeros(1)
            label_npy_2[label_list_2] = 1.0

            all_data_path.append(img_path)
            labels.append(label_npy)
            labels_2.append(label_npy_2)

        all_files = pd.DataFrame({"filename": all_data_path, "label": labels, 'label_2': labels_2})
     
        if mode == 'train':
            print('imgs of train:{}'.format(len(labels)))
        elif mode == 'val':
            print('imgs of valid:{}'.format(len(labels)))
            print(count/len(labels), 1-count/len(labels))

        return all_files

    else:
        print('error')


def get_files_fromtxt_2(root, mode, num_classes):
    if mode == 'train' or mode == 'val':
        r_txt = open(root, 'r', encoding='utf-8')
        all_data_path, labels, labels_2 = [], [], []
        count = 0
        for inf in r_txt:
            img_path = inf[:-1].split('\t')[0]
            if len(inf[:-1].split('\t')) > 1:
                label_list = inf[:-1].split('\t')[1].split(',')
            else:
                label_list = []
            
            tmp = inf[:-1].split('\t')
            
            if len(tmp) > 1:
                if '1' in tmp[1] or '2' in tmp[1] or '3' in tmp[1] or '4' in tmp[1] or '5' in tmp[1] or '6' in tmp[1]:
                    count += 1
                    label_list_2 = [0]
                else:
                    label_list_2 = []
            else:
                label_list_2 = []

            label_list = [int(x) for x in label_list if x != '']
            label_npy = np.zeros(num_classes)
            label_npy[label_list] = 1.0
            label_npy_2 = np.zeros(1)
            label_npy_2[label_list_2] = 1.0

            all_data_path.append(img_path)
            labels.append(label_npy)
            labels_2.append(label_npy_2)

        all_files = pd.DataFrame({"filename": all_data_path, "label": labels, 'label_2': labels_2})
     
        if mode == 'train':
            print('imgs of train:{}'.format(len(labels)))
        elif mode == 'val':
            print('imgs of valid:{}'.format(len(labels)))
            print(count/len(labels), 1-count/len(labels))

        return all_files

    else:
        print('error')

