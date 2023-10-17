import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path

class Lyme(torch.utils.data.Dataset):

    def __init__(self, dataroot = None, train= True, val = False, transform = None):
        self.train = train
        self.val = val
        # self.CLF_RAW_SPLITS_DIR = Path("/home/hyuan4/raw_splits")
        # self.ROOT_CLF_IMG_DIR = Path("/home/hyuan4/scratch4-ycao43/lyme_image")
        self.CLF_RAW_SPLITS_DIR = Path('/home/haolin/Projects/lyme/code/disease_clf_assets/raw_splits')
        self.ROOT_CLF_IMG_DIR = Path("/home/haolin/Projects/lyme/media/HDD1/hadzia1/lyme_data/images_July2019")
        if self.train:
            df = pd.read_csv(str(Path(self.CLF_RAW_SPLITS_DIR ) / 'NO_vs_EM_vs_HZ_vs_TC_train.csv'))
        elif self.val:
            df = pd.read_csv(str(Path(self.CLF_RAW_SPLITS_DIR) / 'NO_vs_EM_vs_HZ_vs_TC_val.csv'))
        else:
            df = pd.read_csv(str(Path(self.CLF_RAW_SPLITS_DIR) / 'NO_vs_EM_vs_HZ_vs_TC_test.csv'))
        df.columns = ['images', 'disease_labels']

        self.img_path, self.targets = self.prune_broken_images(df['images'].to_list(),
                                                           df['disease_labels'].to_list())
        self.transform = transform
        self.prefix = True

    def prune_broken_images(self, img_path, targets):
        clean_im_paths = []
        clean_lbls = []
        valid_format = ["png", "jpeg", "jpg"]
        for im_pth, lbl in zip(img_path, targets):
            if im_pth.split(".")[-1] in valid_format:
                clean_im_paths.append(im_pth)
                clean_lbls.append(lbl)
        img_path = clean_im_paths
        labels = clean_lbls
        return img_path, labels


    def __getitem__(self, item):
        if self.prefix:
            img_path = self.ROOT_CLF_IMG_DIR/self.img_path[item]
        else:
            img_path = self.img_path[item]
        img = Image.open(img_path)
        if transforms.functional.pil_to_tensor(img).shape[0] != 3:        #For CMYK images or images with 1 channel
            img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        targets = self.targets[item]
        return img, targets

    def __len__(self):
        return len(self.img_path)


