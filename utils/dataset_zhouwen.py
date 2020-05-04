from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def data_arg(self, pil_img, mask, scale):
        pil_img = pil_img.resize((630, 630))
        mask = mask.resize((630, 630))
        x0 = random.randint(0, 29)
        y0 = random.randint(0, 29)
        x1 = x0 + 600
        y1 = y0 + 600
        pil_img = pil_img.crop((x0, y0, x1, y1))
        mask = mask.crop((x0, y0, x1, y1))
        choice = [True, False]
        if (random.choice(choice)):
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if (random.choice(choice)):
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        pil_img = self.preprocess(pil_img, scale)
        mask = self.preprocess(mask, scale)
        return pil_img, mask

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.jpg')
        img_file = glob(self.imgs_dir + idx + '.jpg')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # img = self.preprocess(img, self.scale)
        # mask = self.preprocess(mask, self.scale)

        img, mask = self.data_arg(img, mask, self.scale)
        h = mask.shape[1]
        w = mask.shape[2]
        mask_label = np.zeros(shape=(h, w), dtype=np.int)

        for i in range(h):
            for j in range(w):
                if mask[0][i][j] == 1:
                    mask_label[i][j] = 1
                else:
                    mask_label[i][j] = 0

        # for i in range(h):
        #     for j in range(w):
        #         if (mask[:,i,j]-np.array([0,0,0])).sum()>0:
        #             mask_label[i][j] = 1
        #         else:
        #             mask_label[i][j] = 0

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask_label)}
