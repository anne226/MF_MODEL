
import os
import sys
import pickle

import matplotlib.pyplot
from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

class CESM(Dataset):
    def __init__(self, base_dir=None, labeled_num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.targets = []
        # self.split = split
        self.transform = transform
        dir = os.listdir(self._base_dir)
        dir.sort()
        for name in dir:
            image = os.path.join(self._base_dir, name)
            self.sample_list.append(image)


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, 'r')

        LOW_ENERGY_CCL = h5f['LOW_ENERGY_CCL'][:].astype(np.float32)
        RECOMBINED_CCL = h5f['RECOMBINED_CCL'][:].astype(np.float32)
        LOW_ENERGY_CCR = h5f['LOW_ENERGY_CCR'][:].astype(np.float32)
        RECOMBINED_CCR = h5f['RECOMBINED_CCR'][:].astype(np.float32)
        LOW_ENERGY_MLOL = h5f['LOW_ENERGY_MLOL'][:].astype(np.float32)
        RECOMBINED_MLOL = h5f['RECOMBINED_MLOL'][:].astype(np.float32)
        LOW_ENERGY_MLOR = h5f['LOW_ENERGY_MLOR'][:].astype(np.float32)
        RECOMBINED_MLOR = h5f['RECOMBINED_MLOR'][:].astype(np.float32)
        label1 = h5f['label1'][()]
        label2 = h5f['label2'][()]
        label3 = h5f['label3'][()]
        label4 = h5f['label4'][()]


        LOW_ENERGY_CCL = cv2.resize(LOW_ENERGY_CCL, (256, 512))
        RECOMBINED_CCL = cv2.resize(RECOMBINED_CCL, (256, 512))
        LOW_ENERGY_CCR = cv2.resize(LOW_ENERGY_CCR, (256, 512))
        RECOMBINED_CCR = cv2.resize(RECOMBINED_CCR, (256, 512))
        LOW_ENERGY_MLOL = cv2.resize(LOW_ENERGY_MLOL, (256, 512))
        RECOMBINED_MLOL = cv2.resize(RECOMBINED_MLOL, (256, 512))
        LOW_ENERGY_MLOR = cv2.resize(LOW_ENERGY_MLOR, (256, 512))
        RECOMBINED_MLOR = cv2.resize(RECOMBINED_MLOR, (256, 512))



        seed = np.random.randint(255)
        if self.transform:
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY_CCL = Image.fromarray(np.uint8(LOW_ENERGY_CCL*255))
            torch.manual_seed(seed)
            LOW_ENERGY_CCR= Image.fromarray(np.uint8(LOW_ENERGY_CCR * 255))
            torch.manual_seed(seed)
            RECOMBINED_CCL = Image.fromarray(np.uint8(255 * RECOMBINED_CCL))
            torch.manual_seed(seed)
            RECOMBINED_CCR = Image.fromarray(np.uint8(255*RECOMBINED_CCR))
            torch.manual_seed(seed)
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY_MLOL = Image.fromarray(np.uint8(LOW_ENERGY_MLOL*255))
            torch.manual_seed(seed)
            LOW_ENERGY_MLOR= Image.fromarray(np.uint8(LOW_ENERGY_MLOR * 255))
            torch.manual_seed(seed)
            RECOMBINED_MLOL = Image.fromarray(np.uint8(255 * RECOMBINED_MLOL))
            torch.manual_seed(seed)
            RECOMBINED_MLOR = Image.fromarray(np.uint8(255*RECOMBINED_MLOR))
            torch.manual_seed(seed)
            LOW_ENERGY_CCL = self.transform(LOW_ENERGY_CCL)

            torch.manual_seed(seed)
            RECOMBINED_CCL=self.transform(RECOMBINED_CCL)
            torch.manual_seed(seed)
            LOW_ENERGY_CCR = self.transform(LOW_ENERGY_CCR)
            torch.manual_seed(seed)
            RECOMBINED_CCR = self.transform(RECOMBINED_CCR)
            torch.manual_seed(seed)
            LOW_ENERGY_MLOL = self.transform(LOW_ENERGY_MLOL)
            torch.manual_seed(seed)
            RECOMBINED_MLOL=self.transform(RECOMBINED_MLOL)
            torch.manual_seed(seed)
            LOW_ENERGY_MLOR = self.transform(LOW_ENERGY_MLOR)
            torch.manual_seed(seed)
            RECOMBINED_MLOR = self.transform(RECOMBINED_MLOR)
            torch.manual_seed(seed)


        sample = {'LOW_ENERGY_CCL': LOW_ENERGY_CCL, 'RECOMBINED_CCL': RECOMBINED_CCL,'LOW_ENERGY_CCR': LOW_ENERGY_CCR,
                  'RECOMBINED_CCR': RECOMBINED_CCR,'LOW_ENERGY_MLOL': LOW_ENERGY_MLOL, 'RECOMBINED_MLOL': RECOMBINED_MLOL,
                  'LOW_ENERGY_MLOR': LOW_ENERGY_MLOR, 'RECOMBINED_MLOR': RECOMBINED_MLOR,
                  'label1': label1,'label2': label2,'label3': label3,'label4': label4}

        return sample
