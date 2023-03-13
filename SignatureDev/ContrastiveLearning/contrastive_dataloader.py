import glob
import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from feature_util import readPickle


class CNVImagesContrastive(Dataset):
    def __init__(self, img_dir ):
        self.img_dir = img_dir
        self.valid_IDs = []
        for file in glob.glob(img_dir + "*.pickle"):
            df = readPickle(file)
            if np.shape(df) == (22, 22, 9):
                # split path by / and take the last which is ID.pickle and then split by . to get ID only
                self.valid_IDs.append(file.split("/")[-1].split(".")[0] + ".pickle")
        self.annotation = pd.DataFrame(self.valid_IDs)
        self.img_path = os.path.join(self.img_dir, self.annotation.iloc[0, 0])
        self.image = readPickle(self.img_path)
        #self.image = self.image.reshape((22, 9, 9))
        print("Example Image: ", np.shape(self.image))
        #print(self.image)

    def __len__(self):
        return len(self.annotation)

    def applyRandomGaussianNoise(self, image):
        noise = np.random.normal(0, .5, image.shape)
        image = image + noise
        return image

    def getRandomRowsColumns(self):
        random_chr = random.randint(0, 21)
        if random_chr == 0:
            next_chr = 1
        elif random_chr == 22:
            next_chr = 21
        else:
            next_chr = random_chr - 1
        return random_chr,next_chr

    def switchChromosomesNextToEachOther(self, image):
        random_chr, next_chr = self.getRandomRowsColumns()
        image[[random_chr, next_chr]] = image[[next_chr, random_chr]]
        return image

    def switchBinsNextToEachOther(self, image):
        random_chr, next_chr = self.getRandomRowsColumns()
        image[:, [next_chr, random_chr]] = image[:, [random_chr, next_chr]]
        return image

    def augmentImageWithProb(self, image, p=0.5):
        for i in range(9): # for all features
            if random.random() >= p:
                image[i, :, :] = self.applyRandomGaussianNoise(image[i, :, :])
            if random.random() >= p:
                image[i, :, :] = self.switchChromosomesNextToEachOther(image[i, :, :])
            if random.random() >= p:
                image[i, :, :] = self.switchBinsNextToEachOther(image[i, :, :])
        return image

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotation.iloc[idx, 0])
        image = readPickle(img_path).reshape((9, 22, 22))
        image1 = self.augmentImageWithProb(image)
        image2 = self.augmentImageWithProb(image)
        return image1,image2, img_path.split("/")[-1].split(".")[0]