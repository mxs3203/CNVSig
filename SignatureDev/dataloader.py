import glob
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

from feature_util import readPickle


class CNVImages(Dataset):
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

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotation.iloc[idx, 0])
        image = readPickle(img_path).reshape((9, 22, 22))
        return image,img_path.split("/")[-1].split(".")[0]