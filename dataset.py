import torch.utils.data as data
import torch
import h5py
import scipy.misc
import random
import numpy as np
import os
from PIL import Image


class DatasetFromYouKu(data.Dataset):
    def __init__(self):
        super(DatasetFromYouKu, self).__init__()
        file_l = open("train_l.txt", 'r')
        file_h = open("train_h.txt", 'r')
        self.img_l_path = file_l.readlines()
        self.img_h_path = file_h.readlines()
        self.batch_index = 0
        file_l.close()
        file_h.close()

    def __getitem__(self, index):
        return self.load_image(self.img_l_path[index]), self.load_image(self.img_h_path[index])

    def __len__(self):
        return len(self.img_l_path)

    def load_image(self, path):
        path = path[:-1]
        img = scipy.misc.imread(path)
        img = np.asarray(img)
        img = np.transpose(img, (2, 0, 1))
        return img


    def crop_center(self, img,cropx,cropy):
	    y,x,_ = img.shape
	    startx = random.sample(range(x-cropx-1),1)[0]#x//2-(cropx//2)
	    starty = random.sample(range(y-cropy-1),1)[0]#y//2-(cropy//2)
	    return img[starty:starty+cropy,startx:startx+cropx]

    def load_imgs(self, index, path):
        imgs = []
        for i in index:
            imgs.append(self.load_image(path[i]))
        return imgs

    def get_batch(self, batch_size):
        max_counter = int(len(self.img_l_path)/batch_size)
        counter = int(batch_size % max_counter)
        window = [x for x in range(counter*batch_size, (counter+1)*batch_size)]
        x = self.load_imgs(window, self.img_l_path)
        y = self.load_imgs(window, self.img_h_path)
        batch_index = (self.batch_index+1) % max_counter
        x = np.asarray(x)
        y = np.asarray(y)
        x = np.transpose(x, (0, 3, 1, 2))
        y = np.transpose(y, (0, 3, 1, 2))
        return x, y
