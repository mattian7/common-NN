from model import *
import os
import struct
import numpy as np
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            self.images_path = os.path.join(root, 'train-images-idx3-ubyte')
            self.labels_path = os.path.join(root, 'train-labels-idx1-ubyte')
        else:
            self.images_path = os.path.join(root, 't10k-images-idx3-ubyte')
            self.labels_path = os.path.join(root, 't10k-labels-idx1-ubyte')

        with open(self.images_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            self.images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)

        with open(self.labels_path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            self.labels = np.fromfile(f, dtype=np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def write_steup(args,wp):
    wp.write("*******************************\n")
    wp.write("model:"+args.model+"\n")
    wp.write("learning_rate:"+str(args.lr)+"\n")
    wp.write("dropout:"+str(args.dropout)+"\n")
    wp.write("batch_size:"+str(args.batchsize)+"\n")
    wp.write("*******************************\n")



