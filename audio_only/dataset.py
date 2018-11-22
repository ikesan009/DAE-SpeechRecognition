# encoding: utf-8
import time
import glob
import random
import scipy.io
import numpy as np


def load_file(filename, kind):
    arrays = scipy.io.loadmat(filename)
    arrays = arrays[kind]
    arrays = arrays.flatten()
    arrays = arrays.astype(float)
    return arrays


class MyDataset():
    def __init__(self, folds, kind, path):
        self.folds = folds
        self.path = path
        self.kind = kind
        self.clean = 1 / 7.
        with open('../label_sorted.txt') as myfile:
            lines = myfile.read().splitlines()
        self.data_dir = [self.path + item for item in lines]
        self.data_files = glob.glob(self.path+'*/'+self.folds+'/*.mat')
        self.list = {}
        for i, x in enumerate(self.data_files):
            for j, elem in enumerate(self.data_dir):
                if elem in x:
                    self.list[i] = [x]
                    self.list[i].append(j)
        print('Load {} part'.format(self.folds))

    def normalisation(self, inputs):
        inputs_std = np.std(inputs)
        if inputs_std == 0.:
            inputs_std = 1.
        return (inputs - np.mean(inputs))/inputs_std

    def __getitem__(self, idx):
        noise_prop = (1-self.clean)/6.
        temp = random.random()
        kind = 'audio'
        if self.kind:
            kind = self.kind
        elif self.folds == 'train':
            if temp < noise_prop:
                kind = 'm5db'
            elif temp < 2 * noise_prop:
                kind = 'p0db'
            elif temp < 3 * noise_prop:
                kind = 'p5db'
            elif temp < 4 * noise_prop:
                kind = 'p10db'
            elif temp < 5 * noise_prop:
                kind = 'p15db'
            elif temp < 6 * noise_prop:
                kind = 'p20db'
        inputs = load_file(self.list[idx][0], kind)
        labels = self.list[idx][1]
        inputs = self.normalisation(inputs)
        return inputs, labels

    def __len__(self):
        return len(self.data_files)