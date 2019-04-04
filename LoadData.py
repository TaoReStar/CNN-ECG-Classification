import json
import keras
import numpy as np
import os
import random
import scipy.io as sio

#### Load dataset functions
STEP=256
labels=[]
ecgs=[]
def load_ecg_mat(ecg_file):
     if os.path.splitext(ecg_file)[1] == ".npy":
        ecg = np.load(ecg_file)
     elif os.path.splitext(ecg_file)[1] == ".mat":
        ecg = sio.loadmat(ecg_file)['val'].squeeze()
     else: # Assumes binary 16 bit integers
        with open(ecg_file, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)
     return ecg
def load_all(data_path):
    label_file = data_path+ "/Refrence.csv"
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]
    dataset = []
    for record, label in records:
        ecg_file = os.path.join(data_path, record + ".npy")
        ecg_file = os.path.abspath(ecg_file)
        ecg = load_ecg_mat(ecg_file)
        num_labels = ecg.shape[0] / STEP
        dataset.append((ecg_file, [label]*int(num_labels)))
    return dataset

def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')

def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in data:
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

data_path=r"/mnt/data0/tao/ECG/"
dataset=load_all(data_path)
random.seed(2018)
dev_frac=.1
train, dev = split(dataset, dev_frac)
make_json("train.json", train)
make_json("dev.json", dev)