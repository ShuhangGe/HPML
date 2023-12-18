import numpy as np
import torch
import torch.utils.data as data
import os
import cv2
import torchvision.transforms as transforms
from PIL import Image
from macro import *
import h5py
'''data:
        train.txt
        test.txt
        pic:
            00000001(index):
                00000000_f.png
                00000000_t.png
                00000000_r.png'''
class CADGENdataset(data.Dataset):
    def __init__(self,cfg,test):
        self.test = test
        self.data_root= cfg.data_root
        self.cmd_root = cfg.cmd_root
        
        self.train_lis = os.path.join(self.data_root,'train.txt')
        self.test_lis = os.path.join(self.data_root,'test.txt')
        if self.test:
            with open(self.test_lis, 'r') as f:
                lines = f.readlines()
        else:
            with open(self.train_lis, 'r') as f:
                lines = f.readlines()
        self.file_list = []
        for line in lines:
            self.file_list.append(line.strip())
        self.max_total_len  = cfg.max_total_len

    
    def __getitem__(self, index):
        
        data_num = self.file_list[index]
        
        h5_path = os.path.join(self.cmd_root, data_num+'.h5') 
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)
        # print('cad_vec.shape[0]: ',cad_vec.shape[0])
        pad_len = self.max_total_len - cad_vec.shape[0]

        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        command = cad_vec[:, 0]
        paramaters = cad_vec[:, 1:]
        
        command = torch.tensor(command, dtype=torch.long)
        paramaters = torch.tensor(paramaters, dtype=torch.long)
        command = command.clamp(0,5)
        paramaters = paramaters.clamp(-1,255)
        


        data = (command, paramaters, data_num)
        return data
    
    def __len__(self):
        return len(self.file_list)

