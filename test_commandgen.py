import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from dataset import CADGENdataset
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable,Function
import time
import torchvision
from models_mae import MaskedAutoencoderViT
import config
from macro import *
from loss import Command_loss
import torch.nn.functional as F
if __name__ == '__main__':
    '''
    8:
    Data-loading time for each epoch:  3.7831521034240723
    16:
    Data-loading time for each epoch:  2.560725688934326
    '''
    print('start')
    '''img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False'''
    parser = argparse.ArgumentParser(description='class')
    parser.add_argument('--lr', type=float, default=config.LR, help='learning rate')
    parser.add_argument('--epochs',type = int, default = config.EPOCH)
    parser.add_argument('--num_works', type=int, default=config.NUM_WORKS, help='number of cpu')
    parser.add_argument('--epoch',type = int, default = config.EPOCH)
    parser.add_argument('--train_batch', type=int, default=config.TRAIN_BATCH)
    parser.add_argument('--test_batch', type=int, default=config.TEST_BATCH)
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT, help='train and test data list, in txt format')  
    parser.add_argument('--cmd_root', type=str, default=config.H5_ROOT,help='data path of cad commands, in hdf5 format')    
    parser.add_argument('--device', type=str, default=config.DEVICE, help='GPU or CPU')
    parser.add_argument('--save_path', type=str, default=config.SAVE_PATH, help='path to save the model')
    parser.add_argument('--train_data_num', type=int, default=None)
    #commands paramaters
    parser.add_argument('--max_total_len', type=int, default=MAX_TOTAL_LEN, help='maximum cad sequence length 64')
    parser.add_argument('--n_args', type=int, default=N_ARGS, help='number of paramaters of each command 16')
    parser.add_argument('--n_commands', type=int, default=len(ALL_COMMANDS), help='Number of commands categories 6')
    #paramaters of model
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='mask ratio of MAE')
    parser.add_argument('--embed_dim', type=int, default=256, help='embedding dimension of MAE encoder')
    parser.add_argument('--depth', type=int, default=12, help='depth of encoder')
    parser.add_argument('--num_heads', type=int, default=16, help='num_heads of encoder')
    parser.add_argument('--decoder_embed_dim', type=int, default=128)
    parser.add_argument('--decoder_depth', type=int, default=8)
    parser.add_argument('--decoder_num_heads', type=int, default=16)
    parser.add_argument('--mlp_ratio', type=float, default=4.)
    
    #load parmaters
    args = parser.parse_args()
    epochs = args.epochs
    device = args.device
    if device =='gpu' or device=='GPU':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    save_path = args.save_path
    model_dir = os.path.join(save_path,'model')
    log_dir = os.path.join(save_path,'log')
    LR =args.lr
    print('paramaters set')


    test_dataset = CADGENdataset(args, test = True)
    # length = len(train_dataset)
    if args.train_data_num is not None:
        length_test = int(length_train*0.2)
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [length_test, len(test_dataset)-length_test])
    #train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [length, len(train_dataset)-length])
    print('len(train_dataset): ',len(train_dataset))
    print('len(test_dataset): ',len(test_dataset))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.test_batch,
                                               shuffle=True,
                                               num_workers=args.num_works)
    print('data ready')
    model = MaskedAutoencoderViT(args,mask_ratio=args.mask_ratio, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                 decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
                 mlp_ratio=args.mlp_ratio)
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('model:',model)
    loss_function = Command_loss(args)
    
    save_root = '/scratch/sg7484/CADGen/results/bulletpoints/mae/alldata/mask_0.5-en_12_12-de_8_16-1e-4/out'
    model_path = '/scratch/sg7484/CADGen/results/bulletpoints/mae/alldata/mask_0.5-en_12_12-de_8_16-1e-4/model/MAE_CAD_278_0.7673679815871375.path'
    if os.path.exists(save_root) == False:
        os.makedirs(save_root)
    model.load_state_dict(torch.load(model_path))
    
    model = model.to(device)
    print('start train')
    total_length = len(test_loader)
    for epoch in range(epochs):
        epoch_start = time.time()
        for index, data in enumerate(test_loader):
            print(f'train: total length: {total_length}, index: {index}')
            model.eval()
            command, paramaters, data_num = data
            # print('command.shape: ',command.shape)
            # print('paramaters.shape: ',paramaters.shape)
            '''command.shape:  torch.Size([512, 64])
                paramaters.shape:  torch.Size([512, 64, 16])'''
            command, paramaters = command.to(device),paramaters.to(device)
            optimizer.zero_grad()
            pred, mask = model(command, paramaters)
            '''pred.shape:  torch.Size([512, 64, 6])
            mask.shape:  torch.Size([512, 64])'''
            loss = loss_function(pred, command,mask)
            print('loss_test',loss.item())




