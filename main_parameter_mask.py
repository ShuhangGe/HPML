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
from gsh.CADGen.bulletpoints.mae_cad.models_parameter_mask import MaskedAutoencoderViT
import config
from macro import *
from loss import CADLoss
import torch.nn.functional as F
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]

        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
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
    #commands paramaters
    parser.add_argument('--max_total_len', type=int, default=MAX_TOTAL_LEN, help='maximum cad sequence length 64')
    parser.add_argument('--n_args', type=int, default=N_ARGS, help='number of paramaters of each command 16')
    parser.add_argument('--n_commands', type=int, default=len(ALL_COMMANDS), help='Number of commands categories 6')
    #paramaters of model embdeeing
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='mask ratio of MAE')
    parser.add_argument('--embed_dim', type=int, default=256, help='embedding dimension of MAE encoder')
    parser.add_argument('--dim_feedforward', type=int, default=16, help='FF dimensionality: forward dimension in transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate used in basic layers and Transformers')
    parser.add_argument('--depth', type=int, default=12, help='depth of encoder')
    parser.add_argument('--num_heads', type=int, default=16, help='num_heads of encoder')
    #deocder
    parser.add_argument('--decoder_embed_dim', type=int, default=128)
    parser.add_argument('--decoder_depth', type=int, default=8)
    parser.add_argument('--decoder_num_heads', type=int, default=16)
    parser.add_argument('--mlp_ratio', type=float, default=4.)
    parser.add_argument('--args_dim', type=int, default=256)
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


    train_dataset = CADGENdataset(args, test = False)
    test_dataset = CADGENdataset(args, test = True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_batch,
                                               shuffle=True,
                                               num_workers=args.num_works)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))
    loss_fun = CADLoss(args).to(device)
    model = model.to(device)
    
    print('start train')
    total_length = len(train_loader)
    writer = SummaryWriter(log_dir)
    best_test = 10000000
    for epoch in range(epochs):
        epoch_start = time.time()
        for index, data in enumerate(train_loader):
            print(f'train: total length: {total_length}, index: {index}')
            # model.train()
            command, paramaters, data_num = data
            # print('command.shape: ',command.shape)
            # print('paramaters.shape: ',paramaters.shape)
            # print('command: ',command)
            bool_matrix = (command == 5)
            index = torch.nonzero(bool_matrix)
            # print('index: ',index)
            # print('bool_matrix: ',bool_matrix)
            
            '''command.shape:  torch.Size([512, 64])
                paramaters.shape:  torch.Size([512, 64, 16])'''
            command, paramaters = command.to(device),paramaters.to(device)
            optimizer.zero_grad()
            output, mask = model(command, paramaters)
            '''pred.shape:  torch.Size([512, 64, 6])
            mask.shape:  torch.Size([512, 64])'''
            output["tgt_commands"] = command
            output["tgt_args"] = paramaters
            output["command_logits"] = command.type(torch.float32)
            loss = loss_fun(output)
            print('loss: ',loss)
            # loss = loss_dict.values()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss_train',loss.item(),global_step=epoch)
            print('loss_train',loss.item())

        average_loss = 0
        test_loss = 0.0
        test_total = 0
        total_length = len(test_loader)
        with torch.no_grad():
            for index, data in enumerate(test_loader):
                print(f'test: total length: {total_length}, index: {index}')
                model.eval()
                command, paramaters, data_num = data
                command, paramaters = command.to(device),paramaters.to(device)
                output, mask = model(command, paramaters)
                '''pred.shape:  torch.Size([512, 64, 6])
                mask.shape:  torch.Size([512, 64])'''
                output["tgt_commands"] = command
                output["tgt_args"] = paramaters
                output["command_logits"] = command.type(torch.float32)
                loss = loss_fun(output)
                print('loss: ',loss)        
                test_loss += loss.item()
                # print('loss',loss)
        average_loss = test_loss/(index+1)
        # print('average_loss',average_loss)
        writer.add_scalar('average_loss', average_loss, global_step=epoch)
        print('average_loss: ',average_loss)
        if average_loss<best_test:
            best_test = average_loss
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, f'MAE_CAD_{epoch}_{average_loss}.path')
            torch.save(model.state_dict(), model_path)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f'MAE_CAD_last_{epoch}.path')
    torch.save(model.state_dict(), model_path)




