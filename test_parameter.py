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
import h5py

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
    
def logits2vec( outputs, refill_pad=True, to_numpy=True):
    """network outputs (logits) to final CAD vector"""
    out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
    out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
    if refill_pad: # fill all unused element to -1
        mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
        # print('out_args.shape: ',out_args.shape)
        # print('mask.shape: ',mask.shape)
        out_args[mask] = -1

    out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
    if to_numpy:
        out_cad_vec = out_cad_vec.detach().cpu().numpy()
    return out_cad_vec

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
    parser.add_argument('--save_root', type=str, default='./save_root', help='path to save the output')
    parser.add_argument('--model_path', type=str, default='./model_path', help='trained model ')
    parser.add_argument('--out_num', type=int, default=1000, help='trained model ')
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
    out_num = args.out_num
    epochs = args.epochs
    device = args.device
    save_root = args.save_root
    model_path = args.model_path
    print(torch.cuda.is_available())
    if device =='gpu' or device=='GPU':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    

    test_dataset = CADGENdataset(args, test = True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.test_batch,
                                               shuffle=True,
                                               num_workers=args.num_works)
    print('data ready')
    model = MaskedAutoencoderViT(args,mask_ratio=args.mask_ratio, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                 decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
                 mlp_ratio=args.mlp_ratio)
    # for arg in vars(args):
    #     print(arg, ':', getattr(args, arg))
    # print('model:',model)
    

    if os.path.exists(save_root) == False:
        os.makedirs(save_root)
    model.load_state_dict(torch.load(model_path))
    loss_fun = CADLoss(args).to(device)
    model = model.to(device)
    print('start train')
    total_length = len(test_loader)
    out_count = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        with torch.no_grad():
            for index, data in enumerate(test_loader):
                print(f'train: total length: {total_length}, index: {index}')
                model.eval()
                command, paramaters, data_num = data
                # print('command.shape: ',command.shape)
                # print('paramaters.shape: ',paramaters.shape)
                '''command.shape:  torch.Size([512, 64])
                    paramaters.shape:  torch.Size([512, 64, 16])'''
                command, paramaters = command.to(device),paramaters.to(device)
                output, mask = model(command, paramaters)
                '''pred.shape:  torch.Size([512, 64, 6])
                mask.shape:  torch.Size([512, 64])'''
                output["tgt_commands"] = command
                output["tgt_args"] = paramaters
                # command = command.unsqueeze(-1)
                # print('command.shape: ',command.shape)
                command_out = F.one_hot(command, num_classes=6)
                # print('command_out.shape: ',command_out.shape) 
                output["command_logits"] = command_out.type(torch.float32)
                out_cad_vec = logits2vec(output)
                gt_vec = torch.cat([command.unsqueeze(-1), paramaters], dim=-1).squeeze(1).detach().cpu().numpy()
                batch_size = command.shape[0]
                # loss = loss_fun(output)
                #print('loss_train',loss.item())
            for j in range(batch_size):
                out_vec = out_cad_vec[j]
                seq_len = command[j].tolist().index(EOS_IDX)
                data_id = epoch*batch_size+j
                save_path = os.path.join(save_root, '{}_vec.h5'.format(data_id))
                print('save_path: ',save_path)
                with h5py.File(save_path, 'w') as fp:
                    fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=np.int)
                    fp.create_dataset('gt_vec', data=gt_vec[j][:seq_len], dtype=np.int)
                #print('out_vec.shape: ',out_vec.shape)
                #print('gt_vec.shape: ',gt_vec.shape)
                np.savetxt(os.path.join(save_root,f'{data_id}_out_vec.txt'), out_vec[:seq_len])
                np.savetxt(os.path.join(save_root,f'{data_id}_gt_vec.txt'), gt_vec[j][:seq_len])
                out_count += 1
            if out_count >= out_num:
                break



'''
singularity exec --nv \
                    --overlay /scratch/sg7484/pytorch-example/CMDGen-15GB-500K.ext3:ro \
                            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
                             /bin/bash -c "source /ext3/env.sh; python test_paramater.py \
                             --epochs 9000 --lr 1e-4 --mask_ratio 0.5\
                             --train_batch 256 --test_batch 128 --max_total_len 64 --depth 12 --decoder_depth 8\
                             --num_heads 16 --decoder_num_heads 16\
                             --save_path /scratch/sg7484/CADGen/results/bulletpoints/mae/alldata/mask_0.5-en_12_12-de_8_16-1e-4 \
                             --data_root /scratch/sg7484/data/CMDGen/all_data \
                             --cmd_root /scratch/sg7484/data/CMDGen/all_data/cad_vec "

'''
