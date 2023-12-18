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
from models_command import MaskedAutoencoderViT
import config
from macro import *
from loss import Command_loss
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

def forward_loss(command, pred, mask):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    # print('command.shape: ',command.shape)
    # print('pred.shape: ',pred.shape)
    # print('mask.shape: ',mask.shape)
    '''command.shape:  torch.Size([20, 64])
        pred.shape:  torch.Size([20, 64, 6])
        mask.shape:  torch.Size([20, 64])'''
    command = F.one_hot(command, num_classes=6)
    # print('command.shape: ',command.shape)
    # if self.norm_pix_loss:
    #     mean = target.mean(dim=-1, keepdim=True)
    #     var = target.var(dim=-1, keepdim=True)
    #     target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - command) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    print('loss: ',loss)
    return loss
def squared_emd_loss_one_hot_labels(y_pred, y_true, mask=None):
    """
    Squared EMD loss that considers the distance between classes as opposed to the cross-entropy
    loss which only considers if a prediction is correct/wrong.

    Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks.
    Le Hou, Chen-Ping Yu, Dimitris Samaras
    https://arxiv.org/abs/1611.05916

    Args:
        y_pred (torch.FloatTensor): Predicted probabilities of shape (batch_size x ... x num_classes)
        y_true (torch.FloatTensor): Ground truth one-hot labels of shape (batch_size x ... x num_classes)
        mask (torch.FloatTensor): Binary mask of shape (batch_size x ...) to ignore elements (e.g. padded values)
                                  from the loss
    
    Returns:
        torch.tensor: Squared EMD loss
    """
    tmp = torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)
    if mask is not None:
        tmp = tmp * mask
    return torch.sum(tmp) / tmp.shape[0]


def squared_emd_loss(logits, labels, num_classes=-1, mask=None):
    """
    Squared EMD loss that considers the distance between classes as opposed to the cross-entropy
    loss which only considers if a prediction is correct/wrong.

    Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks.
    Le Hou, Chen-Ping Yu, Dimitris Samaras
    https://arxiv.org/abs/1611.05916

    Args:
        logits (torch.FloatTensor): Predicted logits of shape (batch_size x ... x num_classes)
        labels (torch.LongTensor): Ground truth class labels of shape (batch_size x ...)
        mask (torch.FloatTensor): Binary mask of shape (batch_size x ...) to ignore elements (e.g. padded values)
                                  from the loss
    
    Returns:
        torch.tensor: Squared EMD loss
    """
    y_pred = torch.softmax(logits, dim=-1)
    y_true = F.one_hot(labels, num_classes=num_classes).float()
    return squared_emd_loss_one_hot_labels(y_pred, y_true, mask=mask)
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
    # length = len(train_dataset)
    if args.train_data_num is not None:
        length_train = args.train_data_num
        length_test = int(length_train*0.2)
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [length_train, len(train_dataset)-length_train])
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [length_test, len(test_dataset)-length_test])
    #train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [length, len(train_dataset)-length])
    print('len(train_dataset): ',len(train_dataset))
    print('len(test_dataset): ',len(test_dataset))
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
    loss_function = Command_loss(args)

    model = model.to(device)
    print('start train')
    total_length = len(train_loader)
    writer = SummaryWriter(log_dir)
    for epoch in range(epochs):
        epoch_start = time.time()
        train_Loss_total = 0
        train_acc_total = 0
        for index, data in enumerate(train_loader):
            print(f'train: total length: {total_length}, index: {index}')
            model.train()
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
            # loss = F.cross_entropy(pred.permute(0,2,1), command,reduction = 'none') * mask
            # loss = loss.sum()/mask.sum()
            # loss = squared_emd_loss(pred, command, args.n_commands, mask)
            # print('loss.shape: ',loss.shape)
            loss, acc= loss_function(command_logits =pred, tgt_commands =command,mask = mask)
            train_Loss_total +=loss.item()
            train_acc_total += acc
            loss.backward()
            optimizer.step()
            # print('loss_train',loss.item())
            # print('loss_train',acc)
        loss_average = train_Loss_total/(index+1)
        acc_average = train_acc_total/(index+1)
        writer.add_scalar('loss_train',loss_average,global_step=epoch)
        writer.add_scalar('acc_train',acc_average,global_step=epoch)
        average_loss = 0
        test_loss = 0.0
        test_acc = 0
        test_total = 0
        best_test = 10000000
        total_length = len(test_loader)
        for index, data in enumerate(test_loader):
            print(f'test: total length: {total_length}, index: {index}')
            model.eval()
            command, paramaters, data_num = data
            command, paramaters = command.to(device),paramaters.to(device)
            pred, mask = model(command, paramaters)
            # loss = F.cross_entropy(pred.permute(0,2,1), command,reduction = 'none') * mask
            # loss = loss.sum()/mask.sum()      
            # loss = squared_emd_loss(pred, command, args.n_commands, mask)  
            loss, acc = loss_function(command_logits =pred, tgt_commands =command, mask = mask)    
            test_loss += loss.item()
            test_acc += acc
            # print('loss',loss)
        average_loss = test_loss/(index+1)
        average_acc = test_acc/(index+1)
        # print('average_loss',average_loss)
        writer.add_scalar('test_loss', average_loss, global_step=epoch)
        writer.add_scalar('test_acc', average_acc, global_step=epoch)
        # print('average_loss: ',average_loss)
        if average_loss<best_test:
            best_test = average_loss
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, f'MAE_CAD_{epoch}_{average_loss}.path')
            torch.save(model.state_dict(), model_path)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f'MAE_CAD_last.path')
    torch.save(model.state_dict(), model_path)




