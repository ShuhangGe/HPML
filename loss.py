import torch
import torch.nn as nn
import torch.nn.functional as F
from macro import CMD_ARGS_MASK
from macro import EOS_IDX, SOL_IDX, EXT_IDX
from model_utils import _get_padding_mask, _get_visibility_mask
import numpy as np
import logging


logging.basicConfig(level=logging.INFO,  
                    filename='main_mae_cross.log',
                    filemode='a', 
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    )
class loss_fun():
    def __init__(self):
        self.loss = nn.CrossEntropyLoss(reduction="none")
        
    def L2loss(self, command, pred, mask):
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

        loss = (pred - command) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        print('loss: ',loss)
        return loss
    def command_loss(self, command, pred, mask):
        # command = F.one_hot(command, num_classes=6)
        # print('command.shape: ',command.shape)
        # print('pred.shape: ',pred.shape)
        # print('mask.shape: ',mask.shape)
        output = torch.mul(pred, mask.unsqueeze(-1).repeat(1,1,6))
        output = output.type(torch.float32)
        target = torch.mul(command, mask)
        target = target.type(torch.long)
        # print('output.shape: ',output.shape)
        # print('target.shape: ',target.shape)
        # print('type(output): ',type(output))
        # print('type(target): ',type(target))
        loss = self.loss(output, target)
        #print('loss.shape: ',loss.shape)
        a=b
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
    # print('logits.shape: ',logits.shape)
    # print('labels.shape: ',labels.shape)
    # print('mask.shape: ',mask.shape)
    '''logits.shape:  torch.Size([6, 6])
        labels.shape:  torch.Size([6])
        mask.shape:  torch.Size([6])'''
    y_pred = torch.softmax(logits, dim=-1)
    y_true = F.one_hot(labels, num_classes=num_classes).float()

    y_pred_temp = torch.argmax(y_pred, dim=-1)
    y_true_temp = torch.argmax(y_true, dim=-1)
    #print('y_pred_temp: ',y_pred_temp)
    #print('y_true_temp: ',y_true_temp)
    correct = y_pred_temp.eq(y_true_temp).sum().item()
    num = y_pred_temp.shape[0]
    acc = correct/num
    # logging.info(f'y_pred: {y_pred}')
    # logging.info(f'y_true: {y_true}\n')
    return squared_emd_loss_one_hot_labels(y_pred, y_true, mask=mask),acc

    
class CADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, output):
        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]
        # print('tgt_commands.shape: ',tgt_commands.shape)
        # print('tgt_args.shape: ',tgt_args.shape)
        '''tgt_commands.shape:  torch.Size([256, 64])
            tgt_args.shape:  torch.Size([256, 64, 16])'''
        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)
        # print('visibility_mask.shape: ',visibility_mask.shape)
        # print('padding_mask.shape: ',padding_mask.shape)
        '''visibility_mask.shape:  torch.Size([256])
            padding_mask.shape:  torch.Size([256, 64])'''
        # command_logits, args_logits = output["command_logits"], output["args_logits"]
        args_logits = output["args_logits"]
        mask = self.cmd_args_mask[tgt_commands.long()]
        if torch.isnan(args_logits).any():
            print('\n','nanananan','\n')

        # loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()].reshape(-1, self.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long().clamp(0, 6),ignore_index=-1)
        #loss_cmd = F.cross_entropy(command_logits.reshape(-1, self.n_commands), tgt_commands.reshape(-1).long().clamp(0, 6),ignore_index=-1)

        np.set_printoptions(threshold=np.inf)
        loss_args = F.cross_entropy(args_logits[mask.bool()].reshape(-1, self.args_dim), (tgt_args[mask.bool()].reshape(-1).long() + 1).clamp(0, 256),ignore_index=-1)  # shift due to -1 PAD_VAL

        # loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        # loss_args = self.weights["loss_args_weight"] * loss_args
        # print('loss_cmd: ',loss_cmd, 'loss_args: ',loss_args)
        '''
        mask_token.shape:  torch.Size([50, 19, 128])
        x_full.shape:  torch.Size([50, 32, 128])
        pos_full.shape:  torch.Size([50, 32, 128])
        out_logits[0].shape:  torch.Size([50, 60, 6])
        out_logits[1].shape:  torch.Size([50, 60, 16, 257])'''
        # res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return loss_args
    
class CADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        # self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, output):
        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]
        # print('tgt_commands.shape: ',tgt_commands.shape)
        # print('tgt_args.shape: ',tgt_args.shape)
        '''tgt_commands.shape:  torch.Size([256, 64])
            tgt_args.shape:  torch.Size([256, 64, 16])'''
        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)
        # print('visibility_mask.shape: ',visibility_mask.shape)
        # print('padding_mask.shape: ',padding_mask.shape)
        '''visibility_mask.shape:  torch.Size([256])
            padding_mask.shape:  torch.Size([256, 64])'''
        # command_logits, args_logits = output["command_logits"], output["args_logits"]
        args_logits = output["args_logits"]
        mask = self.cmd_args_mask[tgt_commands.long()]
        if torch.isnan(args_logits).any():
            print('\n','nanananan','\n')

        # loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()].reshape(-1, self.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long().clamp(0, 6),ignore_index=-1)
        #loss_cmd = F.cross_entropy(command_logits.reshape(-1, self.n_commands), tgt_commands.reshape(-1).long().clamp(0, 6),ignore_index=-1)

        np.set_printoptions(threshold=np.inf)
        loss_args = F.cross_entropy(args_logits[mask.bool()].reshape(-1, self.args_dim), (tgt_args[mask.bool()].reshape(-1).long() + 1).clamp(0, 256),ignore_index=-1)  # shift due to -1 PAD_VAL

        # loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        # loss_args = self.weights["loss_args_weight"] * loss_args
        # print('loss_cmd: ',loss_cmd, 'loss_args: ',loss_args)
        '''
        mask_token.shape:  torch.Size([50, 19, 128])
        x_full.shape:  torch.Size([50, 32, 128])
        pos_full.shape:  torch.Size([50, 32, 128])
        out_logits[0].shape:  torch.Size([50, 60, 6])
        out_logits[1].shape:  torch.Size([50, 60, 16, 257])'''
        # res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return loss_args
    
# class Command_loss(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()

#         self.n_commands = cfg.n_commands
#         # self.args_dim = cfg.args_dim + 1
#         # self.weights = cfg.loss_weights

#         # self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

#     def forward(self, command_logits, tgt_commands, mask):
#         # Target & predictions
#         np.set_printoptions(threshold=np.inf)
#         torch.set_printoptions(profile="full")
#         '''command_logits.shape:  torch.Size([256, 64, 6])'''
#         '''tgt_commands.shape:  torch.Size([256, 64])'''
#         '''mask.shape:  torch.Size([256, 64])'''
#         visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
#         padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)
#         '''visibility_mask.shape:  torch.Size([256])'''
#         '''padding_mask.shape:  torch.Size([256, 64])'''
#         # loss = squared_emd_loss(command_logits[padding_mask.bool()].reshape(-1, self.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long().clamp(0, 6), num_classes=6, mask=None)
#         padding_mask  = padding_mask.bool()
#         print('padding_mask.shape: ',padding_mask.shape)
#         print('padding_mask: ',padding_mask)
#         loss_total = 0 
#         correct_total = 0
#         num_total = 0
#         for index, mask_temp in enumerate(mask):
#             mask_temp = mask_temp.unsqueeze(0)
#             padding_mask_temp = padding_mask[index,:].unsqueeze(0)
#             tgt_commands_temp = tgt_commands[index,:].unsqueeze(0)
#             command_logits_temp = command_logits[index,:,:].unsqueeze(0)
#             # print('mask_temp.shape: ',mask_temp.shape)
#             # print('padding_mask_temp.shape: ',padding_mask_temp.shape)
#             # print('tgt_commands_temp.shape: ',tgt_commands_temp.shape)
#             # print('command_logits_temp.shape: ',command_logits_temp.shape)
            
#             loss_temp,correct ,num= squared_emd_loss(command_logits_temp[padding_mask_temp], tgt_commands_temp[padding_mask_temp].long().clamp(0, 6), num_classes=6, mask=mask_temp[padding_mask_temp])
#             loss_total = loss_total + loss_temp
#             num_total += num
#             correct_total += correct
#         loss_total = loss_total/index
#         acc = correct_total/num_total
#         return loss_total, acc
class Command_loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1

    def forward(self, command_logits, tgt_commands, mask):
        # Target & predictions
        #print('tgt_commands.shape: ',tgt_commands.shape)
        #print('tgt_args.shape: ',tgt_args.shape)
        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)
        print('command_logits.shape: ',command_logits.shape)
        print('padding_mask.shape: ',padding_mask.shape)

        print('shape1:  ',(command_logits[padding_mask.bool()].reshape(-1, self.n_commands)).shape)
        print('shape2:  ',(tgt_commands[padding_mask.bool()].reshape(-1).long()).shape)
        loss_cmd,correct = squared_emd_loss(command_logits[padding_mask.bool()].reshape(-1, self.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long())
        # loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()].reshape(-1, self.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long())
        print('correct: ',correct)
        return loss_cmd,correct