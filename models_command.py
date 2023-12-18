# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_1d_sincos_pos_embed_from_grid
import torch.nn.functional as F
import logging
from loss import CADLoss
from model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask
from layers.transformer import *
from layers.improved_transformer import *
from layers.positional_encoding import *

class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, args, seq_len, use_group=False):
        super().__init__()
        self.command_embed = nn.Embedding(args.n_commands, args.embed_dim)
        args_dim = args.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * args.n_args, args.embed_dim)

    def forward(self, commands, args, groups=None):
        S, N = commands.shape
        command_embed = self.command_embed(commands.long())
        src =  command_embed+ \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL
        # src = self.pos_encoding(src)
        return src,command_embed
    
class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim+ 1
        self.d_model = d_model
        # self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * self.args_dim)
        

    def forward(self, out):
        S, N, _ = out.shape
        # print('out.shape: ',out.shape)
        # print('self.d_model: ',self.d_model)
        # command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return args_logits

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, args, mask_ratio,
                 embed_dim=256, depth=24, num_heads=16,
                 decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        self.mask_ratio = mask_ratio
        self.max_len = args.max_total_len
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.max_len, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, args.n_commands, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        # self.text_encoder = Text_Encoder(args)
        self.args = args.n_commands
        self.embed_dim =embed_dim
        self.text_encoder = nn.Linear(args.n_args+1, embed_dim)
        # self.command_encoder = nn.Linear(1, embed_dim)
        # self.paramater_encoder = nn.Linear(args.n_args, embed_dim)
        self.embedding = CADEmbedding(args, args.max_total_len)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], int(self.max_len))#########################################
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_embed.shape[-1], int(self.max_len))######################################
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, command, paramaters, mask_ratio):
        # # embed patches
        # x = self.patch_embed(x)
        padding_mask, key_padding_mask = _get_padding_mask(command, seq_dim=-1), _get_key_padding_mask(command, seq_dim=-1)
        print('command.shape: ',command.shape)
        x, command_embed= self.embedding(command, paramaters)
        # # add pos embed w/o cls token
        x = x + self.pos_embed[:, :, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        #print('x.shape: ',x.shape)
        '''x.shape:  torch.Size([20(batchsize), 16(patch after mask), 256])'''

        # # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        #print('\nx0.shape: ',x.shape)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)#+1 include cls token 
        #print('mask_tokens.shape: ',mask_tokens.shape)
        '''mask_tokens.shape:  torch.Size([256, 32, 128])'''
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        #print('x_0.shape: ',x_.shape)
        '''x_0.shape:  torch.Size([256, 64, 128])'''
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        #print('x_1.shape: ',x_.shape)
        '''x_1.shape:  torch.Size([256, 64, 128])'''
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        #print('self.decoder_pos_embed.shape: ',self.decoder_pos_embed.shape)
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        #print('x1.shape: ',x.shape)
        # predictor projection
        x =F.sigmoid(self.decoder_pred(x)) 
        #print('x2.shape: ',x.shape)
        # # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward_loss(self, command, pred, mask):
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
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - command) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        #print('loss: ',loss)
        return loss

    def forward(self, command, paramaters, mask_ratio=0.75):
        mask_ratio = self.mask_ratio
        # command and paramaters embedding
        # command = command.unsqueeze(-1)
        command = command.type(torch.float32)
        paramaters = paramaters.type(torch.float32)
        # commmand_embed = self.command_encoder(command)
        # paramater_embed = self.paramater_encoder(paramaters)
        # text = commmand_embed + paramater_embed
        #print('text.shape: ',text.shape)
        '''text.shape:  torch.Size([20, 64, 256])'''
        latent, mask, ids_restore = self.forward_encoder(command, paramaters, mask_ratio)
        #print('latent.shape: ',latent.shape)
        #print('mask.shape: ',mask.shape)
        #print('ids_restore.shape: ',ids_restore.shape)
        '''latent.shape:  torch.Size([20, 16, 256])
            mask.shape:  torch.Size([20, 64])
            ids_restore.shape:  torch.Size([20, 64])'''
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        #print('pred.shape: ',pred.shape)
        '''pred.shape:  torch.Size([20, 64, 6])'''
        # loss = self.forward_loss(command, pred, mask)
        # logging.info(f'loss: {loss.item()}')
        #print('pred: ',pred)
        #print('loss: ',loss)
        return pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
