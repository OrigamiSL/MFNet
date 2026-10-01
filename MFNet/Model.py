# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from MFNet.Modules import *
from MFNet.embed import DataEmbedding
from utils.RevIN import RevIN


class Encoder_process(nn.Module):
    def __init__(self, patch_size, encoder_num, encoders):
        super(Encoder_process, self).__init__()
        self.patch_size = patch_size
        self.encoder_num = encoder_num
        self.encoders = encoders

    def forward(self, x_enc, U_enc):
        Var_U = U_enc.shape[1]
        x_patch_attn = torch.cat([x_enc, U_enc], dim=1)

        encoder_out_list = []
        for i in range(self.encoder_num):
            x_out, x_patch_attn = self.encoders[i](x_patch_attn, Var_U)
            encoder_out_list.append(x_out[:, :-Var_U])
        return encoder_out_list


class Encoder_map(nn.Module):
    def __init__(self, input_len, encoder_layer=3, patch_size=12, d_model=4, S_num=7, dropout=0.05):
        super(Encoder_map, self).__init__()
        self.input_len = input_len
        self.patch_size = patch_size
        self.encoder_num = encoder_layer
        self.d_model = d_model

        self.Embed1 = DataEmbedding(patch_size, d_model)

        self.encoders = ([Encoder_Cross(input_len, self.patch_size * 2 ** i,
                                        d_model * 2 ** i, S_num, dropout, split=True)
                          for i in range(encoder_layer - 1)] +
                         [Encoder_Cross(input_len, self.patch_size * 2 ** (encoder_layer - 1),
                                        d_model * 2 ** (encoder_layer - 1), S_num, dropout, split=False)])
        self.encoders = nn.ModuleList(self.encoders)
        self.encoder_process = Encoder_process(patch_size, self.encoder_num, self.encoders)

    def forward(self, x, U):
        x_enc = self.Embed1(x).transpose(1, 2)
        U_enc = self.Embed1(U).transpose(1, 2)

        encoder_out_list = self.encoder_process(x_enc, U_enc)
        return encoder_out_list


class Model(nn.Module):
    def __init__(self, input_len, pred_len, encoder_layer,
                 patch_size, d_model, S_num, dropout):
        super(Model, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.encoder_num = encoder_layer
        self.d_model = d_model
        self.align = nn.Linear(input_len // 12, input_len)
        self.revin_x = RevIN()
        self.revin_u = RevIN()
        self.Encoder_process = (
            Encoder_map(input_len, encoder_layer, patch_size, d_model, S_num, dropout))
        self.projection0 = nn.Linear(d_model * 2 ** (encoder_layer - 1),
                                     self.patch_size * 2 ** (encoder_layer - 1))
        self.projection1 = nn.Sequential(nn.Linear(input_len,
                                                   max(2 * input_len, 2 * pred_len)),
                                         nn.Linear(max(2 * input_len, 2 * pred_len),
                                                   pred_len),
                                         )

    def forward(self, x, U):
        B, L, V = x.shape
        self.revin_x(x, 'stats')
        self.revin_u(U, 'stats')
        x = self.revin_x(x, 'norm')
        U = self.revin_u(U, 'norm')
        if x.shape[1] != U.shape[1]:
            U = self.align(U.transpose(1, 2)).transpose(1, 2)
        enc_list = self.Encoder_process(x, U)
        x_out = self.projection0(enc_list[-1]).contiguous().view(B, V, -1)
        x_out = self.projection1(x_out).transpose(1, 2)
        x_out = self.revin_x(x_out, 'denorm')
        return x_out
