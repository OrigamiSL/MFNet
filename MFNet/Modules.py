import torch
import torch.nn as nn
import math


class Attn_VarLevel(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Attn_VarLevel, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.kv_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, P, V, D = queries.shape
        _, _, R, _ = keys.shape
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries)
        keys = self.kv_projection(keys)
        values = self.kv_projection(values)

        scores = torch.einsum("bpvd,bprd->bpvr", queries, keys)  # [B P V R]
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bpvr,bprd->bpvd", attn, values)  # [B P V LD]
        out = self.out_projection(out)

        return out  # [B P V LD]


class Encoder_Cross(nn.Module):
    def __init__(self, input_len, patch_size, d_model, S_num, dropout=0.1, split=False):
        super(Encoder_Cross, self).__init__()
        self.patch_dim = d_model
        self.input_len = input_len
        self.patch_num = input_len // patch_size
        self.S_num = S_num
        pshifts = torch.arange(1, S_num).reshape(-1, 1).expand(S_num - 1, d_model)
        pshifts = pshifts * 2 * math.pi / S_num
        self.pshifts = nn.Parameter(pshifts)
        self.attn = Attn_VarLevel(self.patch_dim, dropout)

        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)
        self.norm3 = nn.LayerNorm(self.patch_dim)
        self.norm4 = nn.LayerNorm(self.patch_dim)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, self.patch_dim)
        self.linear2 = nn.Linear(self.patch_dim, self.patch_dim)
        self.linear3 = nn.Linear(self.patch_dim, self.patch_dim * 4)
        self.linear4 = nn.Linear(self.patch_dim * 4, self.patch_dim)
        self.linear5 = nn.Linear(self.patch_dim * 2, self.patch_dim * 2)
        self.split = split

    def forward(self, x, Var_U):
        B, V, P, D = x.shape
        x = x.permute(0, 2, 1, 3)  # B, P, V, D

        y = x.clone()
        y = self.linear1(y)

        y_fft = torch.fft.fft(y)
        y_fft = self.activation(y_fft.real) + 1j * self.activation(y_fft.imag)
        y_h_fft = y_fft.clone()
        for i in range(self.S_num - 1):
            shift = self.pshifts[i].unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, P, 1, D)
            shift = torch.cumsum(shift, dim=1)
            y_h_fft += y_fft * torch.exp(-1j * shift)
        y_h = torch.fft.ifft(y_h_fft / self.S_num).real
        y_h = self.norm1(y_h)

        y_h = self.dropout(self.linear2(y_h))
        x = self.norm2(x + y_h)

        x = self.norm3(
            x + self.dropout(self.attn(x, x[:, :, -Var_U:], x[:, :, -Var_U:])))
        z = x.clone()

        z = self.activation(self.linear3(z))
        x = x + self.dropout(self.linear4(z))

        x_out = self.norm4(x).permute(0, 2, 1, 3)  # B, V, P, D
        if self.split:
            x_next = x_out.contiguous().view(B, V, P // 2, 2 * D)
            x_next = self.linear5(x_next)
        else:
            x_next = x_out

        return x_out, x_next
