import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from math import sqrt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use("Qt5Agg")

"""  EMBEDDING """
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


"""  Attention Mechanism """
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


"""   Encoder  """
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

"""   ITransformer Model   """

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 输入序列长度
        self.seq_len = configs.seq_len
        # 输出预测长度
        self.pred_len = configs.pred_len

        self.use_norm = configs.use_norm

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len,
                               configs.d_model,
                               configs.embed,
                               configs.freq,
                               configs.dropout)
        # 初始化模型
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False,
                               factor=configs.factor,
                               attention_dropout=configs.dropout,
                               output_attention=configs.output_attention),
                               configs.d_model,
                               configs.n_heads
                               ),
                        d_model=configs.d_model,
                        d_ff=configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                      ) for layer in range(configs.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.number_features = configs.factor
        self.net = nn.Sequential(
            nn.Linear(self.number_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()

                     )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        stdev = None
        means = None
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Input: [B, L, N]
        # B: Batch_size
        # L: Length
        # N: The number of variate (变量个数）

        B, L, N = x_enc.shape
        # First step: Embedding (转置输入矩阵，对整个序列进行编码）
        # 编码输出的是转至后的编码矩阵：B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Second step: 进行基于特征维度的注意力编码
        # B N E -> B N E
        enc_out, attns = self.encoder(enc_out, x_mark_enc)

        # Third step: 最后的模块, 进行线性编码，随后转置
        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return self.net(dec_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :].squeeze(-1).contiguous()  # [B, L, D]


class dataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


"""   Train iTransformer Model   """
import matplotlib.pyplot as plt

from pathlib import Path
"""  data_process  """
from Backtest_utils.load_index_data import IndexData

Index_data = IndexData(root_path='E:/Second/Indices/2020-2024', resample_freq='30S', need_time_period='morning',
                       X_window=288, Y_window=96
                       , time_adjust=True, padding=True, start_date='2020-01-01', end_date='2020-12-31')
Index_data.load_index_data(index='SPY')
# 5min 如果用半天级别的预测，则X_window=48，Y_window=12
# 30s 如果用半天级别的预测，则X_window=480，Y_window=60
Index_data.X_window = 120
Index_data.Y_window = 20
Index_data.load_data_sample(label='return', overnight_return=False)
print(Index_data.X_train.shape, Index_data.Y_train.shape)

data_set = dataset(x=Index_data.X_train, y=Index_data.Y_train)
dataloader = DataLoader(dataset=data_set, num_workers=0, shuffle=True, batch_size=64)


import gc
gc.collect()



class Model_train:
    def __init__(self, configs, n_epochs):
        super().__init__()
        self.itransformer_model = Model(configs).to(device=configs.device)
        self.train_data = dataloader
        self.lr = configs.lr
        self.optimizer = optim.Adam(lr=self.lr, params=self.itransformer_model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, verbose=True)
        self.n_epochs = n_epochs
        self.device = configs.device

    def do(self, ):
        model_save_path2 = Path('D:\\PyCharmProject\\Strategy-Backtesting\\Capstone_Local\\ITransformer\\Local_ITransformer.pth')
        all_loss_value = []
        for epoch in tqdm(range(self.n_epochs)):
            epoch_loss = 0
            for idx, (x_batch, y_batch) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                predict = self.itransformer_model(x_enc=x_batch.to(self.device), x_mark_enc=None, x_dec=None, x_mark_dec=None)
                loss_value = F.l1_loss(predict, y_batch.to(self.device)) # L1 Loss 绝对误差损失函数MAE L2 Loss 均方误差损失函数MSE
                epoch_loss = epoch_loss + loss_value.item()
                loss_value.backward()
                self.optimizer.step()

            epoch_loss = epoch_loss / len(self.train_data)
            print(f'------ >>> Epochs: {epoch} >>>  ------ 损失值：{epoch_loss:.8f}')
            all_loss_value.append(epoch_loss)
            self.scheduler.step(epoch_loss)

        print('\n ---------- 模型训练完毕---------- \n ')
        torch.save(self.itransformer_model.state_dict(),model_save_path2  #记得改
        ,_use_new_zipfile_serialization=False)
        self.model_save_path = model_save_path2

        # 损失
        plt.figure(figsize=(8, 5))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.plot(range(0, len(all_loss_value)), all_loss_value, linestyle='-', label='Loss')
        plt.legend(loc='upper right')
        figure_path2 = Path('D:\\PyCharmProject\\Strategy-Backtesting\\Capstone_Local\\ITransformer\\Local_ITransformer_loss.png')
        plt.savefig(figure_path2) #记得改
        plt.show()

    def continue_training(self, next_dataloader):
        """
        继续训练已有的本地模型
        """
        if self.model_save_path.exists():
            print(">>> 载入已有模型，继续训练...")
            self.itransformer_model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        else:
            print(">>> 未找到已保存的模型，将从头开始训练...")
        self.train_data = next_dataloader
        self.do()

if __name__ == '__main__':
    # Create an argument parser
    def set_parser():
        parser = argparse.ArgumentParser(description='ITransformer')
        # Define arguments
        parser.add_argument('--seq_len', type=int, default=Index_data.X_window, help='input sequence length')
        parser.add_argument('--pred_len', type=int, default=Index_data.Y_window, help='output sequence length')
        parser.add_argument('--d_model', type=int, default=128, help='model dimension')
        parser.add_argument('--n_layers', type=int, default=4, help='number of layers')
        parser.add_argument('--factor', type=int, default=9, help='number of features')
        parser.add_argument('--n_heads', type=int, default=4, help='number of heads')
        parser.add_argument('--activation', type=str, default='gelu', help='activation function')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--output_attention', action='store_true', help='output attention in encoder')
        parser.add_argument('--use_norm', type=int, default=False, help='use normalization')
        parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
        parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')  # Changed to float
        parser.add_argument('--freq', type=str, default='s', help='time frequency')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use')

        # Parse arguments and create configs object
        args = parser.parse_args([])  # Pass an empty list to avoid parsing from sys.argv
        return args
    args = set_parser()

    train_model = Model_train(configs=args, n_epochs=5)
    train_model.do()