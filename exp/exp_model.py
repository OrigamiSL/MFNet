from data.data_loader import *
from exp.exp_basic import Exp_Basic
from MFNet.Model import Model

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
import warnings
from loguru import logger

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        # model = Model(
        #     self.args.input_len,
        #     self.args.pred_len[-1],
        #     self.args.encoder_layer,
        #     self.args.patch_size,
        #     self.args.d_model,
        #     self.args.S_num,
        #     self.args.dropout,
        # ).float()
        # input_x = torch.randn(1, self.args.input_len, self.args.enc_in)
        # U_x = torch.randn(1, self.args.input_len, int(self.args.U_num * self.args.enc_in))
        # flops, params = profile(model, inputs=(input_x, U_x))
        # flops, params = clever_format([flops, params], '%.3f')
        # print(f"flops：{flops}, params：{params}")
        #
        # del model
        model = Model(
            self.args.input_len,
            self.args.pred_len[-1],
            self.args.encoder_layer,
            self.args.patch_size,
            self.args.d_model,
            self.args.S_num,
            self.args.dropout,
        ).float()
        return model

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'PEMS03': Dataset_PEMS,
            'PEMS04': Dataset_PEMS,
            'PEMS07': Dataset_PEMS,
            'PEMS08': Dataset_PEMS,
            'Traffic': Dataset_Traffic,
            'CA-D5': Dataset_CA_D5
        }
        Data = data_dict[self.args.data]

        size = [args.input_len, args.pred_len[-1]]
        if flag == 'train':
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = False
            drop_last = True
            batch_size = 1

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=size,
            U_num=args.U_num,
            aug_p=args.aug_p,
            aug_e=args.aug_e
        )
        logger.info(flag)
        logger.info(len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data=None, vali_loader=None):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, U) in enumerate(vali_loader):
                pred, true = self._process_one_batch(batch_x, U)
                pred = pred.squeeze(0).detach().cpu().numpy()
                true = true.squeeze(0).detach().cpu().numpy()
                loss = np.mean(np.abs(pred - true))
                total_loss.append(loss)
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting=None):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        train_steps = len(train_loader)

        lr = self.args.learning_rate

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self.model.train()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, U) in enumerate(train_loader):
                if batch_x.shape[-1] > 100:
                    index_list = np.arange(batch_x.shape[-1])
                    index_list = np.random.choice(index_list,
                                                  int(5 * np.log2(batch_x.shape[-1])), replace=False)
                    c_batch_x = batch_x[:, :, index_list]
                else:
                    c_batch_x = batch_x
                model_optim.zero_grad()
                iter_count += 1
                pred, true = self._process_one_batch(c_batch_x, U)
                # # plot
                # plot_input = batch_x[0, :self.args.input_len, :]
                # plot_input = plot_input.detach().cpu().numpy()
                # for j in range(plot_input.shape[-1]):
                #     plt.figure(figsize=(24, 16))
                #     plt.plot(plot_input[:, j], 'k')
                #     plt.tight_layout()
                #     plt.show()

                loss = torch.mean((pred - true) ** 2) + torch.mean(abs(pred - true))
                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                                  torch.mean(loss).item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader)

            # logger.info("Pred_len: {0}| Epoch: {1}, Steps: {2} | Total: Vali Loss: {3:.7f} Test Loss: {4:.7f}| "
            #             .format(self.args.pred_len, epoch + 1, train_steps, vali_loss, test_loss))
            logger.info("Pred_len: {0}| Epoch: {1}, Steps: {2} | Total: Vali Loss: {3:.7f} "
                        .format(self.args.pred_len[-1], epoch + 1, train_steps, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
            adjust_learning_rate(model_optim, (epoch + 1), self.args)

        self.args.learning_rate = lr

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, load=True):
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        test_data, test_loader = self._get_data(flag='test')
        time_now = time.time()

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, (batch_x, U) in enumerate(test_loader):
                pred, true = self._process_one_batch(batch_x, U)

                # # plot
                # plot_pred = torch.cat([batch_x[:, :self.args.input_len, :].to(pred.device), pred], dim=1).squeeze()
                # plot_true = batch_x.squeeze()
                # plot_pred = plot_pred.detach().cpu().numpy()
                # plot_true = plot_true.detach().cpu().numpy()
                # for j in range(plot_pred.shape[-1]):
                #     plt.figure(figsize=(24, 16))
                #     plt.plot(plot_pred[:, j], 'r')
                #     plt.plot(plot_true[:, j], 'k')
                #     plt.tight_layout()
                #     plt.show()

                pred = pred.squeeze(0).detach().cpu().numpy()
                true = true.squeeze(0).detach().cpu().numpy()

                np.save(folder_path + 'pred_{}.npy'.format(i), pred)
                np.save(folder_path + 'true_{}.npy'.format(i), true)

        logger.info("inference time: {}".format(time.time() - time_now))

    def _process_one_batch(self, batch_x, U):
        batch_x = batch_x.float().to(self.device)
        input_seq = batch_x[:, :self.args.input_len, :]
        input_U = U.float().to(self.device)

        batch_y = batch_x[:, -self.args.pred_len[-1]:, :]
        pred_data = self.model(input_seq, input_U)
        return pred_data, batch_y
