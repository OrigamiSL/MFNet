import os
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from loguru import logger
from tqdm import tqdm
from utils.tools import StiefelT

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class Dataset_Traffic(Dataset):
    def __init__(self, root_path, data_path='Traffic.csv',
                 flag='train', size=None, U_num=0.1, aug_p=0.5, aug_e=0.2):
        # size [label_len, pred_len]
        # info
        self.input_len = size[0]
        self.pred_len = size[1]
        self.U_num = U_num
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.aug_p = aug_p
        self.aug_e = aug_e
        self.__read_data__()
        self.__preprocess__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns)
        cols.remove('date')
        df_data = df_raw[cols]
        df_data.fillna(method='ffill', limit=len(df_data), inplace=True)
        df_data.fillna(method='bfill', limit=len(df_data), inplace=True)
        df_raw.fillna(0, inplace=True)
        df_value = df_data.values

        num_train = int(len(df_value) * 0.7)
        num_test = int(len(df_value) * 0.2)
        num_vali = len(df_value) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_value) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_value)]
        self.border1s = border1s
        self.border2s = border2s
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)

        self.data = data
        self.data_x = data
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        if self.set_type == 0:
            seq_U = self.U_train[index]
            random_U = np.random.rand()
            if random_U < self.aug_p:
                current_e = np.random.rand() * self.aug_e
                seq_U = StiefelT(seq_U, current_e)
        elif self.set_type == 1:
            seq_U = self.U_vali[index]
        else:
            seq_U = self.U_test[index]
        return seq_x, seq_U

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

    def __preprocess__(self):
        self.U_path = ('./Preprocess/' +
                       self.data_path[:-4] + '_' +
                       str(self.input_len) + '_' +
                       str(self.pred_len) + '_' + str(self.U_num))
        if not os.path.exists(self.U_path):
            train_U, vali_U, test_U = [], [], []
            data_train = self.data[self.border1s[0]:self.border2s[0]]
            data_vali = self.data[self.border1s[1]:self.border2s[1]]
            data_test = self.data[self.border1s[2]:self.border2s[2]]
            train_num = data_train.shape[0] - self.input_len - self.pred_len + 1
            vali_num = data_vali.shape[0] - self.input_len - self.pred_len + 1
            test_num = data_test.shape[0] - self.input_len - self.pred_len + 1
            U_num = int(data_train.shape[-1] * self.U_num)

            logger.info('Prepare the U of train subset')
            for i in tqdm(range(train_num)):
                seq_x = data_train[i: i + self.input_len]
                U, S, V = np.linalg.svd(seq_x)
                train_U.append(U[:, :U_num])
            self.U_train = np.stack(train_U, axis=0)
            os.makedirs(self.U_path)
            np.save(os.path.join(self.U_path, 'train_U.npy'), self.U_train)

            logger.info('Prepare the U of vali subset')
            for i in tqdm(range(vali_num)):
                seq_x = data_vali[i: i + self.input_len]
                U, S, V = np.linalg.svd(seq_x)
                vali_U.append(U[:, :U_num])
            self.U_vali = np.stack(vali_U, axis=0)
            np.save(os.path.join(self.U_path, 'vali_U.npy'), self.U_vali)

            logger.info('Prepare the U of test subset')
            for i in tqdm(range(test_num)):
                seq_x = data_test[i: i + self.input_len]
                U, S, V = np.linalg.svd(seq_x)
                test_U.append(U[:, :U_num])
            self.U_test = np.stack(test_U, axis=0)
            np.save(os.path.join(self.U_path, 'test_U.npy'), self.U_test)
        else:
            self.U_train = np.load(os.path.join(self.U_path, 'train_U.npy'))
            self.U_vali = np.load(os.path.join(self.U_path, 'vali_U.npy'))
            self.U_test = np.load(os.path.join(self.U_path, 'test_U.npy'))

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# https://github.com/guoshnBJTU/ASTGNN/tree/main/data
class Dataset_PEMS(Dataset):
    def __init__(self, root_path, data_path='PEMS08.npz',
                 flag='train', size=None, U_num=0.1, aug_p=0.5, aug_e=0.2):
        # size [label_len, pred_len]
        # info
        self.input_len = size[0]
        self.pred_len = size[1]
        self.U_num = U_num
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.aug_p = aug_p
        self.aug_e = aug_e
        self.__read_data__()
        self.__preprocess__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = np.load(os.path.join(self.root_path, self.data_path), allow_pickle=True)
        df_raw = df_raw['data'][:, :, 0]
        df_raw = pd.DataFrame(df_raw)
        df_raw.fillna(method='ffill', limit=len(df_raw), inplace=True)
        df_raw.fillna(method='bfill', limit=len(df_raw), inplace=True)
        df_raw.fillna(0, inplace=True)
        df_value = df_raw.values

        num_train = int(len(df_value) * 0.7)
        num_test = int(len(df_value) * 0.2)
        num_vali = len(df_value) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_value) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_value)]
        self.border1s = border1s
        self.border2s = border2s
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)

        self.data = data
        self.data_x = data
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        if self.set_type == 0:
            seq_U = self.U_train[index]
            random_U = np.random.rand()
            if random_U < self.aug_p:
                current_e = np.random.rand() * self.aug_e
                seq_U = StiefelT(seq_U, current_e)
        elif self.set_type == 1:
            seq_U = self.U_vali[index]
        else:
            seq_U = self.U_test[index]
        return seq_x, seq_U

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

    def __preprocess__(self):
        self.U_path = ('./Preprocess/' +
                       self.data_path[:-4] + '_' +
                       str(self.input_len) + '_' +
                       str(self.pred_len) + '_' + str(self.U_num))
        if not os.path.exists(self.U_path):
            train_U, vali_U, test_U = [], [], []
            data_train = self.data[self.border1s[0]:self.border2s[0]]
            data_vali = self.data[self.border1s[1]:self.border2s[1]]
            data_test = self.data[self.border1s[2]:self.border2s[2]]
            train_num = data_train.shape[0] - self.input_len - self.pred_len + 1
            vali_num = data_vali.shape[0] - self.input_len - self.pred_len + 1
            test_num = data_test.shape[0] - self.input_len - self.pred_len + 1
            U_num = int(data_train.shape[-1] * self.U_num)

            logger.info('Prepare the U of train subset')
            for i in tqdm(range(train_num)):
                seq_x = data_train[i: i + self.input_len: 12]
                U, S, V = np.linalg.svd(seq_x)
                train_U.append(U[:, :U_num])
            self.U_train = np.stack(train_U, axis=0)
            os.makedirs(self.U_path)
            np.save(os.path.join(self.U_path, 'train_U.npy'), self.U_train)

            logger.info('Prepare the U of vali subset')
            for i in tqdm(range(vali_num)):
                seq_x = data_vali[i: i + self.input_len: 12]
                U, S, V = np.linalg.svd(seq_x)
                vali_U.append(U[:, :U_num])
            self.U_vali = np.stack(vali_U, axis=0)
            np.save(os.path.join(self.U_path, 'vali_U.npy'), self.U_vali)

            logger.info('Prepare the U of test subset')
            for i in tqdm(range(test_num)):
                seq_x = data_test[i: i + self.input_len: 12]
                U, S, V = np.linalg.svd(seq_x)
                test_U.append(U[:, :U_num])
            self.U_test = np.stack(test_U, axis=0)
            np.save(os.path.join(self.U_path, 'test_U.npy'), self.U_test)
        else:
            self.U_train = np.load(os.path.join(self.U_path, 'train_U.npy'))
            self.U_vali = np.load(os.path.join(self.U_path, 'vali_U.npy'))
            self.U_test = np.load(os.path.join(self.U_path, 'test_U.npy'))

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# https://github.com/liuxu77/LargeST
class Dataset_CA_D5(Dataset):
    def __init__(self, root_path, data_path='CA-D5.npy',
                 flag='train', size=None, U_num=0.1, aug_p=0.5, aug_e=0.2):
        # size [label_len, pred_len]
        # info
        self.input_len = size[0]
        self.pred_len = size[1]
        self.U_num = U_num
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.aug_p = aug_p
        self.aug_e = aug_e
        self.__read_data__()
        self.__preprocess__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = np.load(os.path.join(self.root_path, self.data_path), allow_pickle=True)
        df_raw = pd.DataFrame(df_raw)
        df_raw.fillna(method='ffill', limit=len(df_raw), inplace=True)
        df_raw.fillna(method='bfill', limit=len(df_raw), inplace=True)
        df_raw.fillna(0, inplace=True)
        df_value = df_raw.values

        num_train = int(len(df_value) * 0.7)
        num_test = int(len(df_value) * 0.2)
        num_vali = len(df_value) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_value) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_value)]
        self.border1s = border1s
        self.border2s = border2s
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)

        self.data = data
        self.data_x = data
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        if self.set_type == 0:
            seq_U = self.U_train[index]
            random_U = np.random.rand()
            if random_U < self.aug_p:
                current_e = np.random.rand() * self.aug_e
                seq_U = StiefelT(seq_U, current_e)
        elif self.set_type == 1:
            seq_U = self.U_vali[index]
        else:
            seq_U = self.U_test[index]
        return seq_x, seq_U

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

    def __preprocess__(self):
        self.U_path = ('./Preprocess/' +
                       self.data_path[:-4] + '_' +
                       str(self.input_len) + '_' +
                       str(self.pred_len) + '_' + str(self.U_num))
        if not os.path.exists(self.U_path):

            train_U, vali_U, test_U = [], [], []
            data_train = self.data[self.border1s[0]:self.border2s[0]]
            data_vali = self.data[self.border1s[1]:self.border2s[1]]
            data_test = self.data[self.border1s[2]:self.border2s[2]]
            train_num = data_train.shape[0] - self.input_len - self.pred_len + 1
            vali_num = data_vali.shape[0] - self.input_len - self.pred_len + 1
            test_num = data_test.shape[0] - self.input_len - self.pred_len + 1
            U_num = int(data_train.shape[-1] * self.U_num)

            logger.info('Prepare the U of train subset')
            for i in tqdm(range(train_num)):
                seq_x = data_train[i: i + self.input_len: 12]
                U, S, V = np.linalg.svd(seq_x)
                train_U.append(U[:, :U_num])
            self.U_train = np.stack(train_U, axis=0)
            os.makedirs(self.U_path)
            np.save(os.path.join(self.U_path, 'train_U.npy'), self.U_train)

            logger.info('Prepare the U of vali subset')
            for i in tqdm(range(vali_num)):
                seq_x = data_vali[i: i + self.input_len: 12]
                U, S, V = np.linalg.svd(seq_x)
                vali_U.append(U[:, :U_num])
            self.U_vali = np.stack(vali_U, axis=0)
            np.save(os.path.join(self.U_path, 'vali_U.npy'), self.U_vali)

            logger.info('Prepare the U of test subset')
            for i in tqdm(range(test_num)):
                seq_x = data_test[i: i + self.input_len: 12]
                U, S, V = np.linalg.svd(seq_x)
                test_U.append(U[:, :U_num])
            self.U_test = np.stack(test_U, axis=0)
            np.save(os.path.join(self.U_path, 'test_U.npy'), self.U_test)
        else:
            self.U_train = np.load(os.path.join(self.U_path, 'train_U.npy'))
            self.U_vali = np.load(os.path.join(self.U_path, 'vali_U.npy'))
            self.U_test = np.load(os.path.join(self.U_path, 'test_U.npy'))

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
