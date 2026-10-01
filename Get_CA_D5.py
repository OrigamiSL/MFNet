import h5py
import pandas as pd
import numpy as np

file_path = './data/CA-D5/ca_his_raw_2017.h5'
f = h5py.File(file_path, 'r')

# print("Objects in the HDF5 file:")
# for name in f:
#     print(name)
#
#     for dataset_name in f[name]:
#         dataset = f[name][dataset_name]
#         print(dataset_name, dataset.shape)
#

CA_D5 = f['t']['block0_values'][:16992, 2832:3043]
df = pd.DataFrame(CA_D5)
CA_D5 = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

print(CA_D5.shape)
np.save('data/CA-D5/CA-D5.npy', CA_D5)
f.close()
