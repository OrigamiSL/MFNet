import numpy as np
import torch
from scipy.linalg import expm
from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric


def adjust_learning_rate(optimizer, epoch, args, fine_tuning=False):
    args.learning_rate = args.learning_rate * args.decay
    lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if not fine_tuning:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = -np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def StiefelT(U, perc):
    INJ_RADIUS = 0.89 * np.pi

    dim1, dim2 = U.shape
    st = Stiefel(dim1, dim2)
    st_metric = StiefelCanonicalMetric(dim1, dim2)

    tan_plane_vec = st.random_tangent_vec(U, 1)

    canonical_ip = st_metric.inner_product(tan_plane_vec, tan_plane_vec, U)
    # print('canonical_ip', canonical_ip)
    scaled_tan = tan_plane_vec / np.sqrt(canonical_ip) * perc * INJ_RADIUS

    # Canonical Stiefel exponential (the same block construction as geomstats).
    # Calling scipy.expm directly avoids a read-only view produced by the
    # geomstats 2.5 NumPy backend with some NumPy/SciPy combinations.
    n = dim2
    a = U.T @ scaled_tan
    q, r = np.linalg.qr(scaled_tan - U @ a, mode='reduced')
    block = np.block([[a, -r.T], [r, np.zeros((n, n))]])
    block_exp = expm(np.array(block, copy=True))
    St_mat = U @ block_exp[:n, :n] + q @ block_exp[n:, :n]

    return St_mat
