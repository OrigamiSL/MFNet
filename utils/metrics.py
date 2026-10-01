import numpy as np


def MAE(pred, true):
    mae = np.abs(pred - true)
    return np.mean(mae)


def MSE(pred, true):
    mse = (pred - true) ** 2
    return np.mean(mse)


def MAPE(pred, true):
    APE = np.abs((pred - true) / (true + 1e-6)) * 100
    APE = np.where(APE > 500, 0, APE)
    return np.mean(APE)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, mape
