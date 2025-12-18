import math

import numpy as np
import pandas as pd


def update_metrics(pred, label, statistics, target_variate=None):
    if isinstance(pred, tuple):
        pred = pred[0]
    if target_variate is not None:
        pred = pred[:, :, target_variate]
        if label.dim() == 3:
            label = label[:, :, target_variate]

    balance = pred - label
    # statistics['all_preds'].append(pred)
    statistics['y_sum'] += label.abs().sum().item()
    statistics['total'] += len(label.view(-1))
    statistics['MAE'] += balance.abs().sum().item()
    statistics['MSE'] += (balance ** 2).sum().item()
    
    # 为RSE和R2计算累积true的平方和和true的和
    if 'y_sum_sq' not in statistics:
        statistics['y_sum_sq'] = 0
    if 'y_sum_val' not in statistics:
        statistics['y_sum_val'] = 0
    statistics['y_sum_sq'] += (label ** 2).sum().item()
    statistics['y_sum_val'] += label.sum().item()
    
    # 为MAPE计算累积
    if 'MAPE_sum' not in statistics:
        statistics['MAPE_sum'] = 0
    # 避免除以0，添加小的epsilon
    epsilon = 1e-8
    mape_contrib = (balance.abs() / (label.abs() + epsilon)).sum().item()
    statistics['MAPE_sum'] += mape_contrib
    # RRSE += (balance ** 2).sum()
    # x2_sum += (target_batch ** 2).sum()
    # x_sum += target_batch.sum()


def calculate_metrics(statistics):
    MSE, MAE, total, y_sum = statistics['MSE'], statistics['MAE'], statistics['total'], statistics['y_sum']
    metrics = {'MSE': MSE / total, 'MAE': MAE / total}
    
    # 计算RMSE
    metrics['RMSE'] = math.sqrt(MSE / total)
    
    # 计算RSE和R2
    if 'y_sum_sq' in statistics and 'y_sum_val' in statistics:
        y_mean = statistics['y_sum_val'] / total
        y_var = statistics['y_sum_sq'] / total - y_mean ** 2
        if y_var > 1e-10:  # 避免除以0
            # RSE = sqrt(sum((pred - true)^2)) / sqrt(sum((true - true_mean)^2))
            metrics['RSE'] = math.sqrt(MSE / total) / math.sqrt(y_var)
            # R2 = 1 - sum((true - pred)^2) / sum((true - true_mean)^2)
            metrics['R2'] = 1 - (MSE / total) / y_var
        else:
            metrics['RSE'] = float('inf') if MSE > 0 else 0.0
            metrics['R2'] = 0.0
    else:
        metrics['RSE'] = 0.0
        metrics['R2'] = 0.0
    
    # 计算MAPE
    if 'MAPE_sum' in statistics:
        metrics['MAPE'] = statistics['MAPE_sum'] / total
    else:
        metrics['MAPE'] = 0.0
    
    # metrics['NMAE'] = MAE / y_sum
    # metrics['NRMSE'] = math.sqrt((MSE / total)) / (y_sum / total)
    # var = x2_sum / total - (x_sum / total) ** 2
    # RRSE = math.sqrt(RRSE.item() / total) / var.item()
    return metrics

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


def calc_ic(pred=None, label=None, index=None, df=None, return_type='all', reduction='sum'):
    if df is None:
        if isinstance(pred, tuple):
            pred = pred[0]
        df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
    if index is None:
        res = []
        if return_type != 'ric':
            res.append(df['pred'].corr(df['label']))
        if return_type != 'ic':
            res.append(df['pred'].corr(df['label'], method='spearman'))
        return res
    else:
        groups = df.groupby('datetime')
        res = []
        if return_type != 'ric':
            res.append(groups.apply(lambda df: df["pred"].corr(df["label"], method="pearson")))
        if return_type != 'ic':
            res.append(groups.apply(lambda df: df["pred"].corr(df["label"], method="spearman")))
        if reduction == 'sum':
            return [r.sum() for r in res] + [len(groups)]
        elif reduction == 'mean':
            return [r.mean() for r in res]
        else:
            return [r.to_numpy().tolist() for r in res]
