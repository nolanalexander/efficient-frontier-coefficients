import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import hmean

def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))

def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def mad(y, y_pred):
    return np.max(np.abs(y - y_pred))

def aic(y, y_pred, k):
    sse = sum((y - y_pred)**2)
    return 2 * k - 2 * np.log(sse)

def aicc(y, y_pred, k, n):
    return aic(y, y_pred, k) + 2 * k * (k + 1) / (n - k - 1)

def bic(y, y_pred, k, n):
    sse = sum((y - y_pred)**2)
    return k * np.log(n) + n * np.log(sse/n)

def ppv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0

def min_ppv_npv(y_true, y_pred):
    return min(ppv(y_true, y_pred), npv(y_true, y_pred))

def hmean_ppv_npv(y_true, y_pred):
    return hmean([ppv(y_true, y_pred), npv(y_true, y_pred)])