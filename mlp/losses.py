import numpy as np

def cross_entropy_loss(y_pred, y_true):
    # y_pred: probabilidades do softmax (batch x 10)
    # y_true: labels em one-hot (batch x 10)
    n = y_pred.shape[0]
    # Clip evita log(0) que daria infinito
    log_likelihood = -np.log(np.clip(y_pred[range(n), y_true], 1e-12, 1.0))
    return np.mean(log_likelihood)

def cross_entropy_derivative(y_pred, y_true):
    # gradiente da cross-entropy combinada com softmax
    # resultado: (probabilidades - one-hot) / batch_size
    n = y_pred.shape[0]
    grad = y_pred.copy()
    grad[range(n), y_true] -= 1
    return grad / n