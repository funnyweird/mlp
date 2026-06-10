import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # subtrai o máximo para estabilidade numérica e evita overflow
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# relu: zera valores negativos, mantém positivos
# relu_derivative: retorna 1 onde era positivo, 0 onde era negativo (usada no backprop)
# softmax: converte saída da rede em probabilidades que somam 1