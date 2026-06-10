import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, weights, biases, dW, dB):
        for i in range(len(weights)):
            weights[i] -= self.lr * dW[i]
            biases[i]  -= self.lr * dB[i]


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr      = learning_rate
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.t       = 0
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None

    def update(self, weights, biases, dW, dB):
        # Inicializa momentos na primeira chamada
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

        self.t += 1

        for i in range(len(weights)):
            # Média móvel dos gradientes (momento 1)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * dB[i]

            # Média móvel dos gradientes ao quadrado (momento 2)
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * dW[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * dB[i]**2

            # Correção de viés
            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

            weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i]  -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

# SGD: subtrai o gradiente direto.
# Adam: adapta o learning rate por peso usando média dos gradientes e dos gradientes²
