import numpy as np
from mlp.activations import relu, relu_derivative, softmax
from mlp.losses import cross_entropy_loss, cross_entropy_derivative

class MLP:
    def __init__(self, layer_sizes, optimizer=None):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.optimizer = optimizer
        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            w = np.random.randn(fan_in, self.layer_sizes[i+1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            self.activations.append(relu(z))

        z_out = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z_out)
        self.activations.append(softmax(z_out))

        return self.activations[-1]

    def backward(self, y_true):
        n_layers = len(self.weights)
        dW = [None] * n_layers
        dB = [None] * n_layers

        delta = cross_entropy_derivative(self.activations[-1], y_true)

        for i in reversed(range(n_layers)):
            dW[i] = self.activations[i].T @ delta
            dB[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                delta = delta @ self.weights[i].T * relu_derivative(self.z_values[i-1])

        return dW, dB

    def train(self, X, y, epochs, batch_size):
        loss_history = []
        acc_history = []
        n = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X, y = X[indices], y[indices]

            for start in range(0, n, batch_size):
                X_batch = X[start:start+batch_size]
                y_batch = y[start:start+batch_size]
                self.forward(X_batch)
                dW, dB = self.backward(y_batch)
                self.optimizer.update(self.weights, self.biases, dW, dB)

            y_pred = self.forward(X)
            loss = cross_entropy_loss(y_pred, y)
            acc = np.mean(np.argmax(y_pred, axis=1) == y)
            loss_history.append(loss)
            acc_history.append(acc)
            print(f"Época {epoch+1}/{epochs} — Loss: {loss:.4f} — Acurácia: {acc:.4f}")

        return loss_history, acc_history

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# __init__: recebe lista de tamanhos ex [784, 256, 128, 10] e o otimizador a usar
# _init_weights: pesos aleatórios com inicialização He (funciona bem com ReLU). Zeros zeraria o aprendizado
# forward: multiplica entrada pelos pesos camada a camada. Ocultas usam ReLU, saída usa Softmax
# backward: calcula gradientes e retorna dW, dB — atualização fica por conta do otimizador
# train: loop de épocas — embaralha dados, divide em batches, roda forward+backward+update em cada um
# predict: roda forward e retorna o dígito com maior probabilidade