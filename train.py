import numpy as np
from tensorflow.keras.datasets import mnist
from mlp.network import MLP
from mlp.optimizers import Adam

# Carrega e prepara o MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test  = X_test.reshape(-1, 784).astype(np.float32) / 255.0

# Treina
model = MLP(layer_sizes=[784, 256, 128, 10], optimizer=Adam(learning_rate=0.001))
model.train(X_train, y_train, epochs=20, batch_size=64)

# Avalia
acc = np.mean(model.predict(X_test) == y_test)
print(f"\nAcurácia final no teste: {acc:.4f}")
