import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlp.activations import relu, relu_derivative, softmax

def test_relu_positivo():
    assert relu(3.0) == 3.0

def test_relu_negativo():
    assert relu(-2.0) == 0.0

def test_relu_zero():
    assert relu(0.0) == 0.0

def test_relu_derivative_positivo():
    assert relu_derivative(np.array([2.0])) == 1.0

def test_relu_derivative_negativo():
    assert relu_derivative(np.array([-1.0])) == 0.0

def test_softmax_soma_um():
    z = np.array([[1.0, 2.0, 3.0]])
    resultado = softmax(z)
    assert np.isclose(np.sum(resultado), 1.0)

def test_softmax_valores_entre_0_e_1():
    z = np.array([[1.0, 2.0, 3.0]])
    resultado = softmax(z)
    assert np.all(resultado >= 0) and np.all(resultado <= 1)

#Cada função testa um comportamento específico das ativações
#test_relu_*: verifica os três casos possíveis da ReLU (positivo, negativo, zero)
#ReLU: se o valor é número negativo, retorna 0; caso contrário, retorna o próprio número. 
#relu_derivative: retorna 1 para valores positivos e 0 para valores negativos ou zero.
#softmax: transforma os valores da última camada em probabilidades. somam 100%