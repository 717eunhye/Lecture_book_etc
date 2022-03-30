import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time

# -----------------------------------------------------
#  1. 수치 미분을 이용한 심층 신경망 학습
# -----------------------------------------------------
# 유틸리티 함수
epsilon = 0.0001


def _t(x):
    return np.transpose(x)


def _m(A, B):
    return np.matmul(A, B)


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def mean_squared_error(h, y):
    return 1/2 * np.mean(np.square(h-y))


# 뉴런 구현
class Neuron:
    def __init__(self, W, b, a):
        # Model Parameter
        self.W = W
        self.b = b
        self.a = a

        # Gradients
        # 특정 array와 같은 사이즈, 크기의 zeors arrya를 구할때 사용
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def __call__(self, x):
        return self.a(_m(_t(self.W), x)+self.b)  # activation(W^T)x +b


# 심층신경망 구현
class DNN:
    def __init__(self, hidden_depth, num_neuron, num_input, num_output, activation=sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # First hidden layer
        W, b = init_var(num_input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))

        # Hidden layers
        for _ in range(hidden_depth - 1):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # Output layer
        W, b = init_var(num_neuron, num_output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    # Gradient
    def calc_gradient(self, x, y, loss_func):
        def get_new_sequence(layer_index, nuw_neuron):
            new_sequence = list()
            for i, layer in enumerate(self.sequence):
                if i == layer_index:
                    new_sequence.append(new_neuron)
                else:
                    new_sequence.append(layer)
            return new_sequence

        def eval_sequence(x, sequence):
            for layer in sequence:
                x = layer(x)
            return x

        loss = loss_func(self(x), y)  # self(x) :  self.__init__과 동일

        # 아래는 비효율적인 루프
        for layer_id, layer in enumerate(self.sequence):  # iterate layer
            for w_i, w in enumerate(layer.W):  # iterate W(row)
                for w_j, ww in enumerate(w):  # iterate W(col)
                    W = np.copy(layer.W)
                    W[w_i][w_j] = ww + epsilon

                    new_neuron = Neuron(W, layer.b, layer.a)
                    new_seq = get_new_sequence(layer_id, new_neuron)
                    h = eval_sequence(x, new_seq)

                    num_grad = (loss_func(h, y)-loss) / \
                        epsilon  # (f(x+eps)-f(x)) / epsilon

                    layer.dW[w_i][w_j] = num_grad

                for b_i, bb in enumerate(layer.b):
                    b = np.copy(layer.b)
                    b[b_i] = bb + epsilon

                    new_neuron = Neuron(layer.W, b, layer.a)
                    new_seq = get_new_sequence(layer_id, new_neuron)
                    h = eval_sequence(x, new_seq)

                    num_grad = (loss_func(h, y)-loss) / \
                        epsilon  # (f(x+eps)-f(x)) / epsilon

                    layer.db[b_i] = num_grad
            return loss


# 경사하강 학습법
def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = network.calc_gradient(x, y, loss_obj)
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss


# 동작 테스트
x = np.random.normal(0.0, 1.0, (10, ))
y = np.random.normal(0.0, 1.0, (2,))

dnn = DNN(hidden_depth=5, num_neuron=32, num_input=10,
          num_output=2, activation=sigmoid)

t = time.time()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, mean_squared_error, 0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))


# -----------------------------------------------------
#  2. 역전파 학습법을 이용한 심층 신경망 학습
# -----------------------------------------------------
# 유틸리티 함수
def _t(x):
    return np.transpose(x)


def _m(A, B):
    return np.matmul(A, B)

# 시그모이드 구현


class Sigmoid:
    def __init__(self):
        self.last_o = 1

    def __call__(self, x):
        self.last_o = 1 / (1.0 + np.exp(-x))
        return self.last_o

    def grad(self):  # sigmoid(x)(1-sigmoid(x))
        return self.last_o * (1 - self.last_o)


# MSE 구현
class MeanSquaredError:
    def __init__(self):
        # gradient
        self.dh = 1
        self.last_diff = 1

    def __call__(self, h, y):  # 1/2 * mean((h-y)^2)
        self.lass_diff = h-y
        return 1/2 * np.mean(np.square(h-y))

    def grad(self):  # h-y
        return self.lass_diff


# 뉴런 구현
class Neuron:
    def __init__(self, W, b, a_obj):
        # Model parameters
        self.W = W
        self.b = b
        self.a = a_obj()

        # gradient
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dh = np.zeros_like(_t(self.W))

        self.last_x = np.zeros((self.W.shape[0]))
        self.last_h = np.zeros((self.W.shape[1]))

    def __call__(self, x):
        self.last_x = x
        self.last_h = _m(_t(self.W), x) + self.b
        return self.a(self.last_h)

    def grad(self):  # dy/dh = W
        return self.W * self.a.grad()

    def grad_W(self, dh):  # dh : 누적된 gradient
        grad = np.ones_like(self.W)
        grad_a = self.a.grad()
        for j in range(grad.shape[1]):  # dy/dw =x
            grad[:, j] = dh[j] * grad_a[j] * self.last_x
        return grad

    def grad_b(self, dh):  # dy/dh =1
        return dh * self.a.grad()


# 심층신경망 구현
class DNN:
    def __init__(self, hidden_depth, num_neuron, input, output, activation=Sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # First hidden layer
        W, b = init_var(input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))

        # Hidden Layers
        for index in range(hidden_depth):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # Output Layer
        W, b = init_var(num_neuron, output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    def calc_gradient(self, loss_obj):
        loss_obj.dh = loss_obj.grad()
        self.sequence.append(loss_obj)

        # back-prop loop
        for i in range(len(self.sequence) - 1, 0, -1):
            l1 = self.sequence[i]
            l0 = self.sequence[i - 1]

            l0.dh = _m(l0.grad(), l1.dh)
            l0.dW = l0.grad_W(l1.dh)
            l0.db = l0.grad_b(l1.dh)

        self.sequence.remove(loss_obj)

# 경사하강 학습법


def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = loss_obj(network(x), y)  # Forward inference
    network.calc_gradient(loss_obj)  # Back-propagation
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss


# 동작 테스트
x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

t = time.time()
dnn = DNN(hidden_depth=5, num_neuron=32,
          input=10, output=2, activation=Sigmoid)
loss_obj = MeanSquaredError()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, loss_obj, alpha=0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))
