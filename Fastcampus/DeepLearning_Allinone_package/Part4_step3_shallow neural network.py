import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------
#  1. shallow neural network
# -----------------------------------------------------
# sigmoid


def sigmoid(x):
    return 1/(1+np.exp(-x))

# softmax


def softmax(x):
    e_x = np.exp(x)
    return e_x/np.sum(e_x)

# Define network architecture


class ShallowNN:
    def __init__(self, num_input, num_hidden, num_output):
        self.W_h = np.zeros((num_hidden, num_input), dtype=np.float32)
        self.b_h = np.zeros((num_hidden,), dtype=np.float32)
        self.W_o = np.zeros((num_output, num_hidden), dtype=np.float32)
        self.b_o = np.zeros((num_output,), dtype=np.float32)

    def __call__(self, x):
        h = sigmoid(np.matmul(self.W_h, x) + self.b_h)
        return softmax(np.matmul(self.W_o, h) + self.b_o)


# Import and organize dataset
data_path = 'C:/Users/USER/Desktop/todo/fastcam_programmer/Practice/Fastcam/data/DeepLearning_Allinone/'
dataset = np.load(data_path + 'ch2_dataset.npz')
inputs = dataset['inputs']
labels = dataset['labels']


# Create Model
model = ShallowNN(2, 128, 10)

# 사전에 학습된 파라미터 불러오기
weights = np.load(data_path+'ch2_parameters.npz')
model.W_h = weights['W_h']
model.b_h = weights['b_h']
model.W_o = weights['W_o']
model.b_h = weights['b_h']

# 모델 구동 및 결과 프린터
outputs = list()
for pt, label in zip(inputs, labels):
    output = model(pt)
    outputs.append(np.argmax(output))
    print(np.argmax(output), label)

outputs = np.stack(outputs, axis=0)

# 정답 클래스 스캐터 플랏
plt.figure()
for idx in range(10):
    mask = labels == idx
    plt.scatter(inputs[mask, 0], inputs[mask, 1])
plt.title('true_label')
plt.show()

# 모델 출력 클래스 스캐터 플랏
plt.figure()
for idx in range(10):
    mask = outputs == idx
    plt.scatter(inputs[mask, 0], inputs[mask, 1])
plt.title('model_output')
plt.show()


# -----------------------------------------------------
#  2. Gradient shallow neural network
# -----------------------------------------------------
EPOCHS = 1000

# 구조 정의
# 입력 계층 : 2, 은닉 계층 : 128(sigmoid activation), 출력 계층 : 10(soft actication)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, input_dim=2, activation='sigmoid')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=None, mask=None):
        x = self.d1(x)
        return self.d2(x)

# 학습 루프 정의


@tf.function  # train_step 함수 안의 내용이 최적화됨
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_metric):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    # loss를 model.trainable_variables로 미분한 값을 gradients 넣음.
    # df(x)/dx

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_metric(labels, predictions)


# 데이터셋 생성, 전처리
np.random.seed(0)

pts = list()  # 입력값 저장됨
labels = list()
center_pts = np.random.uniform(-8.0, 8.0, (10, 2))  # float64

for label, center_pt in enumerate(center_pts):
    for _ in range(100):
        pts.append(center_pt + np.random.randn(*center_pt.shape))
        labels.append(label)

pts = np.stack(pts, axis=0).astype(np.float32)  # GPU의 경우 float32로 사용
labels = np.stack(labels, axis=0)

train_ds = tf.data.Dataset.from_tensor_slices(
    (pts, labels)).shuffle(1000).batch(32)

# 모델 생성
model = MyModel()

# 손실 함수 및 최적화 알고리즘 설정(crossEntropy, Adam optimizer)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# 평가 지표(Accuracy)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

# 학습 루프
for epoch in range(EPOCHS):
    for x, label in train_ds:
        train_step(model, x, label, loss_object,
                   optimizer, train_loss, train_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()

# 데이터 셋 및 학습 파라미터 저장
data_path = 'C:/Users/USER/Desktop/todo/fastcam_programmer/Practice/Fastcam/data/DeepLearning_Allinone/'
np.savez_compressed(data_path+'ch2_datast.npz', inputs=pts, labels=labels)

W_h, b_h = model.d1.get_weights()
W_o, b_o = model.d2.get_weights()
W_h = np.transpose(W_h)
W_o = np.transpose(W_o)
np.savez_compressed(data_path+'ch2_parameters.npz',
                    W_h=W_h,
                    b_h=b_h,
                    W_o=W_o,
                    b_o=b_o)
